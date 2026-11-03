"""Build video streams from UCF101 clips and generate questions about their past.

A stream fills STREAM_SECONDS with clips of random length. Activities may repeat,
but two neighbours always differ, so every boundary is a real change of activity.

A question arrives after the model has seen `seen_frames` frames. `make_question`
receives only that visible prefix, cut at the current frame, so it cannot know
what comes next or how long the current activity will last. Every question carries
its answer and its evidence: the exact time spans that prove the answer. Evidence
always lies in finished activities, and the answer never equals the activity on
screen, so answering always needs memory.

Everything is deterministic for a given (split, index, seed).
"""

import json
import os
import zlib
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from vmem.classes import CLASS_NAMES, CLASSES

FPS = 8
STREAM_SECONDS = 32
STREAM_FRAMES = STREAM_SECONDS * FPS
MIN_SEGMENT_FRAMES = 2 * FPS
MAX_SEGMENT_FRAMES = 8 * FPS
BOUNDARY_MARGIN_FRAMES = FPS // 2  # never ask about a moment right next to a cut
NUM_CLASSES = len(CLASS_NAMES)

WHAT_WAS_HAPPENING = 0  # "N seconds ago, what was happening?"
WHAT_AT_TIME = 1        # "At second T, what was happening?"
WHICH_APPEARED = 2      # "Which of these four activities has appeared?"
WHAT_CAME_NEXT = 3      # "What came right after the latest X?"
QUESTION_TYPES = (WHAT_WAS_HAPPENING, WHAT_AT_TIME, WHICH_APPEARED, WHAT_CAME_NEXT)
NUM_OPTIONS = 4
CHANCE = {WHAT_WAS_HAPPENING: 1 / NUM_CLASSES, WHAT_AT_TIME: 1 / NUM_CLASSES,
          WHICH_APPEARED: 1 / NUM_OPTIONS, WHAT_CAME_NEXT: 1 / NUM_CLASSES}

DELAY_CHOICES = tuple(range(1, STREAM_SECONDS - 1))
MOMENT_CHOICES = tuple(range(0, STREAM_SECONDS - 2))
QUESTION_VECTOR_SIZE = (len(QUESTION_TYPES) + len(DELAY_CHOICES) + len(MOMENT_CHOICES)
                        + 2 * NUM_CLASSES)
_MAX_TRIES = 2000
_TIMES_PER_CHOICE = 60


class Span(NamedTuple):
    """One activity on the stream timeline. `end` is exclusive."""
    label: int
    start: int
    end: int


@dataclass(frozen=True)
class Question:
    kind: int
    seen_frames: int        # the model has seen frames 0 .. seen_frames - 1
    answer: int             # always a class index, whatever the question type
    evidence: tuple         # Span values that prove the answer
    age_seconds: float      # time from the end of the newest evidence to the question
    delay_seconds: int = 0  # only for WHAT_WAS_HAPPENING
    moment_seconds: int = -1  # only for WHAT_AT_TIME
    class_x: int = -1       # only for WHAT_CAME_NEXT
    options: tuple = ()     # only for WHICH_APPEARED


@dataclass(frozen=True)
class Segment:
    path: str
    label: int
    start_seconds: float
    fps: float
    frames: int
    first_frame: int  # position in the stream
    num_frames: int

    @property
    def span(self):
        return Span(self.label, self.first_frame, self.first_frame + self.num_frames)

    def source_frames(self):
        """Source frame numbers for this segment, sampled at FPS by time."""
        times = self.start_seconds + np.arange(self.num_frames) / FPS
        return np.minimum(np.round(times * self.fps).astype(int), self.frames - 1)


@dataclass(frozen=True)
class Sample:
    split: str
    index: int
    segments: tuple
    questions: tuple = field(default_factory=tuple)

    @property
    def timeline(self):
        return [segment.span for segment in self.segments]


def prefix_of(timeline, seen_frames):
    """The part of the timeline the model has seen, with the current span cut short."""
    if not 1 <= seen_frames <= timeline[-1].end:
        raise ValueError("seen_frames is outside the stream")
    prefix = [span for span in timeline if span.start < seen_frames]
    last = prefix[-1]
    prefix[-1] = Span(last.label, last.start, min(last.end, seen_frames))
    return prefix


def _age(seen_frames, span):
    return (seen_frames - span.end) / FPS


def _span_at_moment(moment, current, past):
    """The finished span holding this frame, unless it is ambiguous or answerable from the screen."""
    for span in past:
        if span.start <= moment < span.end:
            clear_of_start = span.start == 0 or moment - span.start >= BOUNDARY_MARGIN_FRAMES
            clear_of_end = span.end - 1 - moment >= BOUNDARY_MARGIN_FRAMES
            if clear_of_start and clear_of_end and span.label != current.label:
                return span
            return None
    return None


def make_question(rng, kind, prefix, delay_seconds=0, moment_seconds=-1):
    """Build one question from the visible prefix, or return None if none fits."""
    seen_frames = prefix[-1].end
    current = prefix[-1]
    past = prefix[:-1]
    if not past:
        return None

    if kind == WHAT_WAS_HAPPENING:
        span = _span_at_moment(seen_frames - 1 - delay_seconds * FPS, current, past)
        if span is None:
            return None
        return Question(kind, seen_frames, span.label, (span,), float(delay_seconds),
                        delay_seconds=delay_seconds)

    if kind == WHAT_AT_TIME:
        moment = moment_seconds * FPS
        span = _span_at_moment(moment, current, past)
        if span is None:
            return None
        return Question(kind, seen_frames, span.label, (span,), (seen_frames - 1 - moment) / FPS,
                        moment_seconds=moment_seconds)

    if kind == WHICH_APPEARED:
        candidates = sorted({span.label for span in past} - {current.label})
        unseen = sorted(set(range(NUM_CLASSES)) - {span.label for span in prefix})
        if not candidates or len(unseen) < NUM_OPTIONS - 1:
            return None
        answer = int(rng.choice(candidates))
        distractors = rng.choice(unseen, size=NUM_OPTIONS - 1, replace=False).tolist()
        options = tuple(int(x) for x in rng.permutation([answer] + distractors))
        evidence = tuple(span for span in past if span.label == answer)
        return Question(kind, seen_frames, answer, evidence, _age(seen_frames, evidence[-1]),
                        options=options)

    if kind == WHAT_CAME_NEXT:
        latest = {}
        for position, span in enumerate(past):
            latest[span.label] = position
        usable = []
        for label, position in sorted(latest.items()):
            if label == current.label or position + 1 >= len(past):
                continue
            if past[position + 1].label != current.label:
                usable.append(label)
        if not usable:
            return None
        class_x = int(rng.choice(usable))
        position = latest[class_x]
        evidence = (past[position], past[position + 1])
        return Question(kind, seen_frames, past[position + 1].label, evidence,
                        _age(seen_frames, past[position + 1]), class_x=class_x)

    raise ValueError(f"unknown question kind {kind}")


def _draw_question(rng, timeline):
    """Pick a type and what to ask, then find a moment where it is valid.

    The delay or second is drawn first and kept for many moments, so that all of
    them stay about equally common. Only the visible prefix reaches make_question.
    """
    kind = int(rng.choice(QUESTION_TYPES))
    for _ in range(_MAX_TRIES):
        delay = int(rng.choice(DELAY_CHOICES)) if kind == WHAT_WAS_HAPPENING else 0
        moment = int(rng.choice(MOMENT_CHOICES)) if kind == WHAT_AT_TIME else -1
        lowest = delay * FPS + 1 if kind == WHAT_WAS_HAPPENING else max(moment, 0) * FPS + 1
        for _ in range(_TIMES_PER_CHOICE):
            seen_frames = int(rng.integers(lowest, STREAM_FRAMES + 1))
            question = make_question(rng, kind, prefix_of(timeline, seen_frames), delay, moment)
            if question is not None:
                return question
    raise RuntimeError(f"no valid question of kind {kind} for this stream")


def _question_key(question):
    """Two questions with the same key ask the same thing."""
    return (question.kind, question.delay_seconds, question.moment_seconds,
            question.class_x, question.options)


class StreamDataset:
    """Deterministic streams for one split. Reads metadata only, no pixels."""

    def __init__(self, root, split, seed=0, num_questions=4):
        with open(os.path.join(root, "metadata.json")) as handle:
            metadata = json.load(handle)
        if metadata["classes"] != CLASS_NAMES:
            raise ValueError("metadata.json was built with a different class list")
        self.split = split
        self.seed = seed
        self.num_questions = num_questions
        self.root = root
        self.by_class = {i: [] for i in range(NUM_CLASSES)}
        for clip in metadata["clips"]:
            if clip["split"] == split and clip["duration"] >= MIN_SEGMENT_FRAMES / FPS:
                self.by_class[clip["label"]].append(clip)
        if not all(self.by_class.values()):
            raise ValueError(f"a class has no usable clip in split {split}")

    def _draw_clip(self, rng, needed_frames, previous_label, used_paths):
        """Pick an activity uniformly, then an unused clip of at least needed_frames."""
        for _ in range(_MAX_TRIES):
            label = int(rng.integers(NUM_CLASSES))
            if label == previous_label:
                continue
            options = [clip for clip in self.by_class[label]
                       if clip["duration"] * FPS >= needed_frames and clip["path"] not in used_paths]
            if options:
                return options[int(rng.integers(len(options)))]
        raise RuntimeError(f"no clip of {needed_frames} frames in split {self.split}")

    def _draw_length(self, rng, clip, remaining):
        """Random length this clip can supply, never leaving a remainder too short to use."""
        longest = min(MAX_SEGMENT_FRAMES, int(clip["duration"] * FPS), remaining)
        length = int(rng.integers(MIN_SEGMENT_FRAMES, longest + 1))
        if 0 < remaining - length < MIN_SEGMENT_FRAMES:
            length = remaining - MIN_SEGMENT_FRAMES
        return length

    def sample(self, index):
        split_key = zlib.crc32(self.split.encode())
        rng = np.random.default_rng([self.seed, split_key, index])
        segments = []
        used_paths = set()
        first_frame = 0
        previous_label = -1
        while first_frame < STREAM_FRAMES:
            remaining = STREAM_FRAMES - first_frame
            must_fill = remaining < 2 * MIN_SEGMENT_FRAMES
            needed = remaining if must_fill else MIN_SEGMENT_FRAMES
            clip = self._draw_clip(rng, needed, previous_label, used_paths)
            num_frames = remaining if must_fill else self._draw_length(rng, clip, remaining)
            slack = max(clip["duration"] - num_frames / FPS, 0.0)
            segments.append(Segment(clip["path"], clip["label"], float(rng.uniform(0, slack)),
                                    clip["fps"], clip["frames"], first_frame, num_frames))
            used_paths.add(clip["path"])
            previous_label = clip["label"]
            first_frame += num_frames
        timeline = [segment.span for segment in segments]
        drawn = {}
        for _ in range(_MAX_TRIES):
            if len(drawn) == self.num_questions:
                break
            question = _draw_question(rng, timeline)
            drawn.setdefault(_question_key(question), question)
        questions = sorted(drawn.values(), key=lambda q: q.seen_frames)
        return Sample(self.split, index, tuple(segments), tuple(questions))


def encode_question(question):
    """Binary vector: type, delay, second, X, options. Fed to the network as spikes."""
    vector = np.zeros(QUESTION_VECTOR_SIZE, dtype=np.uint8)
    vector[question.kind] = 1
    offset = len(QUESTION_TYPES)
    if question.delay_seconds:
        vector[offset + DELAY_CHOICES.index(question.delay_seconds)] = 1
    offset += len(DELAY_CHOICES)
    if question.moment_seconds >= 0:
        vector[offset + MOMENT_CHOICES.index(question.moment_seconds)] = 1
    offset += len(MOMENT_CHOICES)
    if question.class_x >= 0:
        vector[offset + question.class_x] = 1
    offset += NUM_CLASSES
    for option in question.options:
        vector[offset + option] = 1
    return vector


def _name(label):
    return CLASSES[CLASS_NAMES[label]]


def _seconds(frames):
    return f"{frames / FPS:.1f}".replace(".", ",")


def question_text(question):
    """Vietnamese wording for the demo."""
    if question.kind == WHAT_WAS_HAPPENING:
        return f"Cách đây {question.delay_seconds} giây đang có hoạt động gì?"
    if question.kind == WHAT_AT_TIME:
        return f"Ở giây thứ {question.moment_seconds} đang xảy ra hoạt động gì?"
    if question.kind == WHICH_APPEARED:
        names = ", ".join(_name(option) for option in question.options)
        return f"Trong các hoạt động {names}, hoạt động nào đã xuất hiện?"
    return f"Ngay sau lần gần nhất có {_name(question.class_x)} là hoạt động gì?"


def answer_text(question):
    return _name(question.answer)


def evidence_text(question):
    """One Vietnamese sentence for each piece of evidence."""
    spans = [f"{_name(s.label)} từ {_seconds(s.start)} đến {_seconds(s.end)} giây"
             for s in question.evidence]
    if question.kind == WHAT_WAS_HAPPENING:
        moment = question.seen_frames - 1 - question.delay_seconds * FPS
        return f"Thời điểm được hỏi là {_seconds(moment)} giây, nằm trong đoạn {spans[0]}."
    if question.kind == WHAT_AT_TIME:
        return f"Giây thứ {question.moment_seconds} nằm trong đoạn {spans[0]}."
    if question.kind == WHICH_APPEARED:
        return "Đã xuất hiện: " + "; ".join(spans) + ". Ba lựa chọn còn lại chưa xuất hiện."
    return f"Lần gần nhất là {spans[0]}. Tiếp ngay sau đó là {spans[1]}."
