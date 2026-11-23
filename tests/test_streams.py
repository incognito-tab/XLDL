"""Leak and correctness checks for the streams.

Run from the repository root: python -m unittest tests.test_streams
"""

import collections
import json
import os
import unittest

import numpy as np

from vmem import streams as S

ROOT = os.environ.get("UCF101_ROOT", "data/ucf101")
N = 2000


def frame_labels(timeline, seen_frames):
    """Independent view of the stream: one label per frame seen so far."""
    labels = []
    for span in timeline:
        labels += [span.label] * (span.end - span.start)
    return labels[:seen_frames]


def runs(labels):
    """Group a frame list into (label, start, end) activities."""
    output = []
    for index, label in enumerate(labels):
        if not output or output[-1][0] != label:
            output.append([label, index, index + 1])
        else:
            output[-1][2] = index + 1
    return output


def oracle_answer(q, timeline):
    """Recompute the answer from frame labels, without using the question's evidence."""
    labels = frame_labels(timeline, q.seen_frames)
    if q.kind == S.WHAT_WAS_HAPPENING:
        return labels[q.seen_frames - 1 - q.delay_seconds * S.FPS]
    if q.kind == S.WHAT_AT_TIME:
        return labels[q.moment_seconds * S.FPS]
    if q.kind == S.WHICH_APPEARED:
        present = [option for option in q.options if option in labels]
        assert len(present) == 1, "exactly one option must have appeared"
        return present[0]
    activities = runs(labels)
    latest = max(i for i, run in enumerate(activities) if run[0] == q.class_x)
    return activities[latest + 1][0]


class StreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, "metadata.json")) as handle:
            cls.clips = {c["path"]: c for c in json.load(handle)["clips"]}
        cls.data = {s: S.StreamDataset(ROOT, s) for s in ("train", "val", "test")}
        cls.samples = {s: [d.sample(i) for i in range(N)] for s, d in cls.data.items()}

    def all_questions(self):
        for sample in self.samples["test"]:
            for q in sample.questions:
                yield sample, q

    def test_stream_uses_only_its_own_split(self):
        for split, samples in self.samples.items():
            for sample in samples:
                for segment in sample.segments:
                    self.assertEqual(self.clips[segment.path]["split"], split)

    def test_groups_do_not_cross_splits(self):
        groups = collections.defaultdict(set)
        for clip in self.clips.values():
            groups[clip["split"]].add(clip["group"])
        self.assertFalse(groups["test"] & groups["train"])
        self.assertFalse(groups["test"] & groups["val"])
        self.assertFalse(groups["val"] & groups["train"])

    def test_deterministic(self):
        again = S.StreamDataset(ROOT, "val")
        for i in range(50):
            self.assertEqual(self.samples["val"][i], again.sample(i))
        self.assertNotEqual(self.samples["val"][1], self.samples["val"][2])
        other_seed = S.StreamDataset(ROOT, "val", seed=1)
        self.assertNotEqual(self.samples["val"][1], other_seed.sample(1))

    def test_timeline_fills_the_stream_with_random_lengths(self):
        lengths = collections.Counter()
        for samples in self.samples.values():
            for sample in samples:
                position = 0
                for segment in sample.segments:
                    self.assertEqual(segment.first_frame, position)
                    position += segment.num_frames
                    lengths[segment.num_frames] += 1
                    self.assertGreaterEqual(segment.num_frames, S.MIN_SEGMENT_FRAMES)
                    self.assertLessEqual(segment.num_frames, S.MAX_SEGMENT_FRAMES)
                self.assertEqual(position, S.STREAM_FRAMES)
        self.assertEqual(sorted(lengths), list(range(S.MIN_SEGMENT_FRAMES, S.MAX_SEGMENT_FRAMES + 1)))

    def test_clips_are_long_enough_and_not_reused_in_a_stream(self):
        for samples in self.samples.values():
            for sample in samples:
                paths = [segment.path for segment in sample.segments]
                self.assertEqual(len(paths), len(set(paths)))
                for segment in sample.segments:
                    frames = segment.source_frames()
                    self.assertEqual(len(frames), segment.num_frames)
                    self.assertTrue((np.diff(frames) >= 0).all())
                    self.assertLess(frames[-1], segment.frames)
                    clip = self.clips[segment.path]
                    end = segment.start_seconds + segment.num_frames / S.FPS
                    self.assertLessEqual(end, clip["duration"] + 1e-6)

    def test_activities_can_repeat_but_neighbours_differ(self):
        repeated = 0
        for sample in self.samples["train"]:
            labels = [segment.label for segment in sample.segments]
            self.assertTrue(all(a != b for a, b in zip(labels, labels[1:])))
            repeated += len(set(labels)) < len(labels)
        self.assertGreater(repeated, N * 0.3)

    def test_answers_match_independent_oracle(self):
        for sample, q in self.all_questions():
            self.assertEqual(q.answer, oracle_answer(q, sample.timeline), q)

    def test_every_question_has_real_evidence_in_the_past(self):
        for sample, q in self.all_questions():
            self.assertTrue(q.evidence)
            labels = frame_labels(sample.timeline, q.seen_frames)
            current_label = labels[-1]
            current_start = runs(labels)[-1][1]
            for span in q.evidence:
                self.assertIn(span, sample.timeline)
                self.assertLessEqual(span.end, current_start)
                self.assertEqual(set(labels[span.start:span.end]), {span.label})
            self.assertNotEqual(q.answer, current_label)
            self.assertGreaterEqual(q.age_seconds, 0)
            self.assertTrue(any(span.label == q.answer for span in q.evidence))
            text = S.evidence_text(q)
            self.assertTrue(text)

    def test_moment_asked_is_inside_its_evidence_and_away_from_cuts(self):
        for _, q in self.all_questions():
            if q.kind not in (S.WHAT_WAS_HAPPENING, S.WHAT_AT_TIME):
                continue
            span = q.evidence[0]
            if q.kind == S.WHAT_WAS_HAPPENING:
                moment = q.seen_frames - 1 - q.delay_seconds * S.FPS
            else:
                moment = q.moment_seconds * S.FPS
            self.assertTrue(span.start <= moment < span.end)
            if span.start > 0:
                self.assertGreaterEqual(moment - span.start, S.BOUNDARY_MARGIN_FRAMES)
            self.assertGreaterEqual(span.end - 1 - moment, S.BOUNDARY_MARGIN_FRAMES)
            self.assertLess(moment, q.seen_frames)

    def test_no_question_is_asked_twice_in_a_stream(self):
        for sample, _ in self.all_questions():
            keys = [S._question_key(q) for q in sample.questions]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertEqual(len(keys), self.data["test"].num_questions)

    def test_next_activity_evidence_is_two_adjacent_spans(self):
        for sample, q in self.all_questions():
            if q.kind == S.WHAT_CAME_NEXT:
                first, second = q.evidence
                self.assertEqual(first.label, q.class_x)
                self.assertEqual(first.end, second.start)
                seen = S.prefix_of(sample.timeline, q.seen_frames)
                later = [s for s in seen if s.start >= second.start]
                self.assertNotIn(q.class_x, [s.label for s in later])

    def test_question_mix_and_delays_are_balanced(self):
        kinds = collections.Counter()
        delays = collections.Counter()
        moments = collections.Counter()
        options_position = collections.Counter()
        for sample in self.samples["train"]:
            for q in sample.questions:
                kinds[q.kind] += 1
                if q.kind == S.WHAT_WAS_HAPPENING:
                    delays[q.delay_seconds] += 1
                if q.kind == S.WHAT_AT_TIME:
                    moments[q.moment_seconds] += 1
                if q.kind == S.WHICH_APPEARED:
                    options_position[q.options.index(q.answer)] += 1
        total = sum(kinds.values())
        for kind in S.QUESTION_TYPES:
            self.assertAlmostEqual(kinds[kind] / total, 1 / len(S.QUESTION_TYPES), delta=0.03)
        self.assertEqual(sorted(delays), list(S.DELAY_CHOICES))
        self.assertEqual(sorted(moments), list(S.MOMENT_CHOICES))
        self.assertLess(max(delays.values()) / min(delays.values()), 3)
        self.assertLess(max(moments.values()) / min(moments.values()), 3)
        self.assertLess(max(options_position.values()) / min(options_position.values()), 1.25)


class FutureTests(unittest.TestCase):
    def test_changing_the_future_does_not_change_the_question(self):
        master = np.random.default_rng(0)
        compared = 0
        for _ in range(3000):
            lengths = [int(master.integers(S.MIN_SEGMENT_FRAMES, 32)) for _ in range(8)]
            labels = master.choice(S.NUM_CLASSES, size=len(lengths), replace=False).tolist()
            start, one = 0, []
            for label, length in zip(labels, lengths):
                one.append(S.Span(label, start, start + length))
                start += length
            cut = int(master.integers(2, len(one)))
            seen_frames = int(master.integers(one[cut].start + 1, one[cut].end + 1))
            # A different future: the current activity runs on, then something else follows.
            other = one[:cut] + [S.Span(one[cut].label, one[cut].start, S.STREAM_FRAMES)]
            prefix_one = S.prefix_of(one, seen_frames)
            prefix_other = S.prefix_of(other, seen_frames)
            self.assertEqual(prefix_one, prefix_other)
            seed = int(master.integers(1 << 30))
            kind = int(master.choice(S.QUESTION_TYPES))
            delay = int(master.choice(S.DELAY_CHOICES))
            moment = int(master.choice(S.MOMENT_CHOICES))
            first = S.make_question(np.random.default_rng(seed), kind, prefix_one, delay, moment)
            second = S.make_question(np.random.default_rng(seed), kind, prefix_other, delay, moment)
            self.assertEqual(first, second)
            compared += first is not None
        self.assertGreater(compared, 500)

    def test_prefix_hides_how_long_the_current_activity_lasts(self):
        timeline = [S.Span(1, 0, 40), S.Span(2, 40, 90), S.Span(3, 90, 256)]
        self.assertEqual(S.prefix_of(timeline, 50), [S.Span(1, 0, 40), S.Span(2, 40, 50)])

    def test_future_activity_can_be_a_wrong_option(self):
        timeline = [S.Span(i, i * 32, (i + 1) * 32) for i in range(8)]
        prefix = S.prefix_of(timeline, 3 * 32 + 1)
        future_used = False
        for seed in range(300):
            q = S.make_question(np.random.default_rng(seed), S.WHICH_APPEARED, prefix)
            wrong = set(q.options) - {q.answer}
            self.assertFalse(wrong & {span.label for span in prefix})
            future_used |= bool(wrong & {4, 5, 6, 7})
        self.assertTrue(future_used)

    def test_question_vector(self):
        span = S.Span(1, 0, 16)
        kinds = [
            (S.Question(S.WHAT_WAS_HAPPENING, 100, 1, (span,), 8.0, delay_seconds=8), 2),
            (S.Question(S.WHAT_AT_TIME, 100, 1, (span,), 8.0, moment_seconds=0), 2),
            (S.Question(S.WHAT_CAME_NEXT, 100, 1, (span,), 8.0, class_x=3), 2),
            (S.Question(S.WHICH_APPEARED, 100, 1, (span,), 8.0, options=(1, 2, 3, 4)), 5),
        ]
        for question, on_bits in kinds:
            vector = S.encode_question(question)
            self.assertEqual(vector.shape[0], S.QUESTION_VECTOR_SIZE)
            self.assertEqual(int(vector.sum()), on_bits)
            timed = S.encode_question(question, clock=True)
            self.assertEqual(timed.shape[0], S.QUESTION_VECTOR_SIZE + S.CLOCK_SIZE)
            self.assertTrue(np.array_equal(timed[:S.QUESTION_VECTOR_SIZE], vector))
            self.assertEqual(int(timed[S.QUESTION_VECTOR_SIZE:].argmax()), (question.seen_frames - 1) // S.FPS)
            both = S.encode_question(question, clock=True, relative=True)
            self.assertEqual(both.shape[0], S.question_size(True, True))
            self.assertTrue(np.array_equal(both[:len(timed)], timed))
            self.assertEqual(int(both[len(timed):].sum()), int(S.delay_frames(question) >= 0))
        # "8 seconds ago" and "at second T" that points to the same moment share the relative spike.
        ago = S.Question(S.WHAT_WAS_HAPPENING, 97, 1, (span,), 8.0, delay_seconds=8)
        at = S.Question(S.WHAT_AT_TIME, 97, 1, (span,), 8.0, moment_seconds=4)
        self.assertEqual(S.delay_frames(ago), S.delay_frames(at))
        tail = slice(S.question_size(True), None)
        self.assertTrue(np.array_equal(S.encode_question(ago, True, True)[tail],
                                       S.encode_question(at, True, True)[tail]))

    def test_relative_delay_stays_in_range_on_real_questions(self):
        dataset = S.StreamDataset(ROOT, "val")
        for i in range(500):
            for q in dataset.sample(i).questions:
                frames = S.delay_frames(q)
                if q.kind in (S.WHAT_WAS_HAPPENING, S.WHAT_AT_TIME):
                    self.assertTrue(0 < frames < q.seen_frames)
                    self.assertEqual(int(S.encode_question(q, relative=True)[S.QUESTION_VECTOR_SIZE:].sum()), 1)
                else:
                    self.assertEqual(frames, -1)


if __name__ == "__main__":
    unittest.main()
