"""Build a self-contained local page for browsing the project's data.

Run from the repository root:
    python -m vmem.build_data_viewer --root data/ucf101

Output stays under data/, which git ignores, because the page embeds dataset frames.
"""

import argparse
import base64
import collections
import json
import os

import cv2

from vmem import streams as S
from vmem.classes import CLASS_NAMES, CLASSES

SHOWN_STREAMS = (("test", 3), ("test", 7), ("test", 11), ("val", 1), ("val", 2), ("train", 0))
QUESTIONS_SHOWN = 6
STATS_STREAMS = 3000
SPLIT_NAMES = {"train": "Train", "val": "Validation", "test": "Test"}
FRAME_SIZE = (224, 168)
JPEG_QUALITY = 62
MIN_CLIP_SECONDS = S.MIN_SEGMENT_FRAMES / S.FPS


def to_jpeg(frame, size=FRAME_SIZE, quality=JPEG_QUALITY):
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(data.tobytes()).decode("ascii")


def read_frames(path, numbers):
    """Read the given non-decreasing source frame numbers from one clip."""
    capture = cv2.VideoCapture(path)
    output, position, frame = [], -1, None
    for number in numbers:
        while position < number:
            ok, next_frame = capture.read()
            if not ok:
                break
            frame, position = next_frame, position + 1
        if frame is None:
            raise RuntimeError(f"cannot read {path}")
        output.append(frame)
    capture.release()
    return output


def segment_index(timeline, frame):
    return next(i for i, span in enumerate(timeline) if span.start <= frame < span.end)


def build_stream(root, split, index):
    sample = S.StreamDataset(root, split, num_questions=QUESTIONS_SHOWN).sample(index)
    timeline = sample.timeline
    frames = []
    for segment in sample.segments:
        path = os.path.join(root, "videos", segment.path)
        frames += [to_jpeg(f) for f in read_frames(path, segment.source_frames())]
    names = " · ".join(CLASSES[CLASS_NAMES[span.label]] for span in timeline[:3])
    return {
        "title": f"{SPLIT_NAMES[split]} số {index}: {names}...",
        "frames": frames,
        "segments": [{"vi": CLASSES[CLASS_NAMES[span.label]], "start": span.start, "end": span.end}
                     for span in timeline],
        "questions": [{
            "seen": q.seen_frames,
            "segment": segment_index(timeline, q.seen_frames - 1),
            "text": S.question_text(q),
            "answer": S.answer_text(q),
            "note": S.evidence_text(q),
            "proof": [timeline.index(span) for span in q.evidence],
        } for q in sample.questions],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ucf101")
    args = parser.parse_args()

    with open(os.path.join(args.root, "metadata.json")) as handle:
        clips = json.load(handle)["clips"]

    counts = collections.defaultdict(collections.Counter)
    for clip in clips:
        if clip["duration"] >= MIN_CLIP_SECONDS:
            counts[clip["label"]][clip["split"]] += 1
    classes = []
    for label, name in enumerate(CLASS_NAMES):
        clip = next(c for c in clips
                    if c["label"] == label and c["split"] == "train" and c["duration"] >= 4)
        middle = clip["frames"] // 2
        frame = read_frames(os.path.join(args.root, "videos", clip["path"]), [middle])[0]
        classes.append({"en": name, "vi": CLASSES[name], "thumb": to_jpeg(frame, (240, 180), 70),
                        "train": counts[label]["train"], "val": counts[label]["val"],
                        "test": counts[label]["test"]})

    bins = collections.Counter(int(clip["duration"]) for clip in clips)
    hist = [{"lo": lo, "n": bins[lo]} for lo in range(1, max(bins) + 1)]

    splits = []
    for key, groups in (("train", "g11 đến g25"), ("val", "g08 đến g10"), ("test", "g01 đến g07")):
        members = [c for c in clips if c["split"] == key]
        splits.append({"name": SPLIT_NAMES[key], "groups": groups, "total": len(members),
                       "usable": sum(c["duration"] >= MIN_CLIP_SECONDS for c in members)})

    dataset = S.StreamDataset(args.root, "train")
    kinds_seen = collections.Counter()
    segments_total = 0
    repeated = 0
    for index in range(STATS_STREAMS):
        sample = dataset.sample(index)
        labels = [segment.label for segment in sample.segments]
        segments_total += len(labels)
        repeated += len(set(labels)) < len(labels)
        for question in sample.questions:
            kinds_seen[question.kind] += 1
    total = sum(kinds_seen.values())
    kinds = [
        {"name": "Cách đây N giây đang có hoạt động gì?", "answer": "Một trong 24 hoạt động", "kind": S.WHAT_WAS_HAPPENING},
        {"name": "Ở giây thứ T đang xảy ra hoạt động gì?", "answer": "Một trong 24 hoạt động", "kind": S.WHAT_AT_TIME},
        {"name": "Trong 4 hoạt động này, hoạt động nào đã xuất hiện?", "answer": "Một trong 4 lựa chọn", "kind": S.WHICH_APPEARED},
        {"name": "Ngay sau lần gần nhất có X là hoạt động gì?", "answer": "Một trong 24 hoạt động", "kind": S.WHAT_CAME_NEXT},
    ]
    for row in kinds:
        row["chance"] = f"{S.CHANCE[row['kind']] * 100:.1f}%".replace(".", ",")
        row["share"] = f"{kinds_seen[row['kind']] / total * 100:.1f}%".replace(".", ",")

    average_segments = f"{segments_total / STATS_STREAMS:.1f}".replace(".", ",")
    payload = {
        "tiles": [["24", "hoạt động"], [f"{len(clips):,}".replace(",", "."), "clip video thật"],
                  ["2–8 giây", "độ dài mỗi hoạt động"],
                  [average_segments, "hoạt động trung bình mỗi luồng 32 giây"],
                  [f"{repeated / STATS_STREAMS * 100:.0f}%", "luồng train có hoạt động lặp lại"],
                  ["4", "loại câu hỏi, đều có bằng chứng"]],
        "streams": [build_stream(args.root, split, index) for split, index in SHOWN_STREAMS],
        "classes": classes,
        "hist": hist,
        "splits": splits,
        "kinds": kinds,
    }

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "viewer_template.html"), encoding="utf-8") as handle:
        template = handle.read()
    output_dir = os.path.join(args.root, "viewer")
    os.makedirs(output_dir, exist_ok=True)
    output = os.path.join(output_dir, "index.html")
    text = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(template.replace("__PAYLOAD__", text))
    print(output, round(os.path.getsize(output) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
