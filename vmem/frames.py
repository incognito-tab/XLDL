"""Decode every project clip once at FPS and store small frames for training.

Frame k of a clip is the source frame nearest to time k / FPS, the same rule the
streams use, so stream segments can index this store directly.

Output in <out_dir>:
    frames.npy  uint8 [total_frames, size, size, 3], RGB
    index.json  {"size": size, "clips": {path: [offset, count]}}
"""

import json
import os
from multiprocessing import Pool

import cv2
import numpy as np

from vmem.streams import FPS

FRAME_SIZE = 64


def frame_count(duration):
    return max(int(np.floor(duration * FPS)), 1)


def clip_frame_numbers(duration, fps, frames):
    times = np.arange(frame_count(duration)) / FPS
    return np.minimum(np.round(times * fps).astype(int), frames - 1)


def decode(path, numbers, size):
    """Read the given non-decreasing frame numbers, resized to size x size RGB."""
    capture = cv2.VideoCapture(path)
    output, position, frame = [], -1, None
    for number in numbers:
        while position < number:
            ok, next_frame = capture.read()
            if not ok:
                break  # keep the last good frame if the file ends early
            frame, position = next_frame, position + 1
        if frame is None:
            raise RuntimeError(f"cannot read {path}")
        small = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        output.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    capture.release()
    return np.stack(output)


def _decode_job(job):
    path, numbers, size = job
    return decode(path, numbers, size)


def stream_feature_rows(segment, index):
    """Rows of the frame store (and of feature arrays aligned to it) for one stream segment."""
    offset, count = index[segment.path]
    first = int(round(segment.start_seconds * FPS))
    rows = np.clip(first + np.arange(segment.num_frames), 0, count - 1)
    return offset + rows


def build_frame_store(root, out_dir, size=FRAME_SIZE, workers=8):
    with open(os.path.join(root, "metadata.json")) as handle:
        clips = json.load(handle)["clips"]
    os.makedirs(out_dir, exist_ok=True)
    counts = [frame_count(c["duration"]) for c in clips]
    total = int(sum(counts))
    store = np.lib.format.open_memmap(os.path.join(out_dir, "frames.npy"), mode="w+",
                                      dtype=np.uint8, shape=(total, size, size, 3))
    jobs = [(os.path.join(root, "videos", c["path"]),
             clip_frame_numbers(c["duration"], c["fps"], c["frames"]), size) for c in clips]
    index = {}
    offset = 0
    with Pool(workers) as pool:
        for clip, count, frames in zip(clips, counts, pool.imap(_decode_job, jobs, chunksize=8)):
            assert len(frames) == count
            store[offset:offset + count] = frames
            index[clip["path"]] = [offset, count]
            offset += count
    store.flush()
    with open(os.path.join(out_dir, "index.json"), "w") as handle:
        json.dump({"size": size, "fps": FPS, "clips": index}, handle)
    return total
