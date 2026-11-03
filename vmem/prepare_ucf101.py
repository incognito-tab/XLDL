"""Extract the project's UCF101 classes and write a clip metadata file.

Run from the repository root:
    python -m vmem.prepare_ucf101 --root data/ucf101

Input:  <root>/UCF-101.zip and <root>/splits/ucfTrainTestlist/ (official split).
Output: <root>/videos/<Class>/<clip>.avi and <root>/metadata.json.

Clips are split by UCF101 group, never by clip, because clips from one group
come from the same source video. Test is official split 1 (groups 1-7).
Validation takes the first groups of official train (see VAL_GROUPS).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor

from vmem.classes import CLASS_NAMES, CLASS_TO_INDEX

VAL_GROUPS = {8, 9, 10}
GROUP_RE = re.compile(r"_g(\d+)_c\d+\.avi$")


def group_of(filename):
    return int(GROUP_RE.search(filename).group(1))


def read_split(path):
    with open(path) as handle:
        return {os.path.basename(line.split()[0]) for line in handle if line.strip()}


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames:format=duration",
         "-of", "json", path],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    stream = info["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(num) / float(den),
        "frames": int(stream.get("nb_frames") or 0),
        "duration": float(info["format"]["duration"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/ucf101")
    parser.add_argument("--workers", default=8, type=int)
    args = parser.parse_args()

    list_dir = os.path.join(args.root, "splits", "ucfTrainTestlist")
    train = read_split(os.path.join(list_dir, "trainlist01.txt"))
    test = read_split(os.path.join(list_dir, "testlist01.txt"))
    video_dir = os.path.join(args.root, "videos")

    clips = []
    with zipfile.ZipFile(os.path.join(args.root, "UCF-101.zip")) as archive:
        for info in archive.infolist():
            parts = info.filename.split("/")
            if info.is_dir() or len(parts) != 3 or parts[1] not in CLASS_TO_INDEX:
                continue
            name = parts[2]
            target = os.path.join(video_dir, parts[1], name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if not (os.path.exists(target) and os.path.getsize(target) == info.file_size):
                with archive.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)  # reading to the end checks the CRC
            if name in test:
                split = "test"
            elif name in train:
                split = "val" if group_of(name) in VAL_GROUPS else "train"
            else:
                raise ValueError(f"{name} is in neither official list")
            clips.append({
                "path": f"{parts[1]}/{name}",
                "label": CLASS_TO_INDEX[parts[1]],
                "group": group_of(name),
                "split": split,
            })

    with ThreadPoolExecutor(args.workers) as pool:
        probes = pool.map(lambda c: probe(os.path.join(video_dir, c["path"])), clips)
        for clip, result in zip(clips, probes):
            clip.update(result)

    clips.sort(key=lambda c: c["path"])
    with open(os.path.join(args.root, "metadata.json"), "w") as handle:
        json.dump({"classes": CLASS_NAMES, "clips": clips}, handle, indent=0)
    print(f"Wrote {len(clips)} clips")


if __name__ == "__main__":
    main()
