"""Local demo: watch a test stream, pause anywhere, ask a question, compare the three models.

Run from the repository root, after downloading the files listed in README to data/demo:
    python -m vmem.demo_server
then open http://127.0.0.1:8000

Everything stays on this machine. The models only ever receive features of frames
up to the moment the question is asked: later frames are zeroed before the call.
"""

import base64
import functools
import glob
import json
import os
import time

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from vmem import streams as S
from vmem.classes import CLASS_NAMES, CLASSES
from vmem.frames import stream_feature_rows
from vmem.memory_models import VideoQA

ROOT = os.environ.get("UCF101_ROOT", "data/ucf101")
DEMO = os.environ.get("DEMO_DIR", "data/demo")
STREAMS_LISTED = 40
FRAME_SIZE = (320, 240)
MODEL_NAMES = {"plus": "Bản nâng cấp", "mixer": "Mixer-SNN", "dmp": "DMP-SNN"}
KIND_ORDER = ("plus", "mixer", "dmp")

torch.set_grad_enabled(False)
app = FastAPI()


def class_name(label):
    return CLASSES[CLASS_NAMES[label]]


def load_models():
    models = {}
    for path in sorted(glob.glob(os.path.join(DEMO, "*.pt"))):
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["config"]
        dim = S.question_size(config.get("clock", False), config.get("relative", False))
        model = VideoQA(config["kind"], FEATURES.shape[1], dim, S.NUM_CLASSES, **config["model"])
        model.load_state_dict(checkpoint["state"])
        model.eval()
        models[config["kind"]] = {"model": model, "config": config,
                                  "params": sum(p.numel() for p in model.parameters())}
    missing = set(KIND_ORDER) - set(models)
    if missing:
        raise SystemExit(f"missing checkpoints in {DEMO} for: {sorted(missing)}")
    return models


FEATURES = np.load(os.path.join(DEMO, "sresnet18_s0_w8.npy"), mmap_mode="r")
with open(os.path.join(DEMO, "index.json")) as handle:
    INDEX = json.load(handle)["clips"]
DATASET = S.StreamDataset(ROOT, "test")
MODELS = load_models()


def results_summary():
    """Test scores of the checkpoints, from the JSON files written by the training runs."""
    rows = {}
    for path in glob.glob(os.path.join("results", "round4", "*", "*.json")):
        with open(path, encoding="utf-8") as handle:
            result = json.load(handle)
        kind = result["config"]["kind"]
        if kind in MODELS and result["config"]["model"] == MODELS[kind]["config"]["model"]:
            rows[kind] = round(result["test"]["mean_acc"] * 100, 1)
    return rows


SCORES = results_summary()


def warm_up():
    """Run each model once so the first question's timing is not inflated by start-up work."""
    rows, steps = torch.tensor([0]), torch.tensor([S.STREAM_FRAMES - 1])
    x = torch.zeros(1, S.STREAM_FRAMES, FEATURES.shape[1])
    for entry in MODELS.values():
        config = entry["config"]
        questions = torch.zeros(1, S.question_size(config.get("clock", False), config.get("relative", False)))
        for _ in range(2):
            entry["model"](x, questions, rows, steps, None, torch.tensor([8]))


warm_up()


def read_frames(path, numbers):
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


def to_jpeg(frame):
    small = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return base64.b64encode(data.tobytes()).decode("ascii")


@functools.lru_cache(maxsize=8)
def stream(index):
    sample = DATASET.sample(index)
    rows = np.concatenate([stream_feature_rows(segment, INDEX) for segment in sample.segments])
    features = torch.from_numpy(np.asarray(FEATURES[rows], dtype=np.float32))
    return sample, features


@functools.lru_cache(maxsize=8)
def stream_frames(index):
    sample, _ = stream(index)
    frames = []
    for segment in sample.segments:
        path = os.path.join(ROOT, "videos", segment.path)
        frames += [to_jpeg(f) for f in read_frames(path, segment.source_frames())]
    return frames


def spans_json(spans, timeline):
    return [{"index": timeline.index(span), "label": class_name(span.label), "start": span.start,
             "end": span.end} for span in spans]


@app.get("/", response_class=HTMLResponse)
def page():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "demo.html"), encoding="utf-8") as handle:
        return handle.read()


@app.get("/api/info")
def info():
    return {
        "fps": S.FPS, "frames": S.STREAM_FRAMES,
        "classes": [{"id": i, "name": class_name(i)} for i in range(S.NUM_CLASSES)],
        "models": [{"kind": kind, "name": MODEL_NAMES[kind], "params": MODELS[kind]["params"],
                    "test": SCORES.get(kind)} for kind in KIND_ORDER],
        "streams": [{"id": i, "title": f"Luồng test số {i}"} for i in range(STREAMS_LISTED)],
    }


@app.get("/api/stream/{index}")
def get_stream(index: int):
    if not 0 <= index < STREAMS_LISTED:
        raise HTTPException(404, "no such stream")
    sample, _ = stream(index)
    timeline = sample.timeline
    return {
        "frames": stream_frames(index),
        "segments": [{"label": class_name(s.label), "start": s.start, "end": s.end} for s in timeline],
        "presets": [{"seen": q.seen_frames, "kind": q.kind, "delay": q.delay_seconds,
                     "moment": q.moment_seconds, "x": q.class_x, "options": list(q.options),
                     "text": S.question_text(q)} for q in sorted(sample.questions, key=lambda q: q.seen_frames)],
    }


class Ask(BaseModel):
    stream: int
    seen: int                 # frames shown so far, 1..256
    kind: int
    delay: int = 0
    moment: int = -1
    x: int = -1
    options: list[int] = []


REASONS = {
    S.WHAT_WAS_HAPPENING: "Thời điểm đó chưa có, đang là hoạt động hiện tại, hoặc quá sát chỗ chuyển cảnh.",
    S.WHAT_AT_TIME: "Giây đó chưa tới, đang là hoạt động hiện tại, hoặc quá sát chỗ chuyển cảnh.",
    S.WHICH_APPEARED: "Chưa có hoạt động nào kết thúc để hỏi.",
    S.WHAT_CAME_NEXT: "Hoạt động X chưa xuất hiện, hoặc ngay sau nó chưa có hoạt động nào kết thúc.",
}


def build_question(request, timeline):
    prefix = S.prefix_of(timeline, request.seen)
    kind = request.kind
    if kind in (S.WHAT_WAS_HAPPENING, S.WHAT_AT_TIME):
        return S.make_question(np.random.default_rng(0), kind, prefix, request.delay, request.moment)
    if kind == S.WHICH_APPEARED:
        question = S.make_question(np.random.default_rng(request.seen), kind, prefix)
        if question and len(request.options) == S.NUM_OPTIONS:
            present = [o for o in request.options if o in {s.label for s in prefix[:-1]}]
            if len(present) == 1 and present[0] != prefix[-1].label:
                evidence = tuple(s for s in prefix[:-1] if s.label == present[0])
                question = S.Question(kind, request.seen, present[0], evidence,
                                      (request.seen - evidence[-1].end) / S.FPS,
                                      options=tuple(request.options))
        return question
    for seed in range(400):  # make_question picks X itself; find the draw that uses the chosen X
        question = S.make_question(np.random.default_rng(seed), kind, prefix)
        if question is None or request.x < 0 or question.class_x == request.x:
            return question
    return None


@app.post("/api/ask")
def ask(request: Ask):
    if not 0 <= request.stream < STREAMS_LISTED or not 1 <= request.seen <= S.STREAM_FRAMES:
        raise HTTPException(400, "bad stream or time")
    sample, features = stream(request.stream)
    timeline = sample.timeline
    question = build_question(request, timeline)
    if question is None:
        return {"error": REASONS[request.kind]}
    seen = question.seen_frames
    x = features.clone().unsqueeze(0)
    x[:, seen:] = 0  # the models get nothing from frames after the question
    rows, steps = torch.tensor([0]), torch.tensor([seen - 1])
    mask = torch.ones(1, S.NUM_CLASSES, dtype=torch.bool)
    if question.options:
        mask[:] = False
        mask[0, list(question.options)] = True
    answers = []
    for kind in KIND_ORDER:
        entry = MODELS[kind]
        config = entry["config"]
        vector = torch.tensor(S.encode_question(question, config.get("clock", False),
                                                config.get("relative", False)), dtype=torch.float32)[None]
        started = time.perf_counter()
        scores, spikes = entry["model"](x, vector, rows, steps, mask, torch.tensor([S.delay_frames(question)]))
        elapsed = (time.perf_counter() - started) * 1000
        probs = torch.softmax(scores[0], 0)
        top = torch.topk(probs, 3 if not question.options else S.NUM_OPTIONS)
        guess = int(top.indices[0])
        answers.append({
            "kind": kind, "name": MODEL_NAMES[kind], "answer": class_name(guess),
            "correct": guess == question.answer, "ms": round(elapsed, 1),
            "firing": round(float(spikes[:, :seen].float().mean()), 3),
            "top": [{"label": class_name(int(i)), "p": round(float(p), 3)}
                    for p, i in zip(top.values, top.indices)],
        })
    return {
        "question": S.question_text(question), "truth": S.answer_text(question),
        "evidence": S.evidence_text(question), "spans": spans_json(question.evidence, timeline),
        "seen": seen, "answers": answers,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
