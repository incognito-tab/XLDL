"""Run the project's training on Modal.

Each stage is a separate command, run in order:

    modal run modal_app.py::prepare                      # data and 64x64 frames on the volume
    modal run modal_app.py::encoder --size lite          # spiking frame encoder
    modal run modal_app.py::features --encoder lite_s0   # per-frame features for every clip
    modal run modal_app.py::sweep --encoder lite_s0      # 10 settings for each of the 3 models
    modal run modal_app.py::final --encoder lite_s0      # best settings, 3 seeds, test once
    modal run modal_app.py::round4                       # latest results (see README)

Everything is stored on the Modal volume "xldl-data". Result files are also
copied to results/ on this machine.
"""

import json
import os

import modal

app = modal.App("xldl-video-memory")
volume = modal.Volume.from_name("xldl-data", create_if_missing=True)

DATA = "/data"
ROOT = f"{DATA}/ucf101"
FRAMES = f"{DATA}/frames64"
UCF_ZIP = "https://huggingface.co/datasets/quchenyuan/UCF101-ZIP/resolve/main/UCF-101.zip"
# Same bytes as the official crcv.ucf.edu file, whose certificate chain fails inside Modal.
UCF_SPLITS = "https://huggingface.co/datasets/quchenyuan/UCF101-ZIP/resolve/main/UCF101TrainTestSplits-RecognitionTask.zip"
UCF_SPLITS_SHA256 = "5c0d1a53b8ed364a2ac830a73f405e51bece7d98ce1254fd19ed4a36b224bd27"
UCF_ZIP_BYTES = 6957373664
LOCAL_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "curl", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.8.0", "numpy", "scipy", "einops", "ml_collections", "matplotlib",
                 "opencv-python-headless", "spikingjelly==0.0.0.0.14")
    .add_local_dir("vmem", "/root/vmem", ignore=["**/__pycache__"])
    .add_local_dir("models", "/root/models", ignore=["**/__pycache__"])
    .add_local_dir("Dual_memory_pathways/smnist/src", "/root/Dual_memory_pathways/smnist/src",
                   ignore=["**/__pycache__"])
)

# The same training budget for every memory model. Only `model` settings differ.
# Round 1 (lite encoder) ran without feature dropout; round 2 adds it for every model.
MEMORY_BASE = {"epochs": 30, "streams_per_epoch": 4096, "batch": 64, "lr": 2e-3,
               "val_streams": 1024, "test_streams": 2048, "eval_every": 5, "feature_dropout": 0.2}

# Round 2 search, validation only. Ranges come from round 1: Mixer peaked at tau 24 to 32,
# DMP changed little with d and theta, and the upgrade gained most from the memory read.
SWEEP_V2 = (
    [["mixer", {"tau": t}] for t in (8.0, 16.0, 24.0, 32.0, 64.0)]
    + [["dmp", {"memory_size": d, "theta": theta}] for d, theta in ((10, 256), (40, 256), (80, 256), (10, 64))]
    + [["plus", {"channels": c, "memory_size": d, "thetas": [16, 64, 256], "query_read": True}]
       for c, d in ((16, 10), (32, 10), (16, 20), (32, 20))]
)

SWEEP = {
    "mixer": [{"tau": t} for t in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)],
    "dmp": [{"memory_size": d, "theta": theta} for d in (5, 10, 20, 40, 80) for theta in (64, 256)],
    "plus": [
        {"channels": 16, "memory_size": 10, "thetas": [256], "query_read": False},          # H1
        {"channels": 32, "memory_size": 10, "thetas": [256], "query_read": False},          # H1
        {"channels": 16, "memory_size": 10, "thetas": [16, 64, 256], "query_read": False},  # H1 + H2
        {"channels": 16, "memory_size": 20, "thetas": [16, 64, 256], "query_read": False},  # H1 + H2
        {"channels": 32, "memory_size": 10, "thetas": [16, 64, 256], "query_read": False},  # H1 + H2
        {"channels": 16, "memory_size": 10, "thetas": [16, 64, 256], "query_read": True},   # H1 + H2 + H3
        {"channels": 32, "memory_size": 10, "thetas": [16, 64, 256], "query_read": True},
        {"channels": 16, "memory_size": 20, "thetas": [16, 64, 256], "query_read": True},
        {"channels": 16, "memory_size": 10, "thetas": [8, 32, 128, 256], "query_read": True},
        {"channels": 32, "memory_size": 20, "thetas": [16, 64, 256], "query_read": True},
    ],
}


def run_name(kind, settings, seed):
    parts = [kind] + [f"{k}{'-'.join(map(str, v)) if isinstance(v, list) else v}" for k, v in settings.items()]
    return "_".join(parts).replace(".0", "") + f"_s{seed}"


def save_local(name, payload):
    os.makedirs(LOCAL_RESULTS, exist_ok=True)
    with open(os.path.join(LOCAL_RESULTS, name), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)


@app.function(image=image, volumes={DATA: volume}, cpu=8, memory=16384, timeout=2 * 3600)
def prepare_data(size: int = 64):
    """Download UCF101, extract the 24 classes, keep them on the volume, build size x size frames."""
    import hashlib
    import shutil
    import subprocess
    import zipfile

    from vmem.frames import build_frame_store

    work = "/tmp/ucf101"
    os.makedirs(work, exist_ok=True)
    subprocess.run(["curl", "-sSL", "--retry", "5", "-o", f"{work}/UCF-101.zip", UCF_ZIP], check=True)
    if os.path.getsize(f"{work}/UCF-101.zip") != UCF_ZIP_BYTES:
        raise RuntimeError("UCF-101.zip has the wrong size")
    subprocess.run(["curl", "-sSL", "--retry", "5", "-o", f"{work}/splits.zip", UCF_SPLITS], check=True)
    if hashlib.sha256(open(f"{work}/splits.zip", "rb").read()).hexdigest() != UCF_SPLITS_SHA256:
        raise RuntimeError("splits.zip differs from the official file")
    zipfile.ZipFile(f"{work}/splits.zip").extractall(f"{work}/splits")
    subprocess.run(["python", "-m", "vmem.prepare_ucf101", "--root", work, "--workers", "8"],
                   check=True, cwd="/root")
    total = build_frame_store(work, f"{DATA}/frames{size}", size=size, workers=8)
    os.makedirs(ROOT, exist_ok=True)
    shutil.copy(f"{work}/metadata.json", f"{ROOT}/metadata.json")
    if not os.path.exists(f"{ROOT}/videos"):
        shutil.copytree(f"{work}/videos", f"{ROOT}/videos")
    volume.commit()
    digest = hashlib.sha256(open(f"{ROOT}/metadata.json", "rb").read()).hexdigest()
    return {"frames": total, "metadata_sha256": digest}


@app.function(image=image, volumes={DATA: volume}, gpu="L4", memory=32768, timeout=8 * 3600)
def train_encoder_job(size, epochs, seed, image_size=64, lr=2e-3):
    from vmem.train_encoder import train

    volume.reload()
    name = f"{size}_s{seed}"
    os.makedirs(f"{DATA}/encoders", exist_ok=True)
    result = train(ROOT, f"{DATA}/frames{image_size}", f"{DATA}/encoders/{name}.pt", size=size, epochs=epochs,
                   seed=seed, lr=lr, log=lambda line: print(line, flush=True))
    result["name"] = name
    with open(f"{DATA}/encoders/{name}.json", "w") as handle:
        json.dump(result, handle)
    volume.commit()
    return result


@app.function(image=image, volumes={DATA: volume}, gpu="L4", memory=32768, timeout=2 * 3600)
def extract_features_job(encoder, image_size=64, reset_every=None, out_name=None):
    from vmem.train_encoder import extract_features

    volume.reload()
    os.makedirs(f"{DATA}/features", exist_ok=True)
    result = extract_features(f"{DATA}/encoders/{encoder}.pt", ROOT, f"{DATA}/frames{image_size}",
                              f"{DATA}/features/{out_name or encoder}.npy", reset_every=reset_every,
                              log=lambda line: print(line, flush=True))
    volume.commit()
    return result


@app.function(image=image, volumes={DATA: volume}, gpu="T4", memory=16384, timeout=4 * 3600)
def train_memory_job(encoder, config):
    import numpy as np
    import torch

    from vmem.train_memory import train

    volume.reload()
    name = run_name(config["kind"], config["model"], config["seed"])
    if config["epochs"] != 30:
        name += f"_e{config['epochs']}"
    if config.get("clock"):
        name += "_clock"
    if config.get("relative"):
        name += "_rel"
    name += "_final" if config.get("test") else ""
    folder = f"{DATA}/memory/{encoder}"
    done = f"{folder}/{name}.json"
    if os.path.exists(done):  # finished in an earlier, interrupted run: reuse it
        with open(done) as handle:
            return json.load(handle)
    features = torch.from_numpy(np.load(f"{DATA}/features/{encoder}.npy")).cuda()
    with open(f"{FRAMES}/index.json") as handle:  # rows are the same for every frame size
        index = json.load(handle)["clips"]
    output, state = train(ROOT, features, index, config, log=lambda line: print(name, line, flush=True))
    output["name"] = name
    output["encoder"] = encoder
    os.makedirs(folder, exist_ok=True)
    torch.save({"state": state, "config": config}, f"{folder}/{name}.pt")
    with open(f"{folder}/{name}.json", "w") as handle:
        json.dump(output, handle)
    volume.commit()
    return output


@app.local_entrypoint()
def prepare():
    print(json.dumps(prepare_data.remote(), indent=1))


@app.local_entrypoint()
def encoder(size: str = "lite", epochs: int = 40, seed: int = 0):
    result = train_encoder_job.remote(size, epochs, seed)
    save_local(f"encoder_{result['name']}.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=1))


@app.local_entrypoint()
def features(encoder: str = "lite_s0"):
    print(json.dumps(extract_features_job.remote(encoder), indent=1))


def _summary(output):
    val = output["val"]
    row = {"name": output["name"], "val_mean_acc": round(val["mean_acc"], 4),
           "val_above_chance": round(val["above_chance"], 4), "firing_rate": round(val["firing_rate"], 4)}
    if "test" in output:
        row["test_mean_acc"] = round(output["test"]["mean_acc"], 4)
    return row


def _run_all(encoder, configs, folder, summary_file):
    rows = []
    for output in train_memory_job.map([encoder] * len(configs), configs, return_exceptions=True):
        if isinstance(output, Exception):
            print("FAILED", repr(output), flush=True)
            continue
        os.makedirs(os.path.join(LOCAL_RESULTS, folder, encoder), exist_ok=True)
        save_local(os.path.join(folder, encoder, output["name"] + ".json"), output)
        rows.append(_summary(output))
        print(json.dumps(rows[-1]), flush=True)
    previous = []
    path = os.path.join(LOCAL_RESULTS, summary_file)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            previous = [r for r in json.load(handle) if r["name"] not in {x["name"] for x in rows}]
    save_local(summary_file, previous + rows)


@app.local_entrypoint()
def sweep(encoder: str = "lite_s0", models: str = "mixer,dmp,plus", runs: str = ""):
    """Validation only, seed 0. `runs` replaces the grid: JSON [[kind, settings], ...]."""
    if runs:
        chosen = json.loads(runs)
    else:
        chosen = [[kind, settings] for kind in models.split(",") for settings in SWEEP[kind]]
    configs = [dict(MEMORY_BASE, kind=kind, model=settings, seed=0) for kind, settings in chosen]
    _run_all(encoder, configs, "memory", f"sweep_{encoder}.json")


@app.local_entrypoint()
def final(encoder: str = "lite_s0", runs: str = "", seeds: str = "0,1,2"):
    """Chosen settings, several seeds, scored once on test. `runs`: JSON [[kind, settings], ...]."""
    chosen, unique = json.loads(runs), []
    for kind, settings in chosen:
        if [kind, settings] not in unique:
            unique.append([kind, settings])
    configs = [dict(MEMORY_BASE, kind=kind, model=settings, seed=int(seed), test=True)
               for kind, settings in unique for seed in seeds.split(",")]
    _run_all(encoder, configs, "final", f"final_{encoder}.json")


def _count_params(kind, settings, in_dim):
    from vmem import streams as S
    from vmem.memory_models import VideoQA

    model = VideoQA(kind, in_dim, S.QUESTION_VECTOR_SIZE, S.NUM_CLASSES, **settings)
    return sum(p.numel() for p in model.parameters())


def _size_matched(kind, settings, in_dim, target):
    """Hidden size that brings this model's parameter count closest to `target`."""
    return min(range(256, 1537, 64),
               key=lambda h: abs(_count_params(kind, dict(settings, hidden=h), in_dim) - target))


def _write_volume(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1)
    volume.commit()


@app.function(image=image, volumes={DATA: volume}, cpu=2, memory=8192, timeout=12 * 3600)
def pipeline(encoder: str = "small_s0", seeds: str = "0,1,2"):
    """Round 2, run entirely on Modal: features, search, pick, size match, final runs with test."""
    import numpy as np

    volume.reload()
    features_path = f"{DATA}/features/{encoder}.npy"
    if not os.path.exists(features_path):
        print("features", extract_features_job.remote(encoder), flush=True)
        volume.reload()
    in_dim = int(np.load(features_path, mmap_mode="r").shape[1])
    summaries = f"{DATA}/summaries/{encoder}"

    configs = [dict(MEMORY_BASE, kind=kind, model=settings, seed=0) for kind, settings in SWEEP_V2]
    searched = []
    for output in train_memory_job.map([encoder] * len(configs), configs, return_exceptions=True):
        if isinstance(output, Exception):
            print("FAILED", repr(output), flush=True)
            continue
        searched.append(output)
        print("search", json.dumps(_summary(output)), flush=True)
    _write_volume(f"{summaries}/search.json", [_summary(o) for o in searched])

    best = {}
    for kind in ("mixer", "dmp", "plus"):
        runs = [o for o in searched if o["config"]["kind"] == kind]
        best[kind] = max(runs, key=lambda o: o["val"]["mean_acc"])["config"]["model"]
    target = _count_params("plus", best["plus"], in_dim)
    mixer_read = dict(best["mixer"], query_read=True)
    dmp_read = dict(best["dmp"], query_read=True)
    mixer_read["hidden"] = _size_matched("mixer", mixer_read, in_dim, target)
    dmp_read["hidden"] = _size_matched("dmp", dmp_read, in_dim, target)
    chosen = [
        ["mixer", best["mixer"]],                       # best tau found
        ["mixer", {"tau": 2.0}],                        # original Mixer-SNN setting
        ["mixer", mixer_read],                          # + memory read, size matched
        ["dmp", best["dmp"]],                           # best d and theta found
        ["dmp", dmp_read],                              # + memory read, size matched
        ["plus", best["plus"]],                         # the upgrade
        ["plus", dict(best["plus"], query_read=False)],  # the upgrade without the memory read
    ]
    plan = {"best": best, "plus_params": target,
            "params": {json.dumps(c): _count_params(c[0], c[1], in_dim) for c in chosen}, "chosen": chosen}
    print("plan", json.dumps(plan), flush=True)
    _write_volume(f"{summaries}/plan.json", plan)

    configs = [dict(MEMORY_BASE, kind=kind, model=settings, seed=int(seed), test=True)
               for kind, settings in chosen for seed in seeds.split(",")]
    finished = []
    for output in train_memory_job.map([encoder] * len(configs), configs, return_exceptions=True):
        if isinstance(output, Exception):
            print("FAILED", repr(output), flush=True)
            continue
        finished.append(_summary(output))
        print("final", json.dumps(finished[-1]), flush=True)
    _write_volume(f"{summaries}/final.json", finished)
    return {"plan": plan, "final": finished}


# Round 3: the upgraded model only, on a pretrained Spiking ResNet18 encoder.
UPGRADE_SWEEP = [
    ({"channels": 16, "memory_size": 10}, 30),
    ({"channels": 32, "memory_size": 10}, 30),
    ({"channels": 16, "memory_size": 20}, 30),
    ({"channels": 16, "memory_size": 10}, 60),
    ({"channels": 32, "memory_size": 10}, 60),
    ({"channels": 16, "memory_size": 10, "hidden": 384}, 60),
]


@app.function(image=image, volumes={DATA: volume}, cpu=2, memory=8192, timeout=24 * 3600)
def upgrade_pipeline(image_size: int = 112, encoder_epochs: int = 30, seeds: str = "0,1,2"):
    """Frames, encoder, features, search and final runs for the upgraded model. Resumes where it stopped."""
    volume.reload()
    encoder = "sresnet18_s0"
    if not os.path.exists(f"{DATA}/frames{image_size}/index.json"):
        print("frames", prepare_data.remote(image_size), flush=True)
    if not os.path.exists(f"{DATA}/encoders/{encoder}.json"):
        result = train_encoder_job.remote("sresnet18", encoder_epochs, 0, image_size, 1e-3)
        print("encoder", json.dumps({k: v for k, v in result.items() if k != "history"}), flush=True)
    volume.reload()
    if not os.path.exists(f"{DATA}/features/{encoder}.npy"):
        print("features", extract_features_job.remote(encoder, image_size), flush=True)
    summaries = f"{DATA}/summaries/{encoder}"

    def config(settings, epochs, seed, test=False):
        model = dict({"thetas": [16, 64, 256], "query_read": True}, **settings)
        return dict(MEMORY_BASE, kind="plus", model=model, seed=seed, epochs=epochs, test=test)

    configs = [config(settings, epochs, 0) for settings, epochs in UPGRADE_SWEEP]
    searched = []
    for output in train_memory_job.map([encoder] * len(configs), configs, return_exceptions=True):
        if isinstance(output, Exception):
            print("FAILED", repr(output), flush=True)
            continue
        searched.append(output)
        print("search", json.dumps(_summary(output)), flush=True)
    _write_volume(f"{summaries}/search.json", [dict(_summary(o), epochs=o["config"]["epochs"]) for o in searched])

    best = max(searched, key=lambda o: o["val"]["mean_acc"])["config"]
    finals = [dict(best, seed=int(seed), test=True) for seed in seeds.split(",")]
    finished = []
    for output in train_memory_job.map([encoder] * len(finals), finals, return_exceptions=True):
        if isinstance(output, Exception):
            print("FAILED", repr(output), flush=True)
            continue
        finished.append(_summary(output))
        print("final", json.dumps(finished[-1]), flush=True)
    _write_volume(f"{summaries}/final.json", {"best": best, "runs": finished})
    return {"best": best, "final": finished}


# Round 4: the best Mixer and DMP from round 2 and the upgrade, on the Spiking ResNet18
# features (reset every 8 frames), 60 epochs, with the clock and the relative question.
# The last run adds the Legendre delay read to the upgrade.
ROUND4_ENCODER = "sresnet18_s0_w8"
ROUND4_PLUS = {"channels": 16, "memory_size": 10, "thetas": [16, 64, 256], "query_read": True}
ROUND4_RUNS = [
    ["mixer", {"tau": 24.0, "query_read": True, "hidden": 448}],
    ["dmp", {"memory_size": 40, "theta": 256}],
    ["plus", ROUND4_PLUS],
    ["plus", dict(ROUND4_PLUS, delay_read=True)],
]


@app.local_entrypoint()
def round4(seeds: str = "0"):
    """modal run modal_app.py::round4. Finished runs are read back from the volume, not retrained."""
    configs = [dict(MEMORY_BASE, kind=kind, model=settings, seed=int(seed), epochs=60, clock=True,
                    relative=True, test=True)
               for kind, settings in ROUND4_RUNS for seed in seeds.split(",")]
    _run_all(ROUND4_ENCODER, configs, "round4", f"round4_{ROUND4_ENCODER}.json")
