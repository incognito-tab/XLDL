"""Train the spiking frame encoder on 1-second windows of single clips, then
extract per-frame features for every clip.

Only train clips are used for learning. Validation clips choose the checkpoint.
Test clips are never looked at here.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from spikingjelly.activation_based import functional

from vmem.encoder import build_encoder

WINDOW = 8  # frames, 1 second at 8 frames/s


def load_store(frames_dir, root):
    frames = np.load(os.path.join(frames_dir, "frames.npy"), mmap_mode="r")
    with open(os.path.join(frames_dir, "index.json")) as handle:
        index = json.load(handle)["clips"]
    with open(os.path.join(root, "metadata.json")) as handle:
        clips = json.load(handle)["clips"]
    return frames, index, clips


def augment(windows, generator):
    """windows: float [T, B, 3, H, W]. Same flip, crop and colour change across a window."""
    steps, batch, _, height, width = windows.shape
    device = windows.device
    flip = torch.rand(batch, device=device, generator=generator) < 0.5
    windows = torch.where(flip.view(1, batch, 1, 1, 1), windows.flip(-1), windows)
    pad = 4
    padded = F.pad(windows.flatten(0, 1), (pad, pad, pad, pad), mode="reflect").view(
        steps, batch, 3, height + 2 * pad, width + 2 * pad)
    dx = torch.randint(0, 2 * pad + 1, (batch,), generator=generator, device=device)
    dy = torch.randint(0, 2 * pad + 1, (batch,), generator=generator, device=device)
    out = torch.stack([padded[:, b, :, dy[b]:dy[b] + height, dx[b]:dx[b] + width] for b in range(batch)], 1)
    scale = 1 + 0.4 * (torch.rand(1, batch, 1, 1, 1, device=device, generator=generator) - 0.5)
    shift = 0.4 * (torch.rand(1, batch, 1, 1, 1, device=device, generator=generator) - 0.5)
    return out * scale + shift


def clip_windows(count):
    """Start frames of non-overlapping windows covering a clip."""
    if count < WINDOW:
        return [0]
    return list(range(0, count - WINDOW + 1, WINDOW))


@torch.no_grad()
def evaluate(model, frames_gpu, index, clips, split, device, batch=64):
    model.eval()
    items = [(c["path"], c["label"]) for c in clips if c["split"] == split]
    correct_clip = correct_window = windows_total = 0
    for path, label in items:
        offset, count = index[path]
        starts = clip_windows(count)
        scores = []
        for i in range(0, len(starts), batch):
            chunk = starts[i:i + batch]
            rows = torch.tensor([[offset + min(s + k, count - 1) for s in chunk] for k in range(WINDOW)],
                                device=device)
            x = model.prepare(frames_gpu[rows])
            scores.append(model(x).mean(0))
            functional.reset_net(model)
        scores = torch.cat(scores)
        correct_window += (scores.argmax(1) == label).sum().item()
        windows_total += len(starts)
        correct_clip += int(scores.softmax(1).mean(0).argmax().item() == label)
    return correct_clip / len(items), correct_window / windows_total


class FrameSource:
    """All frames in memory: on the GPU when small, otherwise in CPU RAM."""

    def __init__(self, frames, device):
        self.device = device
        store = torch.from_numpy(np.ascontiguousarray(frames))
        self.store = store.to(device) if store.numel() < 4e9 else store

    def __getitem__(self, rows):
        return self.store[rows.to(self.store.device)].to(self.device, non_blocking=True)


def train(root, frames_dir, out_path, size="lite", epochs=40, batch=32, lr=2e-3,
          windows_per_clip=4, seed=0, device="cuda", log=print):
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    frames, index, clips = load_store(frames_dir, root)
    frames_gpu = FrameSource(frames, device)
    image_size = frames.shape[1]
    model = build_encoder(size, image_size=image_size).to(device)
    params = sum(p.numel() for p in model.parameters())
    train_items = [(index[c["path"]], c["label"]) for c in clips if c["split"] == "train"]
    steps_per_epoch = len(train_items) * windows_per_clip // batch
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs * steps_per_epoch,
                                                    pct_start=0.1)
    rng = np.random.default_rng(seed)
    history, best = [], -1.0
    for epoch in range(epochs):
        model.train()
        started = time.time()
        order = np.repeat(np.arange(len(train_items)), windows_per_clip)
        rng.shuffle(order)
        loss_sum = 0.0
        for step in range(steps_per_epoch):
            picks = order[step * batch:(step + 1) * batch]
            rows, labels = [], []
            for i in picks:
                (offset, count), label = train_items[i]
                start = int(rng.integers(0, max(count - WINDOW, 0) + 1))
                rows.append([offset + min(start + k, count - 1) for k in range(WINDOW)])
                labels.append(label)
            rows = torch.tensor(rows, device=device).T  # [T, B]
            x = augment(model.prepare(frames_gpu[rows]), generator)
            loss = F.cross_entropy(model(x).mean(0), torch.tensor(labels, device=device),
                                   label_smoothing=0.1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            functional.reset_net(model)
            loss_sum += loss.item()
        clip_acc, window_acc = evaluate(model, frames_gpu, index, clips, "val", device)
        row = {"epoch": epoch + 1, "loss": loss_sum / steps_per_epoch, "val_clip_acc": clip_acc,
               "val_window_acc": window_acc, "seconds": time.time() - started}
        history.append(row)
        log(json.dumps(row))
        if clip_acc > best:
            best = clip_acc
            torch.save({"state": model.state_dict(), "size": size, "image_size": image_size,
                        "epoch": epoch + 1, "val_clip_acc": clip_acc}, out_path)
    return {"size": size, "params": params, "best_val_clip_acc": best,
            "best_epoch": max(history, key=lambda r: r["val_clip_acc"])["epoch"], "history": history}


def load_encoder(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = build_encoder(checkpoint["size"], checkpoint["image_size"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model


@torch.no_grad()
def encode_clips(model, frames, spans, device, reset_every=None):
    """Features for several clips at once. spans: [(offset, count)].

    Clips are padded at the end with their last frame to a common length. The
    encoder is causal, so padding after a clip cannot change its real frames.
    With `reset_every`, neuron states are cleared every that many frames, as in
    training, where the encoder only ever saw windows that start from rest.
    """
    longest = max(count for _, count in spans)
    rows = np.stack([offset + np.minimum(np.arange(longest), count - 1) for offset, count in spans], axis=1)
    x = torch.from_numpy(np.ascontiguousarray(frames[rows.reshape(-1)])).to(device)
    x = x.view(longest, len(spans), *x.shape[1:])                     # [T, B, H, W, 3]
    step = reset_every or longest
    chunks = []
    for start in range(0, longest, step):
        chunks.append(model.features(model.prepare(x[start:start + step])).float().cpu().numpy())
        functional.reset_net(model)
    out = np.concatenate(chunks)                                       # [T, B, C]
    return [out[:count, b] for b, (_, count) in enumerate(spans)]


@torch.no_grad()
def extract_features(encoder_path, root, frames_dir, out_path, device="cuda", batch=16, reset_every=None,
                     log=print):
    """Run the encoder over every whole clip in order and save per-frame spike rates."""
    model = load_encoder(encoder_path, device)
    frames, index, clips = load_store(frames_dir, root)
    features = np.zeros((frames.shape[0], model.feature_dim), dtype=np.float16)
    # Similar lengths together, so little work is spent on padding.
    spans = sorted((tuple(index[c["path"]]) for c in clips), key=lambda span: span[1])
    started = time.time()
    for i in range(0, len(spans), batch):
        group = spans[i:i + batch]
        for (offset, count), out in zip(group, encode_clips(model, frames, group, device, reset_every)):
            features[offset:offset + count] = out
        if (i // batch) % 20 == 0:
            log(f"features {i + len(group)}/{len(spans)} clips, {time.time() - started:.0f}s")
    np.save(out_path, features)
    return {"rows": int(features.shape[0]), "dim": int(features.shape[1]),
            "mean_rate": float(features.astype(np.float32).mean())}
