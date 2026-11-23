"""Train and evaluate a memory model on question streams built from encoder features.

Models compared with the same encoder get the same features, data, optimiser
and number of updates. The spiking models share a spiking question head.
The checkpoint is chosen on validation streams. Test streams are scored only when
`test=True`, which the final runs use once.
"""

import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from vmem import streams as S
from vmem.frames import stream_feature_rows
from vmem.memory_models import VideoQA

KIND_NAMES = {S.WHAT_WAS_HAPPENING: "cach_day_N_giay", S.WHAT_AT_TIME: "o_giay_thu_T",
              S.WHICH_APPEARED: "da_xuat_hien", S.WHAT_CAME_NEXT: "ngay_sau_X"}


class QuestionBatches:
    """Turns stream samples into feature tensors and question tensors on the device."""

    def __init__(self, root, features, index, device, seed=0, clock=False, relative=False):
        self.features = features                      # torch [rows, C] on device
        self.clock = clock
        self.relative = relative
        self.index = index
        self.device = device
        self.datasets = {split: S.StreamDataset(root, split, seed=seed) for split in ("train", "val", "test")}

    def build(self, split, stream_ids):
        samples = [self.datasets[split].sample(int(i)) for i in stream_ids]
        rows = np.concatenate([np.concatenate([stream_feature_rows(seg, self.index) for seg in s.segments])
                               for s in samples])
        x = self.features[torch.from_numpy(rows).to(self.device)].float()
        x = x.view(len(samples), S.STREAM_FRAMES, -1)
        questions = [(b, q) for b, s in enumerate(samples) for q in s.questions]
        vectors = torch.tensor(np.stack([S.encode_question(q, self.clock, self.relative) for _, q in questions]),
                               dtype=torch.float32, device=self.device)
        mask = torch.ones(len(questions), S.NUM_CLASSES, dtype=torch.bool)
        for i, (_, q) in enumerate(questions):
            if q.options:
                mask[i] = False
                mask[i, list(q.options)] = True
        as_tensor = lambda values: torch.tensor(values, device=self.device)
        return {
            "x": x,
            "questions": vectors,
            "rows": as_tensor([b for b, _ in questions]),
            "seen": as_tensor([q.seen_frames - 1 for _, q in questions]),
            "delays": as_tensor([S.delay_frames(q) for _, q in questions]),
            "answers": as_tensor([q.answer for _, q in questions]),
            "kinds": as_tensor([q.kind for _, q in questions]),
            "ages": as_tensor([q.age_seconds for _, q in questions]),
            "mask": mask.to(self.device),
        }


@torch.no_grad()
def evaluate(model, batches, split, stream_ids, batch=128, memory_gate=None):
    model.eval()
    if memory_gate is not None:
        model.memory.set_memory_gate(memory_gate)
    kinds, correct, ages, spike_sum, spike_count = [], [], [], 0.0, 0
    for i in range(0, len(stream_ids), batch):
        b = batches.build(split, stream_ids[i:i + batch])
        scores, spikes = model(b["x"], b["questions"], b["rows"], b["seen"], b["mask"], b["delays"])
        correct.append((scores.argmax(1) == b["answers"]).cpu())
        kinds.append(b["kinds"].cpu())
        ages.append(b["ages"].cpu())
        if spikes is not None:
            spike_sum += spikes.float().sum().item()
            spike_count += spikes.numel()
    if memory_gate is not None:
        model.memory.set_memory_gate(1.0)
    firing_rate = spike_sum / spike_count if spike_count else None
    return summarize(torch.cat(kinds), torch.cat(correct), torch.cat(ages), firing_rate)


def summarize(kinds, correct, ages, firing_rate):
    by_kind = {}
    for kind, name in KIND_NAMES.items():
        hits = correct[kinds == kind]
        by_kind[name] = {"acc": hits.float().mean().item(), "n": int(hits.numel()),
                         "chance": S.CHANCE[kind]}
    by_age = {}
    for low, high in ((0, 4), (4, 8), (8, 16), (16, 32)):
        pick = (ages >= low) & (ages < high)
        if pick.any():
            by_age[f"{low}-{high}s"] = {"acc": correct[pick].float().mean().item(), "n": int(pick.sum())}
    mean_acc = float(np.mean([v["acc"] for v in by_kind.values()]))
    above_chance = float(np.mean([(v["acc"] - v["chance"]) / (1 - v["chance"]) for v in by_kind.values()]))
    return {"mean_acc": mean_acc, "above_chance": above_chance, "by_kind": by_kind, "by_age": by_age,
            "firing_rate": firing_rate}


def train(root, features, index, config, device="cuda", log=print):
    """config: kind, model (settings), seed, epochs, streams_per_epoch, batch, lr, val_streams,
    feature_dropout, clock (bool), relative (bool), test (bool), test_streams."""
    seed = config.get("seed", 0)
    torch.manual_seed(seed)
    clock, relative = config.get("clock", False), config.get("relative", False)
    batches = QuestionBatches(root, features, index, device, clock=clock, relative=relative)
    question_dim = S.question_size(clock, relative)
    model = VideoQA(config["kind"], features.shape[1], question_dim, S.NUM_CLASSES,
                    **config.get("model", {})).to(device)
    params = sum(p.numel() for p in model.parameters())
    epochs, per_epoch, batch = config["epochs"], config["streams_per_epoch"], config["batch"]
    steps = epochs * (per_epoch // batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["lr"], total_steps=steps,
                                                    pct_start=0.1)
    val_ids = np.arange(config["val_streams"])
    rng = np.random.default_rng(seed)
    best, best_state, history = -1.0, None, []
    for epoch in range(epochs):
        model.train()
        started = time.time()
        # Every epoch sees new streams, drawn from train clips only.
        ids = rng.integers(0, 10**9, size=per_epoch)
        loss_sum = 0.0
        for i in range(0, per_epoch - batch + 1, batch):
            b = batches.build("train", ids[i:i + batch])
            # Train-clip features are cleaner than unseen ones, because the encoder learned
            # on those clips. Dropping some during training narrows that gap.
            x = F.dropout(b["x"], p=config.get("feature_dropout", 0.0), training=True)
            scores, _ = model(x, b["questions"], b["rows"], b["seen"], b["mask"], b["delays"])
            loss = F.cross_entropy(scores, b["answers"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
        if (epoch + 1) % config.get("eval_every", 5) == 0 or epoch + 1 == epochs:
            result = evaluate(model, batches, "val", val_ids)
            row = {"epoch": epoch + 1, "loss": loss_sum / (per_epoch // batch), "seconds": time.time() - started,
                   "val_mean_acc": result["mean_acc"], "val_above_chance": result["above_chance"]}
            history.append(row)
            log(json.dumps(row))
            if result["mean_acc"] > best:
                best = result["mean_acc"]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    output = {"config": config, "params": params, "history": history,
              "val": evaluate(model, batches, "val", val_ids)}
    if hasattr(model.memory, "set_memory_gate"):
        output["val_memory_off"] = evaluate(model, batches, "val", val_ids, memory_gate=0.0)
    if config.get("test"):
        test_ids = np.arange(config["test_streams"])
        output["test"] = evaluate(model, batches, "test", test_ids)
        if hasattr(model.memory, "set_memory_gate"):
            output["test_memory_off"] = evaluate(model, batches, "test", test_ids, memory_gate=0.0)
    return output, best_state
