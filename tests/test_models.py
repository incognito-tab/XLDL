"""Shape, causality and smoke tests for the encoder and the three memory models.

Run from the repository root: python -m unittest tests.test_models
"""

import json
import os
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from vmem import streams as S
from vmem.encoder import FrameEncoder
from vmem.frames import frame_count, stream_feature_rows
from vmem.memory_models import VideoQA, reset
from vmem.train_memory import QuestionBatches, evaluate

ROOT = os.environ.get("UCF101_ROOT", "data/ucf101")
SMALL = {
    "mixer": {"hidden": 32, "tau": 2.0, "query_read": True},
    "dmp": {"hidden": 32, "memory_size": 5, "theta": 64, "query_read": True},
    "plus": {"hidden": 32, "channels": 4, "memory_size": 5, "thetas": (16, 64), "query_read": True},
}


class EncoderTests(unittest.TestCase):
    def test_shapes_and_spike_rates(self):
        model = FrameEncoder("lite", image_size=64).eval()
        frames = torch.randint(0, 256, (3, 2, 64, 64, 3), dtype=torch.uint8)
        with torch.no_grad():
            scores = model(model.prepare(frames))
            reset(model)
            features = model.features(model.prepare(frames))
        self.assertEqual(tuple(scores.shape), (3, 2, 24))
        self.assertEqual(tuple(features.shape), (3, 2, model.feature_dim))
        self.assertTrue(((features >= 0) & (features <= 1)).all())


def steady_input(batch):
    """Features that hold steady for 32 steps at a time, like an activity on screen."""
    blocks = torch.rand(batch, S.STREAM_FRAMES // 32, 20) * 3
    return blocks.repeat_interleave(32, dim=1) + 0.1 * torch.rand(batch, S.STREAM_FRAMES, 20)


def neuron_outputs(memory_model, x):
    """Spikes of every LIF layer in call order. Multi-step layers put time first: [T, B, ...]."""
    captured = []
    hooks = [m.register_forward_hook(lambda module, inputs, output: captured.append(output))
             for m in memory_model.modules() if type(m).__name__.endswith("LIFNode")]
    memory_model(x)
    for hook in hooks:
        hook.remove()
    reset(memory_model)
    assert all(c.shape[0] == x.shape[1] for c in captured), "expected time-first outputs"
    return captured


class SpikingResNetEncoderTests(unittest.TestCase):
    def test_shapes_and_binary_block_outputs(self):
        from vmem.encoder import build_encoder

        model = build_encoder("sresnet18", 112, pretrained=False).eval()
        frames = torch.randint(0, 256, (3, 2, 112, 112, 3), dtype=torch.uint8)
        outputs = []
        net = model.net
        hooks = [block.register_forward_hook(lambda m, i, o: outputs.append(o))
                 for block in (net.layer1, net.layer2, net.layer3, net.layer4)]
        with torch.no_grad():
            features = model.features(model.prepare(frames))
        for hook in hooks:
            hook.remove()
        reset(model)
        self.assertEqual(tuple(features.shape), (3, 2, 512))
        for out in outputs:
            self.assertTrue(set(out.unique().tolist()) <= {0.0, 1.0})


class BatchedFeatureTests(unittest.TestCase):
    def test_batched_extraction_matches_one_clip_at_a_time(self):
        from spikingjelly.activation_based import functional
        from vmem.train_encoder import encode_clips

        torch.manual_seed(0)
        model = FrameEncoder("lite", image_size=64).eval()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_var.fill_(0.05)  # so that neurons fire on random frames
        frames = np.random.default_rng(0).integers(0, 256, (120, 64, 64, 3), dtype=np.uint8)
        spans = [(0, 30), (30, 9), (39, 61), (100, 17)]
        batched = encode_clips(model, frames, spans, "cpu")
        for (offset, count), out in zip(spans, batched):
            with torch.no_grad():
                single = model.features(model.prepare(torch.from_numpy(frames[offset:offset + count]).unsqueeze(1)))
            functional.reset_net(model)
            self.assertTrue(np.array_equal(single[:, 0].numpy(), out))
            self.assertGreater(out.mean(), 0)


class ResetEveryTests(unittest.TestCase):
    def test_reset_every_matches_separate_windows(self):
        from spikingjelly.activation_based import functional
        from vmem.train_encoder import encode_clips

        torch.manual_seed(0)
        model = FrameEncoder("lite", image_size=64).eval()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_var.fill_(0.05)
        frames = np.random.default_rng(1).integers(0, 256, (40, 64, 64, 3), dtype=np.uint8)
        out = encode_clips(model, frames, [(0, 20)], "cpu", reset_every=8)[0]
        for start in (0, 8, 16):
            with torch.no_grad():
                window = model.features(model.prepare(torch.from_numpy(frames[start:min(start + 8, 20)]).unsqueeze(1)))
            functional.reset_net(model)
            self.assertTrue(np.array_equal(window[:, 0].numpy(), out[start:start + 8]))


class CausalityTests(unittest.TestCase):
    def test_memory_models_never_look_ahead(self):
        torch.manual_seed(0)
        for kind, config in SMALL.items():
            model = VideoQA(kind, 20, S.QUESTION_VECTOR_SIZE, S.NUM_CLASSES, **config)
            x = steady_input(2)
            changed = x.clone()
            changed[:, 150:] = steady_input(2)[:, 150:]
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm1d):
                    module.momentum = None  # plain average, so 20 batches give exact statistics
            with torch.no_grad():
                for _ in range(20):  # give BatchNorm running statistics, as training would
                    model.memory(steady_input(4))
                    reset(model)
            model.eval()
            with torch.no_grad():
                layers_a = neuron_outputs(model.memory, x)
                layers_b = neuron_outputs(model.memory, changed)
            # Every spiking layer inside the memory model, not only the top one,
            # must agree on the steps before the change. At least one must be active,
            # or the comparison would prove nothing.
            for a, b in zip(layers_a, layers_b):
                self.assertTrue(torch.equal(a[:150], b[:150]), kind)
            self.assertGreater(sum(a.sum().item() for a in layers_a), 0, f"{kind} has no active layer")
            with torch.no_grad():
                _, memory_a = model.memory(x)
                reset(model)
                _, memory_b = model.memory(changed)
                reset(model)
            self.assertTrue(torch.allclose(memory_a[:, :150], memory_b[:, :150], atol=1e-5), kind)
            self.assertFalse(torch.allclose(memory_a[:, 150:], memory_b[:, 150:], atol=1e-5), kind)

            questions = torch.zeros(2, S.QUESTION_VECTOR_SIZE)
            questions[:, 0] = 1
            rows, seen = torch.tensor([0, 1]), torch.tensor([149, 120])
            with torch.no_grad():
                a, _ = model(x, questions, rows, seen)
                b, _ = model(changed, questions, rows, seen)
            self.assertTrue(torch.allclose(a, b, atol=1e-5), kind)


class MemoryReadTests(unittest.TestCase):
    def test_dmp_read_matches_the_cell_internal_memory(self):
        """The memory DMP exposes must be the one its own cell feeds to its neurons."""
        torch.manual_seed(0)
        model = VideoQA("dmp", 20, S.QUESTION_VECTOR_SIZE, S.NUM_CLASSES, **SMALL["dmp"]).eval()
        cell = model.memory.layer2
        captured = {}
        hook = cell.weight.register_forward_hook(lambda m, inputs, out: captured.update(m=inputs[0]))
        with torch.no_grad():
            _, memory = model.memory(steady_input(2))
        hook.remove()
        reset(model)
        self.assertTrue(torch.allclose(memory, captured["m"], atol=1e-5))

    def test_memory_off_silences_the_read(self):
        for kind in ("dmp", "plus"):
            model = VideoQA(kind, 20, S.QUESTION_VECTOR_SIZE, S.NUM_CLASSES, **SMALL[kind]).eval()
            model.memory.set_memory_gate(0.0)
            with torch.no_grad():
                _, memory = model.memory(steady_input(2))
            reset(model)
            self.assertEqual(memory.abs().max().item(), 0.0, kind)

    def test_legendre_decoder_recovers_the_input_at_the_asked_delay(self):
        """The fixed delay read must return roughly what entered the memory tau steps ago."""
        torch.manual_seed(0)
        model = VideoQA("plus", 20, S.QUESTION_VECTOR_SIZE, S.NUM_CLASSES,
                        **dict(SMALL["plus"], memory_size=10, thetas=(64, 256), delay_read=True)).eval()
        cell = model.memory.layer1
        x = steady_input(2)
        with torch.no_grad():
            u = torch.relu(cell.W_u(x))                                       # [B, T, k]
            m = cell.memory(x)
        seen = torch.tensor([207, 207, 207, 207])  # asked moments sit mid-block: 176, 144, 112
        rows = torch.tensor([0, 1, 0, 1])
        delays = torch.tensor([31, 63, 95, -1])
        values = cell.decode(m, rows, seen, delays).view(4, 2, -1)            # [Q, windows, k]
        for q in range(3):
            truth = u[rows[q], seen[q] - delays[q]]
            window = 0 if delays[q] < 64 else 1
            error = (values[q, window] - truth).abs().mean() / truth.abs().mean()
            self.assertLess(error.item(), 0.3, (q, error.item()))
        self.assertEqual(values[1, 0].abs().sum().item() > 0, True)
        self.assertEqual(values[2, 0].abs().max().item(), 0.0)  # 95 steps is outside the 64 window
        self.assertEqual(values[3].abs().max().item(), 0.0)     # no delay asked

    def test_delay_read_is_causal_and_silenced_with_memory_off(self):
        torch.manual_seed(0)
        model = VideoQA("plus", 20, S.QUESTION_VECTOR_SIZE, S.NUM_CLASSES,
                        **dict(SMALL["plus"], delay_read=True)).eval()
        x = steady_input(2)
        changed = x.clone()
        changed[:, 150:] = steady_input(2)[:, 150:]
        rows, seen, delays = torch.tensor([0, 1]), torch.tensor([149, 120]), torch.tensor([30, 10])
        with torch.no_grad():
            model.memory(x)
            a = model.memory.delay_values(rows, seen, delays)
            reset(model)
            model.memory(changed)
            b = model.memory.delay_values(rows, seen, delays)
            reset(model)
            self.assertTrue(torch.allclose(a, b, atol=1e-5))
            model.memory.set_memory_gate(0.0)
            model.memory(x)
            self.assertEqual(model.memory.delay_values(rows, seen, delays).abs().max().item(), 0.0)
            reset(model)

    def test_head_sends_only_spikes_between_neurons(self):
        model = VideoQA("plus", 20, S.QUESTION_VECTOR_SIZE, S.NUM_CLASSES, **SMALL["plus"]).eval()
        seen = []
        hook = model.head.fc1.register_forward_hook(lambda m, inputs, out: seen.append(inputs[0]))
        questions = torch.zeros(2, S.QUESTION_VECTOR_SIZE)
        questions[:, 0] = 1
        with torch.no_grad():
            model(steady_input(2), questions, torch.tensor([0, 1]), torch.tensor([200, 100]))
        hook.remove()
        z = seen[0]
        read_part = z[..., -256:]
        self.assertTrue(set(read_part.unique().tolist()) <= {0.0, 1.0})


class StreamFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, "metadata.json")) as handle:
            clips = json.load(handle)["clips"]
        cls.index, offset = {}, 0
        for clip in clips:
            count = frame_count(clip["duration"])
            cls.index[clip["path"]] = [offset, count]
            offset += count
        cls.rows = offset

    def test_rows_stay_inside_their_clip(self):
        dataset = S.StreamDataset(ROOT, "train")
        for i in range(300):
            sample = dataset.sample(i)
            total = 0
            for segment in sample.segments:
                rows = stream_feature_rows(segment, self.index)
                offset, count = self.index[segment.path]
                self.assertTrue(((rows >= offset) & (rows < offset + count)).all())
                self.assertTrue((np.diff(rows) >= 0).all())
                total += len(rows)
            self.assertEqual(total, S.STREAM_FRAMES)

    def test_one_training_step_and_evaluation_for_each_model(self):
        torch.manual_seed(0)
        features = torch.rand(self.rows, 20).half()
        batches = QuestionBatches(ROOT, features, self.index, "cpu", clock=True, relative=True)
        b = batches.build("train", [0, 1, 2])
        self.assertEqual(tuple(b["x"].shape), (3, S.STREAM_FRAMES, 20))
        self.assertTrue(b["mask"][torch.arange(len(b["answers"])), b["answers"]].all())
        self.assertTrue((b["delays"] <= b["seen"]).all())
        configs = dict(SMALL, plus_delay=dict(SMALL["plus"], delay_read=True))
        for name, config in configs.items():
            kind = name.split("_")[0]
            model = VideoQA(kind, 20, S.question_size(True, True), S.NUM_CLASSES, **config)
            scores, _ = model(b["x"], b["questions"], b["rows"], b["seen"], b["mask"], b["delays"])
            loss = F.cross_entropy(scores, b["answers"])
            loss.backward()
            self.assertTrue(torch.isfinite(loss), kind)
            result = evaluate(model, batches, "val", np.arange(4), batch=2)
            self.assertEqual(set(result["by_kind"]), {"cach_day_N_giay", "o_giay_thu_T", "da_xuat_hien", "ngay_sau_X"})


if __name__ == "__main__":
    unittest.main()
