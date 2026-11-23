"""The three memory models and the question head they share.

All three read the same per-frame encoder features and differ only in how they
keep the past:

    MixerMemory      Mixer-SNN style layers. The only memory is the LIF membrane
                     potential, with a tunable time constant tau.
    DMPMemory        Two original SpikingLMUFFTCell layers from the DMP-SNN code,
                     unmodified. A slow memory of size d over a window theta.
    DualMemoryPlus   The upgrade. DMP's memory dynamics, fed by a vector instead of
                     one number (H1), over several windows at once (H2).

Any of the three can also let the question read its memory directly (H3,
`query_read=True`): Mixer exposes its membrane potentials, DMP and the upgrade
their slow memory. The read goes through spiking neurons, so every signal that
moves between neurons is a spike.

The upgrade can also read its memory at the moment a question points to (H4,
`delay_read=True`). A Legendre memory over a window theta can be decoded back to
its input tau steps ago with fixed weights P_i(tau / theta), the shifted Legendre
polynomials. Those decoded values join the read neurons' input.

Every model is causal: the output at step t depends only on inputs up to t.
"""

import importlib.util
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional, layer, neuron
from torch import fft

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DMP = None


def dmp_module():
    """The original DMP code, loaded from Dual_memory_pathways without changes."""
    global _DMP
    if _DMP is None:
        path = os.path.join(ROOT, "Dual_memory_pathways", "smnist", "src", "dual_pathway.py")
        spec = importlib.util.spec_from_file_location("dmp_dual_pathway", path)
        module = importlib.util.module_from_spec(spec)
        rng_state = torch.random.get_rng_state()
        spec.loader.exec_module(module)  # the file calls torch.manual_seed(42) on import
        torch.random.set_rng_state(rng_state)
        _DMP = module
    return _DMP


def reset(model):
    """Clear neuron states. Covers both spikingjelly APIs (DMP uses the older one)."""
    for module in model.modules():
        if module is not model and hasattr(module, "reset") and callable(module.reset):
            module.reset()


class MixerMemory(nn.Module):
    """Two Mixer-SNN channel-mixing style layers (Linear, BatchNorm, LIF)."""

    def __init__(self, in_dim, hidden=256, tau=2.0, query_read=False, **_):
        super().__init__()
        self.top = neuron.LIFNode(tau=tau, detach_reset=True, store_v_seq=query_read)
        self.layers = nn.Sequential(
            layer.Linear(in_dim, hidden), layer.SeqToANNContainer(nn.BatchNorm1d(hidden)),
            neuron.LIFNode(tau=tau, detach_reset=True),
            layer.Linear(hidden, hidden), layer.SeqToANNContainer(nn.BatchNorm1d(hidden)),
            self.top,
        )
        functional.set_step_mode(self, "m")
        self.hidden = hidden
        self.memory_dim = hidden if query_read else 0

    def forward(self, x):
        spikes = self.layers(x.transpose(0, 1).contiguous())  # [T, B, H]
        memory = self.top.v_seq.transpose(0, 1) if self.memory_dim else None
        return spikes.transpose(0, 1), memory


def legendre_decoders(memory_size, theta, seq_len):
    """[seq_len, d] weights that turn the memory into its input tau steps ago, for each tau.

    Zero for tau >= theta, which the window no longer covers.
    """
    weights = np.zeros((seq_len, memory_size))
    for tau in range(min(theta, seq_len)):
        for i in range(memory_size):
            weights[tau, i] = np.polynomial.legendre.legval(2 * tau / theta - 1, [0] * i + [1])
    return torch.tensor(weights, dtype=torch.float32)


def dmp_cell_memory(cell, x):
    """The slow memory m inside an original SpikingLMUFFTCell, computed the same way the cell does."""
    steps = x.shape[1]
    u = cell.f_u(cell.W_u(x)).permute(0, 2, 1)                        # [B, 1, T]
    m = fft.irfft(fft.rfft(u, n=2 * steps, dim=-1) * cell.fft_H.unsqueeze(0), n=2 * steps, dim=-1)
    return m[:, :, :steps].permute(0, 2, 1).contiguous() * cell.gamma  # [B, T, d]


class DMPMemory(nn.Module):
    """Two layers of the original DMP SpikingLMUFFTCell."""

    def __init__(self, in_dim, hidden=256, memory_size=10, theta=256, seq_len=256, query_read=False, **_):
        super().__init__()
        dmp = dmp_module()
        self.layer1 = dmp.SpikingLMUFFTCell(in_dim, hidden, memory_size, seq_len, theta)
        self.layer2 = dmp.SpikingLMUFFTCell(hidden, hidden, memory_size, seq_len, theta)
        self.seq_len = seq_len
        self.hidden = hidden
        self.memory_dim = memory_size if query_read else 0

    def set_memory_gate(self, gamma):
        self.layer1.set_memory_gate(gamma)
        self.layer2.set_memory_gate(gamma)

    def forward(self, x):
        if x.shape[1] != self.seq_len:
            raise ValueError("the DMP FFT cell needs sequences of exactly seq_len steps")
        spikes1, _ = self.layer1(x)
        spikes, _ = self.layer2(spikes1)
        memory = dmp_cell_memory(self.layer2, spikes1) if self.memory_dim else None
        return spikes, memory


def linear_projection(in_features, out_features, rank=None):
    """Optionally factor a linear map into two smaller matrices."""
    if rank is None:
        return nn.Linear(in_features, out_features)
    if not 0 < rank * (in_features + out_features) < in_features * out_features:
        raise ValueError(f"rank={rank} must reduce the size of the {in_features}x{out_features} projection")
    return nn.Sequential(nn.Linear(in_features, rank, bias=False), nn.Linear(rank, out_features))


class VectorMemoryCell(nn.Module):
    """DMP's slow memory with a vector input and several windows.

    u = ReLU(W_u x) has `channels` values instead of one. Each channel is kept by
    the same Legendre memory as DMP, once per window in `thetas`.
    """

    def __init__(self, in_dim, hidden, channels=16, memory_size=10, thetas=(256,), seq_len=256,
                 memory_rank=None):
        super().__init__()
        dmp = dmp_module()
        responses = [dmp.LMUFFTCell(1, 1, memory_size, seq_len, theta).fft_H for theta in thetas]
        self.register_buffer("fft_H", torch.stack(responses))  # [windows, d, seq_len + 1]
        self.register_buffer("decoders", torch.stack([legendre_decoders(memory_size, theta, seq_len)
                                                      for theta in thetas]))  # [windows, seq_len, d]
        self.shape = (len(thetas), channels, memory_size)
        self.W_u = nn.Linear(in_dim, channels)
        self.W_x = nn.Linear(in_dim, hidden)
        self.memory_dim = len(thetas) * channels * memory_size
        self.weight = linear_projection(self.memory_dim, hidden, memory_rank)
        self.lif = neuron.LIFNode(tau=2.0, detach_reset=True, step_mode="m")  # as in DMP
        self.seq_len = seq_len
        self.gamma = 1.0

    def memory(self, x):
        batch, steps, _ = x.shape
        u = torch.relu(self.W_u(x)).permute(0, 2, 1)                       # [B, k, T]
        spectrum = fft.rfft(u, n=2 * steps, dim=-1)                         # [B, k, T + 1]
        mixed = spectrum[:, None, :, None, :] * self.fft_H[None, :, None, :, :]
        m = fft.irfft(mixed, n=2 * steps, dim=-1)[..., :steps]              # [B, n, k, d, T]
        return m.permute(0, 4, 1, 2, 3).reshape(batch, steps, self.memory_dim) * self.gamma

    def forward(self, x):
        if x.shape[1] != self.seq_len:
            raise ValueError("the FFT memory needs sequences of exactly seq_len steps")
        m = self.memory(x)
        pre = self.W_x(x) + self.weight(m)
        spikes = self.lif(pre.transpose(0, 1).contiguous()).transpose(0, 1)
        return spikes, m

    def decode(self, m, rows, steps_seen, delays):
        """Each channel's input `delays` steps before step `steps_seen`, per window. -1: none."""
        state = m[rows, steps_seen].view(len(rows), *self.shape)            # [Q, n, k, d]
        weights = self.decoders[:, delays.clamp(min=0)].transpose(0, 1)     # [Q, n, d]
        values = torch.einsum("qnkd,qnd->qnk", state, weights).flatten(1)
        return values * (delays >= 0).unsqueeze(1).float()


class DualMemoryPlus(nn.Module):
    def __init__(self, in_dim, hidden=256, channels=16, memory_size=10, thetas=(256,),
                 query_read=False, delay_read=False, seq_len=256, memory_rank=None, read_rank=None, **_):
        super().__init__()
        if delay_read and not query_read:
            raise ValueError("delay_read joins the memory read, so it needs query_read")
        if read_rank is not None and not query_read:
            raise ValueError("read_rank requires query_read")
        thetas = tuple(thetas)
        self.layer1 = VectorMemoryCell(in_dim, hidden, channels, memory_size, thetas, seq_len,
                                       memory_rank=memory_rank)
        self.layer2 = VectorMemoryCell(hidden, hidden, channels, memory_size, thetas, seq_len,
                                       memory_rank=memory_rank)
        self.hidden = hidden
        self.memory_dim = self.layer2.memory_dim if query_read else 0
        self.read_rank = read_rank
        self.delay_dim = 2 * len(thetas) * channels if delay_read else 0
        self._memories = None

    def set_memory_gate(self, gamma):
        self.layer1.gamma = gamma
        self.layer2.gamma = gamma

    def forward(self, x):
        spikes, memory1 = self.layer1(x)
        spikes, memory = self.layer2(spikes)
        self._memories = (memory1, memory) if self.delay_dim else None
        return spikes, (memory if self.memory_dim else None)

    def delay_values(self, rows, steps_seen, delays):
        """Both layers' memories decoded at the moment each question points to."""
        memory1, memory2 = self._memories
        return torch.cat([self.layer1.decode(memory1, rows, steps_seen, delays),
                          self.layer2.decode(memory2, rows, steps_seen, delays)], dim=1)


class QuestionHead(nn.Module):
    """Reads the memory model at the moment a question is asked.

    Always: the spike rate of the memory model's top layer over the last `window`
    steps, and the question as a binary spike vector. When the model allows a
    memory read, the memory state and the question drive a layer of read neurons
    whose spikes join the input; BatchNorm there can be folded into the weights.
    A spiking layer integrates everything for `steps` steps and a linear readout
    of its spikes gives one score per activity.
    """

    def __init__(self, hidden, question_dim, num_classes, memory_dim=0, width=256, steps=4, window=8,
                 read_width=256, delay_dim=0, read_rank=None):
        super().__init__()
        self.window = window
        self.steps = steps
        self.memory_dim = memory_dim
        extra = 0
        if memory_dim:
            self.read = linear_projection(memory_dim + question_dim + delay_dim, read_width, read_rank)
            self.read_norm = nn.BatchNorm1d(read_width)
            self.read_lif = neuron.LIFNode(tau=2.0, detach_reset=True, step_mode="m")
            extra = read_width
        self.fc1 = layer.Linear(hidden + question_dim + extra, width)
        self.lif = neuron.LIFNode(tau=2.0, detach_reset=True)
        self.fc2 = layer.Linear(width, num_classes)
        functional.set_step_mode(self, "m")

    def forward(self, spikes, memory, questions, rows, steps_seen, delayed=None):
        """rows, steps_seen: stream index and last visible step of each question.
        delayed: memory decoded at the asked moment, joined to the read input."""
        total = torch.cumsum(spikes, dim=1)
        total = torch.cat([torch.zeros_like(total[:, :1]), total], dim=1)   # total[:, t] = sum of steps < t
        end = steps_seen + 1
        start = torch.clamp(end - self.window, min=0)
        rate = (total[rows, end] - total[rows, start]) / (end - start).unsqueeze(1).float()
        z = torch.cat([rate, questions], dim=1).unsqueeze(0).expand(self.steps, -1, -1)
        if self.memory_dim:
            parts = [memory[rows, steps_seen], questions] + ([delayed] if delayed is not None else [])
            current = self.read_norm(self.read(torch.cat(parts, dim=1)))
            read_spikes = self.read_lif(current.unsqueeze(0).expand(self.steps, -1, -1).contiguous())
            z = torch.cat([z, read_spikes], dim=2)
        return self.fc2(self.lif(self.fc1(z.contiguous()))).mean(0)


class TemporalConvMemory(nn.Module):
    """An ANN baseline with a causal, 256-frame receptive field."""

    def __init__(self, in_dim, hidden=192, kernel_size=2):
        super().__init__()
        self.hidden = hidden
        self.project = nn.Conv1d(in_dim, hidden, 1)
        dilations = {2: (1, 2, 4, 8, 16, 32, 64, 128), 4: (1, 4, 16, 64)}[kernel_size]
        self.blocks = nn.ModuleList(
            nn.Conv1d(hidden, hidden, kernel_size, dilation=dilation) for dilation in dilations
        )

    def forward(self, features):
        x = self.project(features.transpose(1, 2)).relu()
        for conv in self.blocks:
            left_padding = (conv.kernel_size[0] - 1) * conv.dilation[0]
            x = x + conv(F.pad(x, (left_padding, 0))).relu()
        return x.transpose(1, 2), None


class TemporalConvHead(nn.Module):
    """Classify from the TCN state and the question at the last visible frame."""

    def __init__(self, hidden, question_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden + question_dim, 256), nn.ReLU(), nn.Linear(256, num_classes)
        )

    def forward(self, states, memory, questions, rows, steps_seen, delayed=None):
        return self.classifier(torch.cat([states[rows, steps_seen], questions], dim=1))


MODELS = {"mixer": MixerMemory, "dmp": DMPMemory, "plus": DualMemoryPlus, "tcn": TemporalConvMemory}


class VideoQA(nn.Module):
    def __init__(self, kind, in_dim, question_dim, num_classes, **config):
        super().__init__()
        self.memory = MODELS[kind](in_dim, **config)
        self.delay_dim = getattr(self.memory, "delay_dim", 0)
        self.is_spiking = kind != "tcn"
        if self.is_spiking:
            self.head = QuestionHead(self.memory.hidden, question_dim, num_classes, self.memory.memory_dim,
                                     delay_dim=self.delay_dim, read_rank=getattr(self.memory, "read_rank", None))
        else:
            self.head = TemporalConvHead(self.memory.hidden, question_dim, num_classes)

    def forward(self, features, questions, rows, steps_seen, option_mask=None, delays=None):
        """delays: frames back to the moment each question asks about, -1 when none."""
        states, memory = self.memory(features)
        delayed = None
        if self.delay_dim:
            if delays is None:
                delays = torch.full_like(rows, -1)
            delayed = self.memory.delay_values(rows, steps_seen, delays)
        scores = self.head(states, memory, questions, rows, steps_seen, delayed)
        if option_mask is not None:
            scores = scores.masked_fill(~option_mask, float("-inf"))
        reset(self)
        return scores, states if self.is_spiking else None
