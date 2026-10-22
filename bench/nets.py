"""Model wrappers around the original Mixer-SNN and DMP-SNN code (unmodified)."""

import importlib.util
import os

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MIXER_SIZES = {
    # paper configuration (models/configs.py, "small")
    'small': dict(dim=78, alpha=3, patch_size=4, depths=[2, 8, 14, 2]),
    # same blocks, fewer/narrower layers so long sequences fit in GPU time
    'lite': dict(dim=48, alpha=3, patch_size=4, depths=[1, 2, 2, 1]),
}


def _load_dmp_module():
    # smnist variant: identical cell to shd/ssc, plus the gamma gate used for memory ablation
    path = os.path.join(ROOT, 'Dual_memory_pathways', 'smnist', 'src', 'dual_pathway.py')
    spec = importlib.util.spec_from_file_location('dmp_dual_pathway', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def readout(y, last):
    """y: [B, T, C] -> [B, C], mean over all steps or over the last `last` steps."""
    return y[:, -last:].mean(1) if last else y.mean(1)


class MixerSNN(nn.Module):
    kind = 'mixer'

    def __init__(self, n_classes, in_ch, size='small', T=4, readout_last=0, tau=2.0, cupy=False):
        super().__init__()
        import sys
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        from spikingjelly.activation_based import functional, neuron
        from models.mixers_sparse_patchcell_origin import sMLPNet   # the model train.py uses

        self.net = sMLPNet(in_channels=in_ch, num_classes=n_classes, image_size=32, **MIXER_SIZES[size])
        functional.set_step_mode(self.net, 'm')
        if cupy:
            try:
                functional.set_backend(self.net, 'cupy')
            except Exception as e:                                  # fall back to torch kernels
                print(f'cupy backend unavailable ({e}); using torch')
        for m in self.net.modules():
            if isinstance(m, neuron.LIFNode):
                m.tau = tau
        self.T = T
        self.readout_last = readout_last

    def forward(self, x):                       # x: [T or 1, B, C, 32, 32]
        if x.shape[0] == 1:
            x = x.expand(self.T, *x.shape[1:])  # static image repeated T times (paper setting)
        y = self.net(x)                         # [T, B, C]
        return readout(y.permute(1, 0, 2), self.readout_last)


class DMPSNN(nn.Module):
    kind = 'dmp'

    def __init__(self, n_classes, in_dim, T, hidden=128, d=10, theta=40, n_layers=2,
                 dropout=0.2, readout_last=0, gamma=1.0):
        super().__init__()
        mod = _load_dmp_module()
        dims = [in_dim] + [hidden] * n_layers
        self.cells = nn.ModuleList(
            mod.SpikingLMUFFTCell(input_size=dims[i], hidden_size=dims[i + 1], memory_size=d,
                                  seq_len=T, theta=theta) for i in range(n_layers))
        self.drop = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden, n_classes)
        self.readout_last = readout_last
        self.apply(self._init_weights)
        self.set_memory_gate(gamma)

    @staticmethod
    def _init_weights(m):                       # same init as the SHD ConvLMU2
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def set_memory_gate(self, gamma):
        for c in self.cells:
            c.set_memory_gate(gamma)

    def forward(self, x):                       # x: [B, T, F]
        for c in self.cells:
            x, _ = c(x)
        return readout(self.dense(self.drop(x)), self.readout_last)


def reset_states(model):
    for m in model.modules():
        if m is not model and hasattr(m, 'reset') and callable(m.reset) and hasattr(m, 'v'):
            m.reset()


def lif_modules(model):
    return [m for m in model.modules() if 'LIF' in type(m).__name__]
