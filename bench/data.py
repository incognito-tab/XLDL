"""Datasets for the Mixer-SNN vs DMP-SNN comparison.

Each dataset is stored in a canonical form (images as [N, C, H, W] uint8,
sequences as [N, T, F] uint8) and exposes two formatters that turn a batch into
the input each model expects:

    fmt_mixer(x) -> [T, B, C, 32, 32] float   (Mixer-SNN, multi-step)
    fmt_dmp(x)   -> [B, T, F] float           (DMP-SNN, sequence)

Both models always see the same examples, labels and splits.
"""

import gzip
import os
import shutil
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

SHD_URL = 'https://zenkelab.org/datasets/shd/{}.h5.gz'


@dataclass
class Split:
    x: torch.Tensor
    y: torch.Tensor


@dataclass
class Bundle:
    name: str
    train: Split
    val: Split
    test: Split
    n_classes: int
    T_dmp: int                      # sequence length seen by DMP
    T_mixer: int                    # number of Mixer time steps (0 = use --T)
    in_ch_mixer: int
    in_dim_dmp: int
    readout_last: int               # 0 = mean over all steps, k = mean over last k steps
    fmt_mixer: Callable = field(repr=False, default=None)
    fmt_dmp: Callable = field(repr=False, default=None)
    augment: Callable = field(repr=False, default=None)


def _split_val(x, y, val_frac, seed):
    """Stratified train/val split taken from the official training set."""
    g = np.random.default_rng(seed)
    y_np = y.numpy()
    val_idx = []
    for c in np.unique(y_np):
        idx = np.flatnonzero(y_np == c)
        g.shuffle(idx)
        val_idx.append(idx[:int(round(len(idx) * val_frac))])
    val_idx = np.sort(np.concatenate(val_idx))
    mask = np.ones(len(y_np), dtype=bool)
    mask[val_idx] = False
    tr = torch.from_numpy(np.flatnonzero(mask))
    va = torch.from_numpy(val_idx)
    return Split(x[tr], y[tr]), Split(x[va], y[va])


def _seq_to_image(x):
    """[B, T, F] -> [T, B, 1, 32, 32]: features laid out row-major on a 32x32 canvas."""
    B, T, Fdim = x.shape
    assert Fdim <= 1024, 'feature dim must fit a 32x32 canvas'
    x = F.pad(x, (0, 1024 - Fdim))
    return x.permute(1, 0, 2).reshape(T, B, 1, 32, 32)


# ---------------------------------------------------------------- Fashion-MNIST
FMNIST_MEAN, FMNIST_STD = 0.2860, 0.3530


def _load_fmnist(root):
    from torchvision.datasets import FashionMNIST
    tr = FashionMNIST(root, train=True, download=True)
    te = FashionMNIST(root, train=False, download=True)
    return tr.data.unsqueeze(1), tr.targets, te.data.unsqueeze(1), te.targets


def fmnist(root, seed, stream=False, T_static=4):
    xtr, ytr, xte, yte = _load_fmnist(root)
    train, val = _split_val(xtr, ytr, 0.1, seed=1234)

    def norm(x):
        return (x.float() / 255. - FMNIST_MEAN) / FMNIST_STD

    def fmt_dmp(x):                                   # one image row per step
        return norm(x)[:, 0]                          # [B, 28, 28]

    if stream:
        def fmt_mixer(x):                             # frame t shows only row t
            img = F.pad(norm(x)[:, 0], (2, 2, 2, 2))  # [B, 32, 32]
            B = img.shape[0]
            frames = img.new_zeros(28, B, 1, 32, 32)
            rows = torch.arange(28, device=img.device)
            frames[rows, :, 0, rows + 2, :] = img[:, rows + 2, :].permute(1, 0, 2)
            return frames
        T_mixer, readout = 28, 7                      # answer read from the last 7 rows only
    else:
        def fmt_mixer(x):
            img = F.pad(norm(x), (2, 2, 2, 2))
            return img.unsqueeze(0)                   # repeated over T in the model wrapper
        T_mixer, readout = 0, 0

    return Bundle('fmnist_stream' if stream else 'fmnist', train, val, Split(xte, yte), 10,
                  T_dmp=28, T_mixer=T_mixer, in_ch_mixer=1, in_dim_dmp=28, readout_last=readout,
                  fmt_mixer=fmt_mixer, fmt_dmp=fmt_dmp)


# ---------------------------------------------------------------- CIFAR-10
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def cifar_augment(x):                                   # random crop (pad 4) + horizontal flip
    B = x.shape[0]
    xp = F.pad(x, (4, 4, 4, 4))
    i = torch.randint(0, 9, (B,), device=x.device)
    j = torch.randint(0, 9, (B,), device=x.device)
    ar = torch.arange(32, device=x.device)
    rows = (i[:, None] + ar[None]).view(B, 1, 32, 1).expand(B, 3, 32, 32)
    cols = (j[:, None] + ar[None]).view(B, 1, 1, 32).expand(B, 3, 32, 32)
    out = xp[torch.arange(B, device=x.device)[:, None, None, None],
             torch.arange(3, device=x.device)[None, :, None, None], rows, cols]
    flip = torch.rand(B, device=x.device) < 0.5
    out[flip] = out[flip].flip(-1)
    return out


def cifar10(root, seed):
    from torchvision.datasets import CIFAR10
    tr = CIFAR10(root, train=True, download=True)
    te = CIFAR10(root, train=False, download=True)
    xtr = torch.from_numpy(tr.data).permute(0, 3, 1, 2).contiguous()
    xte = torch.from_numpy(te.data).permute(0, 3, 1, 2).contiguous()
    train, val = _split_val(xtr, torch.tensor(tr.targets), 0.1, seed=1234)

    def norm(x):
        return (x.float() / 255. - CIFAR_MEAN.to(x.device)) / CIFAR_STD.to(x.device)

    def fmt_mixer(x):
        return norm(x).unsqueeze(0)

    def fmt_dmp(x):                                   # one image row (32 px x 3 ch) per step
        return norm(x).permute(0, 2, 3, 1).reshape(x.shape[0], 32, 96)

    return Bundle('cifar10', train, val, Split(xte, torch.tensor(te.targets)), 10,
                  T_dmp=32, T_mixer=0, in_ch_mixer=3, in_dim_dmp=96, readout_last=0,
                  fmt_mixer=fmt_mixer, fmt_dmp=fmt_dmp, augment=cifar_augment)


# ---------------------------------------------------------------- SHD
def _download(url, dst):
    if os.path.exists(dst):
        return
    tmp = dst + '.part'
    print(f'downloading {url}', flush=True)
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dst)


def _shd_events_to_dense(path, T, max_time, pool):
    import h5py
    with h5py.File(path, 'r') as f:
        times, units, labels = f['spikes']['times'], f['spikes']['units'], f['labels'][:]
        n = len(labels)
        out = np.zeros((n, T, 700 // pool), dtype=np.uint8)
        for k in range(n):
            t = np.asarray(times[k])
            u = np.asarray(units[k]).astype(np.int64)
            keep = t < max_time
            b = (t[keep] / max_time * T).astype(np.int64)
            out[k, b, u[keep] // pool] = 1          # max-pool 700 channels by `pool`, binary spikes
    return out, labels.astype(np.int64)


def shd(root, seed, T=100, max_time=1.0, pool=5):
    """Spiking Heidelberg Digits, binned as in the DMP repo: 100 steps, 700 -> 140 channels."""
    d = os.path.join(root, 'shd')
    os.makedirs(d, exist_ok=True)
    cache = os.path.join(d, f'shd_T{T}_p{pool}.npz')
    if not os.path.exists(cache):
        arrays = {}
        for split in ('train', 'test'):
            gz = os.path.join(d, f'shd_{split}.h5.gz')
            h5 = gz[:-3]
            _download(SHD_URL.format(f'shd_{split}'), gz)
            if not os.path.exists(h5):
                with gzip.open(gz, 'rb') as fi, open(h5, 'wb') as fo:
                    shutil.copyfileobj(fi, fo)
            arrays[f'x_{split}'], arrays[f'y_{split}'] = _shd_events_to_dense(h5, T, max_time, pool)
        np.savez_compressed(cache, **arrays)
    z = np.load(cache)
    train, val = _split_val(torch.from_numpy(z['x_train']), torch.from_numpy(z['y_train']), 0.1, seed=1234)
    test = Split(torch.from_numpy(z['x_test']), torch.from_numpy(z['y_test']))
    return Bundle('shd', train, val, test, 20, T_dmp=T, T_mixer=T, in_ch_mixer=1,
                  in_dim_dmp=700 // pool, readout_last=0,
                  fmt_mixer=lambda x: _seq_to_image(x.float()), fmt_dmp=lambda x: x.float())


# ---------------------------------------------------------------- Delayed recall (memory probe)
def make_recall(n, delay, n_classes=10, cue_len=5, query_len=5, n_noise=9, p_cue=0.5,
                p_noise=0.05, n_distractors=2, seed=0):
    """Report the class of the FIRST cue after `delay` steps.

    Channels: [0, n_classes) class cues, n_classes = query flag, the rest pure noise.
    The delay contains background noise on every channel plus `n_distractors`
    later cue bursts of other classes, so the model must keep the first cue,
    not the most recent one.
    """
    g = np.random.default_rng(seed)
    T = cue_len + delay + query_len
    Fdim = n_classes + 1 + n_noise
    x = (g.random((n, T, Fdim)) < p_noise).astype(np.uint8)
    x[:, :, n_classes] = 0
    y = g.integers(0, n_classes, n)
    burst = (g.random((n, cue_len)) < p_cue).astype(np.uint8)
    burst[:, 0] = 1
    for s in range(cue_len):
        x[np.arange(n), s, y] = burst[:, s]
    if delay >= cue_len * (n_distractors + 1):
        for _ in range(n_distractors):
            other = (y + g.integers(1, n_classes, n)) % n_classes
            start = g.integers(cue_len, cue_len + delay - cue_len + 1, n)
            for s in range(cue_len):
                hit = g.random(n) < p_cue
                x[np.arange(n)[hit], start[hit] + s, other[hit]] = 1
    x[:, T - query_len:, n_classes] = 1
    return torch.from_numpy(x), torch.from_numpy(y)


def recall(root, seed, delay=50, n_distractors=2):
    tr = Split(*make_recall(10000, delay, n_distractors=n_distractors, seed=1))
    va = Split(*make_recall(2000, delay, n_distractors=n_distractors, seed=2))
    te = Split(*make_recall(2000, delay, n_distractors=n_distractors, seed=3))
    T = tr.x.shape[1]
    return Bundle(f'recall_D{delay}', tr, va, te, 10, T_dmp=T, T_mixer=T, in_ch_mixer=1,
                  in_dim_dmp=tr.x.shape[2], readout_last=5,
                  fmt_mixer=lambda x: _seq_to_image(x.float()), fmt_dmp=lambda x: x.float())


def get(name, root, seed, delay=50, n_distractors=2):
    if name == 'fmnist':
        return fmnist(root, seed)
    if name == 'fmnist_stream':
        return fmnist(root, seed, stream=True)
    if name == 'cifar10':
        return cifar10(root, seed)
    if name == 'shd':
        return shd(root, seed)
    if name == 'recall':
        return recall(root, seed, delay, n_distractors)
    raise ValueError(name)
