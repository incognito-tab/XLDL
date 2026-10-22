"""Train one model on one dataset and write a JSON result.

    python -m bench.run --dataset shd --model dmp --seed 0 --out /content/drive/MyDrive/XLDL_bench

Protocol (same for both models):
  * official train set -> 90% train / 10% stratified validation; official test set untouched
  * best epoch chosen on validation; the test set is evaluated once, with that checkpoint
  * a run that already has a result file is skipped; an interrupted run resumes from its checkpoint
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from bench import data as D
from bench.nets import DMPSNN, MixerSNN, lif_modules, reset_states

# per-model optimiser defaults: AdamW + cosine; DMP lr/wd follow the DMP SHD script
OPT_DEFAULTS = {'mixer': dict(lr=1e-3, wd=0.01), 'dmp': dict(lr=5e-3, wd=0.1)}
EPOCH_DEFAULTS = {'fmnist': 15, 'fmnist_stream': 15, 'cifar10': 30, 'shd': 60, 'recall': 30}
THETA_DEFAULTS = {'shd': 40}              # DMP SHD setting; elsewhere theta = sequence length


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=list(EPOCH_DEFAULTS))
    p.add_argument('--model', required=True, choices=['mixer', 'dmp'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int)
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--lr', type=float)
    p.add_argument('--wd', type=float)
    p.add_argument('--data-root', default='./data')
    p.add_argument('--out', default='./bench_out')
    p.add_argument('--tag', default='', help='suffix that distinguishes ablation runs')
    p.add_argument('--max-train', type=int, default=0, help='subsample training set (smoke tests)')
    p.add_argument('--time-limit', type=float, default=0, help='minutes; save and exit(3) when exceeded')
    # recall task
    p.add_argument('--delay', type=int, default=50)
    p.add_argument('--distractors', type=int, default=2)
    # Mixer-SNN
    p.add_argument('--mixer-size', default='small', choices=['small', 'lite'])
    p.add_argument('--T', type=int, default=4, help='repetitions of a static image')
    p.add_argument('--tau', type=float, default=2.0, help='LIF time constant of every Mixer neuron')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--cupy', action='store_true')
    # DMP-SNN
    p.add_argument('--d', type=int, default=10, help='memory dimension')
    p.add_argument('--theta', type=float, help='memory window (steps)')
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--gamma', type=float, default=1.0, help='1 = slow memory on, 0 = LIF only')
    return p.parse_args()


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def run_id(a, b):
    if a.model == 'mixer':
        m = f'mixer-{a.mixer_size}' + (f'_T{a.T}' if b.T_mixer == 0 else '') + (f'_tau{a.tau:g}' if a.tau != 2 else '')
    else:
        m = f'dmp_d{a.d}_th{a.theta:g}' + (f'_g{a.gamma:g}' if a.gamma != 1 else '')
    return f'{b.name}__{m}{"_" + a.tag if a.tag else ""}__s{a.seed}'


def build(a, b):
    if a.model == 'mixer':
        T = b.T_mixer or a.T
        return MixerSNN(b.n_classes, b.in_ch_mixer, a.mixer_size, T=T, readout_last=b.readout_last,
                        tau=a.tau, cupy=a.cupy)
    return DMPSNN(b.n_classes, b.in_dim_dmp, b.T_dmp, hidden=a.hidden, d=a.d, theta=a.theta,
                  readout_last=b.readout_last, gamma=a.gamma)


def batches(split, bs, shuffle, device):
    n = len(split.y)
    idx = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        yield split.x[j], split.y[j]


@torch.no_grad()
def evaluate(model, split, fmt, a, device, amp, rates=False):
    model.eval()
    correct, loss, n = 0, 0.0, 0
    hooks, acc = [], []
    if rates:
        for m in lif_modules(model):
            hooks.append(m.register_forward_hook(lambda mod, i, o: acc.append(o.float().mean().item())))
    for x, y in batches(split, a.bs * 2, False, device):
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            out = model(fmt(x)).float()
        loss += nn.functional.cross_entropy(out, y, reduction='sum').item()
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
        reset_states(model)
    for h in hooks:
        h.remove()
    return {'acc': correct / n, 'loss': loss / n, **({'firing_rate': float(np.mean(acc)) if acc else 0.0} if rates else {})}


def main():
    a = parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    a.epochs = a.epochs or EPOCH_DEFAULTS[a.dataset]
    a.lr = a.lr or OPT_DEFAULTS[a.model]['lr']
    a.wd = a.wd if a.wd is not None else OPT_DEFAULTS[a.model]['wd']

    b = D.get(a.dataset, a.data_root, a.seed, a.delay, a.distractors)
    if a.theta is None:
        a.theta = float(THETA_DEFAULTS.get(a.dataset, b.T_dmp))
    if a.max_train:
        b.train = D.Split(b.train.x[:a.max_train], b.train.y[:a.max_train])
    for s in ('train', 'val', 'test'):                # datasets are small: keep them on the GPU as uint8
        sp = getattr(b, s)
        setattr(b, s, D.Split(sp.x.to(device), sp.y.to(device)))
    rid = run_id(a, b)
    res_path = os.path.join(a.out, 'results', rid + '.json')
    ckpt_path = os.path.join(a.out, 'ckpt', rid + '.pt')
    if os.path.exists(res_path):
        print(f'[skip] {rid} already done')
        return
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    set_seed(a.seed)
    model = build(a, b).to(device)
    fmt = b.fmt_mixer if a.model == 'mixer' else b.fmt_dmp
    amp = a.model == 'mixer' and device == 'cuda' and not a.no_amp
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * ((len(b.train.y) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05,
                                                anneal_strategy='cos')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    n_params = sum(p.numel() for p in model.parameters())
    state = {'epoch': 0, 'history': [], 'best_val': -1.0, 'best_epoch': -1, 'best_state': None}

    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        scaler.load_state_dict(ck['scaler'])
        state = ck['state']
        torch.set_rng_state(ck['rng'])
        print(f'[resume] {rid} from epoch {state["epoch"]}')

    print(f'=== {rid} | params {n_params:,} | train {len(b.train.y)} val {len(b.val.y)} '
          f'test {len(b.test.y)} | {device}', flush=True)
    t_start = time.time()
    for ep in range(state['epoch'], a.epochs):
        model.train()
        t0 = time.time()
        tl, tc, tn = 0.0, 0, 0
        for x, y in batches(b.train, a.bs, True, device):
            if b.augment is not None:
                x = b.augment(x)
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                out = model(fmt(x)).float()
                loss = nn.functional.cross_entropy(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            reset_states(model)
            tl += loss.item() * len(y)
            tc += (out.argmax(1) == y).sum().item()
            tn += len(y)
        ep_time = time.time() - t0
        v = evaluate(model, b.val, fmt, a, device, amp)
        state['history'].append({'epoch': ep + 1, 'train_loss': tl / tn, 'train_acc': tc / tn,
                                 'val_loss': v['loss'], 'val_acc': v['acc'], 'epoch_time_s': ep_time})
        if v['acc'] > state['best_val']:
            state['best_val'], state['best_epoch'] = v['acc'], ep + 1
            state['best_state'] = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        state['epoch'] = ep + 1
        print(f'ep {ep + 1:3d}/{a.epochs} | train {tc / tn:.4f} loss {tl / tn:.4f} | val {v["acc"]:.4f} '
              f'| best {state["best_val"]:.4f}@{state["best_epoch"]} | {ep_time:.0f}s', flush=True)
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'scaler': scaler.state_dict(), 'state': state, 'rng': torch.get_rng_state()}, ckpt_path)
        if a.time_limit and (time.time() - t_start) / 60 > a.time_limit and ep + 1 < a.epochs:
            print(f'[pause] time limit reached; rerun the same command to resume')
            sys.exit(3)

    model.load_state_dict(state['best_state'])
    test = evaluate(model, b.test, fmt, a, device, amp, rates=True)
    result = {'run_id': rid, 'dataset': b.name, 'task': a.dataset, 'model': a.model, 'seed': a.seed,
              'params': n_params, 'best_epoch': state['best_epoch'], 'best_val_acc': state['best_val'],
              'test_acc': test['acc'], 'test_loss': test['loss'], 'firing_rate': test['firing_rate'],
              'mean_epoch_time_s': float(np.mean([h['epoch_time_s'] for h in state['history']])),
              'history': state['history'], 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu',
              'args': {k: v for k, v in vars(a).items() if k not in ('out', 'data_root')}}
    if a.model == 'dmp' and a.gamma != 0:
        model.set_memory_gate(0.0)               # same trained weights, slow memory switched off
        result['test_acc_memory_off'] = evaluate(model, b.test, fmt, a, device, amp)['acc']
    with open(res_path, 'w') as f:
        json.dump(result, f, indent=1)
    os.remove(ckpt_path)
    print(f'RESULT {rid}: test_acc={test["acc"]:.4f} (best val {state["best_val"]:.4f} @ ep '
          f'{state["best_epoch"]}) params={n_params:,} firing={test["firing_rate"]:.3f}'
          + (f' memory_off={result["test_acc_memory_off"]:.4f}' if 'test_acc_memory_off' in result else ''),
          flush=True)


if __name__ == '__main__':
    main()
