"""Aggregate bench results: tables (mean ± std over seeds) and plots.

    python -m bench.summarize --out /content/drive/MyDrive/XLDL_bench
"""

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def load(out):
    rows = []
    for p in sorted(glob.glob(os.path.join(out, 'results', '*.json'))):
        r = json.load(open(p))
        rows.append({
            'dataset': r['dataset'], 'config': r['run_id'].split('__')[1], 'model': r['model'],
            'seed': r['seed'], 'test_acc': 100 * r['test_acc'], 'val_acc': 100 * r['best_val_acc'],
            'mem_off_acc': 100 * r['test_acc_memory_off'] if 'test_acc_memory_off' in r else None,
            'params': r['params'], 'epoch_s': r['mean_epoch_time_s'], 'firing': r['firing_rate'],
            'epochs': len(r['history']), 'best_epoch': r['best_epoch'],
            'train_acc_last': 100 * r['history'][-1]['train_acc'], 'history': r['history'],
        })
    return pd.DataFrame(rows)


def table(df):
    g = df.groupby(['dataset', 'config'])
    t = g.agg(model=('model', 'first'), seeds=('seed', 'count'), test_mean=('test_acc', 'mean'),
              test_std=('test_acc', 'std'), val=('val_acc', 'mean'), train_last=('train_acc_last', 'mean'),
              mem_off=('mem_off_acc', 'mean'), params=('params', 'first'), epoch_s=('epoch_s', 'mean'),
              firing=('firing', 'mean')).reset_index()
    t['test_std'] = t['test_std'].fillna(0)
    t['gen_gap'] = t['train_last'] - t['test_mean']        # overfitting indicator
    return t


def to_markdown(t):
    lines = ['| dataset | config | seeds | test acc (%) | val | train (last) | gap | memory-off | params | s/epoch | firing |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _, r in t.iterrows():
        mem = f'{r.mem_off:.2f}' if pd.notna(r.mem_off) else '–'
        lines.append(f'| {r.dataset} | {r.config} | {r.seeds} | {r.test_mean:.2f} ± {r.test_std:.2f} | {r.val:.2f} | '
                     f'{r.train_last:.2f} | {r.gen_gap:.1f} | {mem} | {r.params:,} | {r.epoch_s:.0f} | {r.firing:.3f} |')
    return '\n'.join(lines)


def plot_recall(df, out):
    rec = df[df.dataset.str.startswith('recall_D')].copy()
    if rec.empty:
        return
    rec['delay'] = rec.dataset.str.extract(r'D(\d+)').astype(int)
    # theta varies with delay in the default DMP runs; collapse "th<T>" to one curve label
    rec['curve'] = rec.config.apply(lambda c: re.sub(r'_th[\d.]+', '_th=T', c) if 'th' in c else c)
    fixed = rec.config.str.contains('fixedtheta')
    rec.loc[fixed, 'curve'] = rec.loc[fixed, 'config'].str.replace(r'_fixedtheta', '', regex=True)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, g in rec.groupby('curve'):
        m = g.groupby('delay').test_acc.agg(['mean', 'std']).reset_index()
        ax.errorbar(m.delay, m['mean'], yerr=m['std'].fillna(0), marker='o', capsize=3, label=name)
    ax.axhline(10, ls=':', c='gray', label='chance')
    ax.set(xscale='log', xlabel='delay D (steps)', ylabel='test accuracy (%)', title='Delayed recall: memory vs delay')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'plots', 'recall_vs_delay.png'), dpi=150)
    plt.close(fig)


def plot_curves(df, out):
    for ds, g in df.groupby('dataset'):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for (cfg, seed), r in g.groupby(['config', 'seed']):
            h = r.history.iloc[0]
            ax.plot([e['epoch'] for e in h], [100 * e['val_acc'] for e in h], label=f'{cfg} s{seed}', lw=1)
        ax.set(xlabel='epoch', ylabel='val accuracy (%)', title=ds)
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'plots', f'curves_{ds}.png'), dpi=120)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='./bench_out')
    a = p.parse_args()
    df = load(a.out)
    if df.empty:
        print('no results yet')
        return
    os.makedirs(os.path.join(a.out, 'plots'), exist_ok=True)
    t = table(df)
    t.to_csv(os.path.join(a.out, 'summary.csv'), index=False)
    md = to_markdown(t)
    open(os.path.join(a.out, 'summary.md'), 'w', encoding='utf-8').write(md + '\n')
    plot_recall(df, a.out)
    plot_curves(df, a.out)
    print(md)


if __name__ == '__main__':
    main()
