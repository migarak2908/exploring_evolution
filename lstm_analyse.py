"""
Analysis for the LSTM kin-recognition hypothesis.

Hypothesis: LSTM agents can evolve selective feeding without explicit kin recognition
by using temporal memory to implicitly identify offspring.

Outputs:
  bar_selectivity       — final mean selectivity across 4 conditions
  bar_infant_survival   — final infant survival rate across 4 conditions
  ts_selectivity        — selectivity over training (mean ± std across seeds)
  ts_population         — population over training
  stdout stats          — Mann-Whitney test + % of baseline selectivity reached

Usage:
    python lstm_analyse.py [--checkpoint_dir checkpoints_lstm] [--output_dir figures_lstm]
"""
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

CONDITION_ORDER = ['cnn_kin', 'cnn_nokin', 'lstm_kin', 'lstm_nokin']
COLORS = {
    'cnn_kin':    '#1f77b4',
    'cnn_nokin':  '#aec7e8',
    'lstm_kin':   '#ff7f0e',
    'lstm_nokin': '#ffbb78',
}
LABELS = {
    'cnn_kin':    'CNN + kin',
    'cnn_nokin':  'CNN, no kin',
    'lstm_kin':   'LSTM + kin',
    'lstm_nokin': 'LSTM, no kin',
}

plt.rcParams.update({'font.size': 9, 'figure.autolayout': True})


def load_records(checkpoint_dir):
    records = []
    for fname in sorted(os.listdir(checkpoint_dir)):
        if fname.endswith('_metrics.pkl'):
            with open(os.path.join(checkpoint_dir, fname), 'rb') as f:
                records.append(pickle.load(f))
    print(f"Loaded {len(records)} run(s) from {checkpoint_dir}")
    return records


def get_condition(cfg):
    lstm = cfg['use_lstm']
    kin  = cfg['kin_recognition']
    if not lstm and kin:      return 'cnn_kin'
    if not lstm and not kin:  return 'cnn_nokin'
    if lstm and kin:          return 'lstm_kin'
    if lstm and not kin:      return 'lstm_nokin'


def group_by_condition(records):
    groups = {c: [] for c in CONDITION_ORDER}
    for r in records:
        cond = get_condition(r['config'])
        if cond:
            groups[cond].append(r)
    return groups


def final_val(record, metric, last_n=100):
    m = record['metrics']
    if not m:
        return np.nan
    window = m[-last_n:]
    vals = [step[metric] for step in window if metric in step]
    return np.mean(vals) if vals else np.nan


def plot_bar(groups, metric, ylabel, title, output_dir, fname):
    present = [c for c in CONDITION_ORDER if groups[c]]
    means = [np.nanmean([final_val(r, metric) for r in groups[c]]) for c in present]
    stds  = [np.nanstd( [final_val(r, metric) for r in groups[c]]) for c in present]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(present))
    ax.bar(x, means, yerr=stds, capsize=4,
           color=[COLORS[c] for c in present],
           edgecolor='black', linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in present], rotation=15, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # dashed line separating CNN from LSTM pairs
    ax.axvline(x=1.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)

    for path in [
        os.path.join(output_dir, fname + '.pdf'),
        os.path.join(output_dir, fname + '.png'),
    ]:
        plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {fname}")


def plot_time_series(groups, metric, ylabel, title, output_dir, fname):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for cond in CONDITION_ORDER:
        runs = groups[cond]
        if not runs:
            continue
        steps = [m['step'] for m in runs[0]['metrics']]
        try:
            all_vals = np.array([[m[metric] for m in r['metrics']] for r in runs])
        except ValueError:
            # runs stopped early at different lengths — plot each individually
            for r in runs:
                s = [m['step'] for m in r['metrics']]
                v = [m[metric] for m in r['metrics']]
                ax.plot(s, v, color=COLORS[cond], alpha=0.6, label=LABELS[cond])
            continue
        mean = all_vals.mean(axis=0)
        std  = all_vals.std(axis=0)
        ax.plot(steps, mean, label=LABELS[cond], color=COLORS[cond])
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=COLORS[cond])

    ax.set_xlabel('Step')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # deduplicate legend entries
    handles, lbls = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, lbls):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), fontsize=7)

    for path in [
        os.path.join(output_dir, fname + '.pdf'),
        os.path.join(output_dir, fname + '.png'),
    ]:
        plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {fname}")


def run_stats(groups):
    print(f"\n{'─'*60}")
    print("  Hypothesis: LSTM+no-kin selectivity > CNN+no-kin selectivity")
    print(f"{'─'*60}")

    for cond in CONDITION_ORDER:
        vals = [final_val(r, 'mean_selectivity') for r in groups[cond]]
        vals = [v for v in vals if not np.isnan(v)]
        n = len(vals)
        mean = np.mean(vals) if vals else float('nan')
        std  = np.std(vals)  if vals else float('nan')
        print(f"  {LABELS[cond]:<18}  n={n}  mean={mean:.4f}  std={std:.4f}")

    print()
    a = [final_val(r, 'mean_selectivity') for r in groups['lstm_nokin']]
    b = [final_val(r, 'mean_selectivity') for r in groups['cnn_nokin']]
    a = [v for v in a if not np.isnan(v)]
    b = [v for v in b if not np.isnan(v)]

    if len(a) >= 2 and len(b) >= 2:
        stat, p = stats.mannwhitneyu(a, b, alternative='greater')
        print(f"  Mann-Whitney U (one-sided, LSTM > CNN): U={stat:.1f}, p={p:.4f}")
        print(f"  >> {'SIGNIFICANT' if p < 0.05 else 'Not significant'} (α=0.05)")
    else:
        diff = (np.mean(a) - np.mean(b)) if (a and b) else float('nan')
        print(f"  Raw difference LSTM−CNN (no-kin): {diff:+.4f}")
        print(f"  (run with ≥2 seeds per condition for a statistical test)")

    base = [final_val(r, 'mean_selectivity') for r in groups['cnn_kin']]
    base = [v for v in base if not np.isnan(v)]
    if a and base:
        pct = 100 * np.mean(a) / np.mean(base)
        print(f"\n  LSTM+no-kin reaches {pct:.1f}% of CNN+kin (paper baseline) selectivity")

    print(f"{'─'*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', default='checkpoints_lstm')
    parser.add_argument('--output_dir',     default='figures_lstm')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    records = load_records(args.checkpoint_dir)
    if not records:
        print("No metric files found — run lstm_sweep.py first.")
        return

    groups = group_by_condition(records)
    for c in CONDITION_ORDER:
        print(f"  {LABELS[c]}: {len(groups[c])} run(s)")

    plot_bar(groups, 'mean_selectivity',     'Mean selectivity',     'Final selectivity by condition',     args.output_dir, 'bar_selectivity')
    plot_bar(groups, 'infant_survival_rate', 'Infant survival rate', 'Final infant survival by condition', args.output_dir, 'bar_infant_survival')
    plot_time_series(groups, 'mean_selectivity', 'Mean selectivity', 'Selectivity over training', args.output_dir, 'ts_selectivity')
    plot_time_series(groups, 'population',       'Population',       'Population over training',  args.output_dir, 'ts_population')

    run_stats(groups)
    print(f"\nOutputs saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
