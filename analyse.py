"""
Analysis script — replicates Taylor-Davies et al. (2024) figures and tables.

Loads metrics pickles saved by full_sweep.py and produces:
  - Figure 2 equivalent: infant survival rate vs infant parameters (no-feeding baseline)
  - Figure 3 equivalent: log(feeding amount) and log(selectivity) vs b̂
  - Figure 4 equivalent: selectivity vs feeding amount
  - Figure 5 equivalent: ablation scatter plots
  - Tables 1 & 2: OLS regression coefficients, R², p-values

Usage:
    python analyse.py --checkpoint_dir checkpoints --output_dir figures
"""
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import pandas as pd

# ── Plot style ───────────────────────────────────────────
plt.rcParams.update({
    'font.size': 8,
    'legend.fontsize': 7,
    'figure.autolayout': True,
})
FIG_W = 3.5   # single column width inches
FIG_H = 2.8

PARAM_LABELS = {
    'infant_eat_prop':  'Food energy proportion',
    'infant_eat_prob':  'Eat success probability',
    'infant_move_prob': 'Move success probability',
}
CONDITION_COLORS = {
    'infant_eat_prop':  '#1f77b4',
    'infant_eat_prob':  '#ff7f0e',
    'infant_move_prob': '#2ca02c',
}
# ────────────────────────────────────────────────────────


def load_metrics(checkpoint_dir):
    """Load all *_metrics.pkl files. Returns list of dicts with config + metrics."""
    records = []
    for fname in os.listdir(checkpoint_dir):
        if fname.endswith('_metrics.pkl'):
            with open(os.path.join(checkpoint_dir, fname), 'rb') as f:
                data = pickle.load(f)
            records.append(data)
    print(f"Loaded {len(records)} metric files from {checkpoint_dir}")
    return records


def final_metrics(record):
    """Extract final-step metrics from a record."""
    return record['metrics'][-1] if record['metrics'] else {}


def get_param_value(cfg, param):
    return cfg.get(param, 1.0)


def build_dataframe(records):
    """
    Build a flat DataFrame of final metrics per run.
    Computes b̂ = infant_survival_rate(feeding) - infant_survival_rate(no-feeding)
    by matching feeding/no-feeding pairs on (param, value, seed, condition).
    """
    rows = []
    for r in records:
        cfg = r['config']
        fm  = final_metrics(r)
        if not fm:
            continue
        rows.append({
            'label':            cfg.get('label', ''),
            'use_lstm':         cfg['use_lstm'],
            'kin_recognition':  cfg['kin_recognition'],
            'proximal_reprod':  cfg['proximal_reprod'],
            'feeding_transfer': cfg['feeding_transfer'],
            'infant_eat_prop':  cfg['infant_eat_prop'],
            'infant_eat_prob':  cfg['infant_eat_prob'],
            'infant_move_prob': cfg['infant_move_prob'],
            'seed':             cfg['seed'],
            **{k: fm.get(k, np.nan) for k in [
                'population', 'mean_energy', 'mean_selectivity',
                'mean_feeding_events', 'infant_survival_rate', 'mean_nb_offspring'
            ]},
        })
    df = pd.DataFrame(rows)

    # compute b̂: match feeding vs no-feeding pairs
    merge_keys = ['use_lstm', 'kin_recognition', 'proximal_reprod',
                  'infant_eat_prop', 'infant_eat_prob', 'infant_move_prob', 'seed']
    feed    = df[df['feeding_transfer'] > 0].copy()
    no_feed = df[df['feeding_transfer'] == 0][merge_keys + ['infant_survival_rate']].copy()
    no_feed = no_feed.rename(columns={'infant_survival_rate': 'baseline_survival'})
    df = feed.merge(no_feed, on=merge_keys, how='left')
    df['b_hat'] = df['infant_survival_rate'] - df['baseline_survival'].fillna(0)

    return df


def ols_table(df, x_col, y_col, group_col, group_vals, title):
    """Run OLS for each group and print a table."""
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"  x={x_col}  y={y_col}")
    print(f"{'─'*65}")
    print(f"  {'Condition':<25} {'β':>8} {'R²':>8} {'p':>10}")
    print(f"{'─'*65}")
    results = []
    for gval in group_vals + ['combined']:
        if gval == 'combined':
            sub = df.dropna(subset=[x_col, y_col])
        else:
            sub = df[df[group_col] == gval].dropna(subset=[x_col, y_col])
        if len(sub) < 3:
            continue
        x = sub[x_col].values
        y = sub[y_col].values
        slope, intercept, r, p, se = stats.linregress(x, y)
        r2 = r ** 2
        label = gval if gval == 'combined' else PARAM_LABELS.get(gval, gval)
        print(f"  {label:<25} {slope:>8.4f} {r2:>8.4f} {p:>10.4e}")
        results.append({'condition': label, 'beta': slope, 'R2': r2, 'p': p})
    print(f"{'─'*65}")
    return pd.DataFrame(results)


# ── Figure 2: infant survival vs infant params (no-feeding) ─
def fig2(records, output_dir):
    no_feed = [r for r in records if r['config']['feeding_transfer'] == 0
               and r['config']['kin_recognition']
               and r['config']['proximal_reprod']
               and not r['config']['use_lstm']]
    if not no_feed:
        print("fig2: no no-feeding records found, skipping")
        return

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W * 3, FIG_H), sharey=True)
    for ax, param in zip(axes, ['infant_eat_prop', 'infant_eat_prob', 'infant_move_prob']):
        vals, survivals = [], []
        for r in no_feed:
            fm = final_metrics(r)
            if not fm:
                continue
            vals.append(r['config'][param])
            survivals.append(fm.get('infant_survival_rate', np.nan))
        if not vals:
            continue
        # group by param value, show mean ± std
        df_tmp = pd.DataFrame({'val': vals, 'surv': survivals})
        grp = df_tmp.groupby('val')['surv']
        means = grp.mean(); stds = grp.std().fillna(0)
        ax.errorbar(means.index, means.values, yerr=stds.values,
                    fmt='o-', color=CONDITION_COLORS[param], capsize=3)
        ax.set_xlabel(PARAM_LABELS[param])
        ax.set_ylabel('Infant survival rate' if ax == axes[0] else '')
        ax.set_xlim(0, 1.05)

    fig.suptitle('Figure 2a: Infant survival vs infant parameters (no feeding)')
    plt.savefig(os.path.join(output_dir, 'fig2_infant_survival.pdf'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig2_infant_survival.png'), dpi=150)
    plt.close()
    print("Saved fig2_infant_survival")


# ── Figure 3: log(feeding) and log(selectivity) vs b̂ ────
def fig3(df, output_dir):
    baseline = df[
        df['kin_recognition'] & df['proximal_reprod'] & ~df['use_lstm']
    ].copy()
    if baseline.empty:
        print("fig3: no baseline records, skipping")
        return

    baseline = baseline[baseline['b_hat'].notna()]
    # identify which param was swept (the one that isn't 1.0)
    def swept_param(row):
        for p in ['infant_eat_prop', 'infant_eat_prob', 'infant_move_prob']:
            if row[p] < 0.99:
                return p
        return 'infant_eat_prop'
    baseline['swept'] = baseline.apply(swept_param, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W * 2, FIG_H))
    for param in ['infant_eat_prop', 'infant_eat_prob', 'infant_move_prob']:
        sub = baseline[baseline['swept'] == param]
        if sub.empty:
            continue
        # average over seeds
        sub_grp = sub.groupby(['infant_eat_prop', 'infant_eat_prob', 'infant_move_prob'])[
            ['b_hat', 'mean_feeding_events', 'mean_selectivity']].mean().reset_index()
        x = sub_grp['b_hat'].values
        for ax, y_col, ylabel in zip(axes,
                                     ['mean_feeding_events', 'mean_selectivity'],
                                     ['Feeding amount (log)', 'Selectivity (log)']):
            y = sub_grp[y_col].values
            mask = (x > 0) & (y > 0)
            if mask.sum() < 2:
                continue
            ax.scatter(np.log(x[mask] + 1e-10), np.log(y[mask] + 1e-10),
                       color=CONDITION_COLORS[param],
                       label=PARAM_LABELS[param], alpha=0.7, s=20)
            # regression line
            m, b, *_ = stats.linregress(np.log(x[mask] + 1e-10), np.log(y[mask] + 1e-10))
            xr = np.linspace(np.log(x[mask].min() + 1e-10), np.log(x[mask].max() + 1e-10), 50)
            ax.plot(xr, m * xr + b, color=CONDITION_COLORS[param], lw=1)
            ax.set_xlabel('log(b̂)')
            ax.set_ylabel(ylabel)

    axes[0].legend(fontsize=6)
    fig.suptitle('Figure 3: Feeding amount and selectivity vs b̂')
    plt.savefig(os.path.join(output_dir, 'fig3_bhat_scatter.pdf'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig3_bhat_scatter.png'), dpi=150)
    plt.close()
    print("Saved fig3_bhat_scatter")


# ── Figure 4: selectivity vs feeding amount ──────────────
def fig4(df, output_dir):
    baseline = df[df['kin_recognition'] & df['proximal_reprod'] & ~df['use_lstm']].copy()
    if baseline.empty:
        print("fig4: no baseline records, skipping")
        return

    def swept_param(row):
        for p in ['infant_eat_prop', 'infant_eat_prob', 'infant_move_prob']:
            if row[p] < 0.99:
                return p
        return 'infant_eat_prop'
    baseline['swept'] = baseline.apply(swept_param, axis=1)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for param in ['infant_eat_prop', 'infant_eat_prob', 'infant_move_prob']:
        sub = baseline[baseline['swept'] == param]
        if sub.empty:
            continue
        x = sub['mean_feeding_events'].values
        y = sub['mean_selectivity'].values
        mask = (x > 0) & np.isfinite(y)
        ax.scatter(x[mask], y[mask], color=CONDITION_COLORS[param],
                   label=PARAM_LABELS[param], alpha=0.7, s=20)
        if mask.sum() >= 2:
            m, b, *_ = stats.linregress(x[mask], y[mask])
            xr = np.linspace(x[mask].min(), x[mask].max(), 50)
            ax.plot(xr, m * xr + b, color=CONDITION_COLORS[param], lw=1)

    ax.set_xlabel('Feeding amount')
    ax.set_ylabel('Selectivity')
    ax.legend(fontsize=6)
    fig.suptitle('Figure 4: Selectivity vs feeding amount')
    plt.savefig(os.path.join(output_dir, 'fig4_selectivity_vs_feeding.pdf'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig4_selectivity_vs_feeding.png'), dpi=150)
    plt.close()
    print("Saved fig4_selectivity_vs_feeding")


# ── Figure 5: ablation scatter ───────────────────────────
def fig5(df, output_dir):
    baseline = df[df['kin_recognition'] & df['proximal_reprod'] & ~df['use_lstm']].copy()
    no_kin   = df[~df['kin_recognition'] & df['proximal_reprod'] & ~df['use_lstm']].copy()
    rand_b   = df[df['kin_recognition'] & ~df['proximal_reprod'] & ~df['use_lstm']].copy()

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W * 2, FIG_H), sharey=True)
    for ax, abl_df, abl_label in zip(axes,
                                      [no_kin, rand_b],
                                      ['No kin recognition', 'Random birth']):
        for sub, label, color, marker in [
            (baseline, 'Baseline', '#1f77b4', 'o'),
            (abl_df,   abl_label,  '#d62728', 'x'),
        ]:
            if sub.empty:
                continue
            x = sub['b_hat'].values
            y = sub['mean_selectivity'].values
            mask = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[mask], y[mask], color=color, marker=marker,
                       label=label, alpha=0.7, s=20)
        ax.set_xlabel('b̂')
        ax.set_ylabel('Selectivity' if ax == axes[0] else '')
        ax.legend(fontsize=6)
        ax.set_title(abl_label)

    fig.suptitle('Figure 5: Ablation results')
    plt.savefig(os.path.join(output_dir, 'fig5_ablations.pdf'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig5_ablations.png'), dpi=150)
    plt.close()
    print("Saved fig5_ablations")


# ── Time series ──────────────────────────────────────────
def plot_time_series(records, output_dir):
    """Plot mean selectivity and population over time per condition group."""
    condition_groups = {}
    for r in records:
        cfg = r['config']
        if cfg['feeding_transfer'] == 0:
            continue
        key = (cfg['use_lstm'], cfg['kin_recognition'], cfg['proximal_reprod'])
        condition_groups.setdefault(key, []).append(r['metrics'])

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W * 2, FIG_H))
    for (use_lstm, kin, prox), runs in condition_groups.items():
        label = f"{'lstm' if use_lstm else 'cnn'}_{'kin' if kin else 'nokin'}_{'prox' if prox else 'rand'}"
        steps = [m['step'] for m in runs[0]] if runs else []
        for metric, ax, ylabel in [
            ('mean_selectivity', axes[0], 'Mean selectivity'),
            ('population',       axes[1], 'Population'),
        ]:
            all_vals = np.array([[m[metric] for m in run] for run in runs])
            mean = all_vals.mean(axis=0)
            std  = all_vals.std(axis=0)
            ax.plot(steps, mean, label=label)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
            ax.set_xlabel('Step')
            ax.set_ylabel(ylabel)

    axes[0].legend(fontsize=6)
    fig.suptitle('Training time series')
    plt.savefig(os.path.join(output_dir, 'time_series.pdf'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'time_series.png'), dpi=150)
    plt.close()
    print("Saved time_series")


# ── Tables ───────────────────────────────────────────────
def tables(df, output_dir):
    """Print and save Tables 1 and 2."""
    baseline = df[df['kin_recognition'] & df['proximal_reprod'] & ~df['use_lstm']].copy()

    def swept_param(row):
        for p in ['infant_eat_prop', 'infant_eat_prob', 'infant_move_prob']:
            if row[p] < 0.99:
                return p
        return 'combined'
    baseline['swept'] = baseline.apply(swept_param, axis=1)

    params = ['infant_eat_prop', 'infant_eat_prob', 'infant_move_prob']

    # Table 1: b̂ vs feeding amount and selectivity
    t1a = ols_table(baseline[baseline['b_hat'] > 0],
                    'b_hat', 'mean_feeding_events', 'swept', params,
                    'Table 1a: log(feeding amount) ~ b̂')
    t1b = ols_table(baseline[baseline['b_hat'] > 0],
                    'b_hat', 'mean_selectivity', 'swept', params,
                    'Table 1b: log(selectivity) ~ b̂')

    # Table 2: feeding amount vs selectivity
    t2 = ols_table(baseline,
                   'mean_feeding_events', 'mean_selectivity', 'swept', params,
                   'Table 2: selectivity ~ feeding amount')

    os.makedirs(output_dir, exist_ok=True)
    t1a.to_csv(os.path.join(output_dir, 'table1a.csv'), index=False)
    t1b.to_csv(os.path.join(output_dir, 'table1b.csv'), index=False)
    t2.to_csv(os.path.join(output_dir, 'table2.csv'), index=False)
    print("\nTables saved to CSV.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--output_dir',     default='figures')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    records = load_metrics(args.checkpoint_dir)
    if not records:
        print("No metric files found. Run full_sweepfull_sweep.py first.")
        return

    df = build_dataframe(records)
    print(f"\nDataFrame: {len(df)} rows, conditions: {df['label'].unique()}")

    fig2(records, args.output_dir)
    fig3(df, args.output_dir)
    fig4(df, args.output_dir)
    fig5(df, args.output_dir)
    plot_time_series(records, args.output_dir)
    tables(df, args.output_dir)

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
