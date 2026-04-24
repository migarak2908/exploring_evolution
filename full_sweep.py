"""
Full experiment sweep to replicate and extend Taylor-Davies et al. (2024).

Structure
---------
1. Paper replication sweep:
   - Vary each infant parameter independently (eat_prop, eat_prob, move_prob)
   - Each condition run with feeding enabled AND disabled (to compute b̂)
   - CNN-only (no LSTM), kin recognition on, proximal birth (paper defaults)

2. Ablation conditions (at default infant params):
   - No kin recognition
   - Random birth location

3. Extension (optional, uncomment):
   - Same sweep but with LSTM

Toggle what runs by editing SWEEP_PARAMS, ABLATIONS, and SEEDS below.
"""
import sys
sys.path.insert(0, 'EcoEvoJax')

import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import wandb

from EcoEvoJax.source.gridworld import Gridworld
from EcoEvoJax.source.utils import VideoWriter

# ── Sweep configuration ─────────────────────────────────

SEEDS = [0, 1, 2]   # seeds per condition; reduce to [0] for quick runs

INFANT_PARAM_VALUES = [0.1, 0.25, 0.5, 0.75, 1.0]

# Each entry: (swept_param, fixed_overrides)
SWEEP_PARAMS = [
    ('infant_eat_prop',  {}),
    ('infant_eat_prob',  {}),
    ('infant_move_prob', {}),
]

# Ablation conditions run at default infant params (all 1.0)
ABLATIONS = [
    dict(use_lstm=False, kin_recognition=False, proximal_reprod=True,  label='no-kin'),
    dict(use_lstm=False, kin_recognition=True,  proximal_reprod=False, label='rand-birth'),
    # Uncomment for LSTM extension:
    # dict(use_lstm=True, kin_recognition=True, proximal_reprod=True, label='lstm'),
]

# ── Base config (paper defaults) ────────────────────────
BASE_CONFIG = dict(
    use_lstm             = False,   # paper uses CNN-only
    kin_recognition      = True,
    proximal_reprod      = True,
    nb_agents            = 2500,
    SX                   = 100,
    SY                   = 100,
    init_food            = 1000,
    energy_decay         = 0.1,
    max_ener             = 200.,
    food_value           = 20.,
    action_cost          = 1.0,
    feeding_transfer     = 20.,
    energy_reproduce     = 80.,
    energy_reproduce_cost= 20.,
    energy_initial       = 65.,
    max_age              = 500,
    wall_kill            = False,
    infant_threshold     = 100,
    simple_regrowth      = True,
    regrowth_prob        = 0.003,
    infant_eat_prop      = 1.0,
    infant_eat_prob      = 1.0,
    infant_move_prob     = 1.0,
    n_steps              = 500_000,
    log_every            = 500,
    video_every          = 50_000,
    video_length         = 200,
    checkpoint_every     = 100_000,
    checkpoint_dir       = 'checkpoints',
    project              = 'kin-selection',
)
# ────────────────────────────────────────────────────────


def run_name(cfg):
    parts = [
        cfg.get('label', ''),
        'lstm' if cfg['use_lstm'] else 'no-lstm',
        'kin'  if cfg['kin_recognition'] else 'no-kin',
        'prox' if cfg['proximal_reprod']  else 'rand-birth',
        f"ep{cfg['infant_eat_prop']:.2f}",
        f"eb{cfg['infant_eat_prob']:.2f}",
        f"mp{cfg['infant_move_prob']:.2f}",
        f"feed{int(cfg['feeding_transfer'])}",
        f"s{cfg['seed']}",
    ]
    return '_'.join(p for p in parts if p)


def compute_metrics(state):
    alive = (state.agents.alive > 0).astype(jnp.float32)
    n_alive = alive.sum()
    eps = 1e-10

    mean_energy       = (state.agents.energy * alive).sum() / (n_alive + eps)
    selectivity       = (
        state.agents.n_fed_offspring / (state.agents.n_fed_total + eps)
        - state.agents.n_faced_offspring / (state.agents.n_faced_agent + eps)
    )

    fed_mask = (state.agents.n_fed_total > 0).astype(jnp.float32) * alive
    n_feeders = fed_mask.sum()
    mean_selectivity = (selectivity * fed_mask).sum() / (n_feeders + eps)
    mean_feeding      = (state.agents.n_fed_total.astype(jnp.float32) * alive).sum() / (n_alive + eps)
    infant_survival   = float(state.total_survived_infancy) / (float(state.total_born) + 1e-10)
    return {
        'population':           int(n_alive),
        'mean_energy':          float(mean_energy),
        'mean_selectivity':     float(mean_selectivity),
        'mean_feeding_events':  float(mean_feeding),
        'infant_survival_rate': float(infant_survival),
    }


def render_frame(state):
    grid = np.array(state.state)
    rgb  = np.ones((grid.shape[0], grid.shape[1], 3), dtype=np.float32)
    food    = np.clip(grid[:, :, 1], 0, 1)
    rgb[:, :, 0] -= food;  rgb[:, :, 2] -= food
    agents  = np.clip(grid[:, :, 0], 0, 1)
    rgb[:, :, 0] -= agents; rgb[:, :, 1] -= agents; rgb[:, :, 2] -= agents
    infants = np.clip(grid[:, :, 3], 0, 1)
    rgb[:, :, 2] -= infants * 0.6
    rgb = np.clip(rgb, 0, 1)
    rgb = np.repeat(rgb, 2, axis=0)
    rgb = np.repeat(rgb, 2, axis=1)
    return rgb


def run_condition(cfg):
    name = run_name(cfg)
    wandb.init(project=cfg['project'], name=name, config=cfg, reinit=True)

    env = Gridworld(
        nb_agents             = cfg['nb_agents'],
        SX                    = cfg['SX'],
        SY                    = cfg['SY'],
        init_food             = cfg['init_food'],
        use_lstm              = cfg['use_lstm'],
        kin_recognition       = cfg['kin_recognition'],
        proximal_reprod       = cfg['proximal_reprod'],
        energy_decay          = cfg['energy_decay'],
        max_ener              = cfg['max_ener'],
        food_value            = cfg['food_value'],
        action_cost           = cfg['action_cost'],
        feeding_transfer      = cfg['feeding_transfer'],
        energy_reproduce      = cfg['energy_reproduce'],
        energy_reproduce_cost = cfg['energy_reproduce_cost'],
        energy_initial        = cfg['energy_initial'],
        max_age               = cfg['max_age'],
        wall_kill             = cfg['wall_kill'],
        infant_threshold      = cfg['infant_threshold'],
        simple_regrowth       = cfg['simple_regrowth'],
        regrowth_prob         = cfg['regrowth_prob'],
        infant_eat_prop       = cfg['infant_eat_prop'],
        infant_eat_prob       = cfg['infant_eat_prob'],
        infant_move_prob      = cfg['infant_move_prob'],
    )

    key   = random.PRNGKey(cfg['seed'])
    state = env.reset(key)

    print(f"\n{'='*60}\nRun: {name}\n{'='*60}")

    vid = None; vid_path = None; vid_step = 0
    metrics_history = []

    for step in range(1, cfg['n_steps'] + 1):
        state, rewards, energy = env.step(state)

        # video
        if step % cfg['video_every'] == 1:
            vid_step = step
            vid_path = os.path.join(cfg['checkpoint_dir'], f"{name}_step{step}.mp4")
            os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
            vid = VideoWriter(vid_path, fps=20.0)
        if vid is not None:
            vid.add(render_frame(state))
            if (step - vid_step + 1) >= cfg['video_length']:
                vid.close()
                wandb.log({f'video_step{step}': wandb.Video(vid_path, fps=20, format='mp4')}, step=step)
                vid = None

        # metrics
        if step % cfg['log_every'] == 0:
            m = compute_metrics(state)
            m['step'] = step
            metrics_history.append(m)
            wandb.log(m, step=step)
            print(
                f"step {step:>7} | pop={m['population']:>4} | "
                f"energy={m['mean_energy']:>6.1f} | "
                f"sel={m['mean_selectivity']:>7.4f} | "
                f"surv={m['infant_survival_rate']:.3f}"
            )

        # checkpoint
        if step % cfg['checkpoint_every'] == 0:
            ckpt = os.path.join(cfg['checkpoint_dir'], f"{name}_step{step}.pkl")
            os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
            with open(ckpt, 'wb') as f:
                pickle.dump(state, f)

    if vid is not None:
        vid.close()
        wandb.log({'video': wandb.Video(vid_path, fps=20, format='mp4')}, step=cfg['n_steps'])

    # save final state + metrics history
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    with open(os.path.join(cfg['checkpoint_dir'], f"{name}_final.pkl"), 'wb') as f:
        pickle.dump(state, f)
    with open(os.path.join(cfg['checkpoint_dir'], f"{name}_metrics.pkl"), 'wb') as f:
        pickle.dump({'config': cfg, 'metrics': metrics_history}, f)

    wandb.finish()


def build_conditions():
    conditions = []

    # ── 1. Paper replication sweep ───────────────────────
    for param, overrides in SWEEP_PARAMS:
        for val in INFANT_PARAM_VALUES:
            for seed in SEEDS:
                # feeding enabled
                cfg = {**BASE_CONFIG, **overrides, param: val, 'seed': seed,
                       'label': f'sweep-{param}'}
                conditions.append(cfg)
                # feeding disabled (for b̂ baseline)
                cfg_no_feed = {**cfg, 'feeding_transfer': 0., 'label': f'nofeed-{param}'}
                conditions.append(cfg_no_feed)

    # ── 2. Ablations at default infant params ────────────
    for abl in ABLATIONS:
        for seed in SEEDS:
            cfg = {**BASE_CONFIG, **abl, 'seed': seed}
            conditions.append(cfg)
            cfg_no_feed = {**cfg, 'feeding_transfer': 0., 'label': cfg.get('label','') + '-nofeed'}
            conditions.append(cfg_no_feed)

    return conditions


def already_done(cfg):
    """Return True if this run's metrics file already exists."""
    path = os.path.join(cfg['checkpoint_dir'], f"{run_name(cfg)}_metrics.pkl")
    return os.path.exists(path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', default=None,
                        help='Override checkpoint directory (e.g. /gdrive/MyDrive/kin-selection)')
    parser.add_argument('--start_from', type=int, default=0,
                        help='Skip the first N conditions (useful if you know where you left off)')
    args = parser.parse_args()

    conditions = build_conditions()

    # override checkpoint_dir if provided (e.g. Google Drive path)
    if args.checkpoint_dir:
        for cfg in conditions:
            cfg['checkpoint_dir'] = args.checkpoint_dir

    total = len(conditions)
    skipped = 0
    print(f"Total runs: {total}  |  starting from index {args.start_from}")

    for i, cfg in enumerate(conditions):
        if i < args.start_from:
            continue
        name = run_name(cfg)
        if already_done(cfg):
            print(f"[{i+1}/{total}] SKIP (already done): {name}")
            skipped += 1
            continue
        print(f"\n[{i+1}/{total}] {name}")
        run_condition(cfg)

    print(f"\nDone. {skipped}/{total} runs skipped (already completed).")


if __name__ == '__main__':
    main()
