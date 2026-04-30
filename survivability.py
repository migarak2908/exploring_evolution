"""
Sweep infant_move_prob to find the value that maximises adult feeding and selectivity.

Varies infant_move_prob across [0.1, 0.25, 0.5, 0.75, 1.0] with LSTM+kin (full capability),
10 seeds per value. Use the results to pick the infant_move_prob for lstm_sweep.py.
"""
import sys
sys.path.insert(0, 'EcoEvoJax')

import os
import pickle
import numpy as np
import jax.numpy as jnp
from jax import random
import wandb

from EcoEvoJax.source.gridworld import Gridworld
from EcoEvoJax.source.utils import VideoWriter

# ── Sweep configuration ─────────────────────────────────
SEEDS = list(range(10))

INFANT_MOVE_VALUES = [0.1, 0.25, 0.5, 0.75, 1.0]

BASE_CONFIG = dict(
    use_lstm             = True,
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
    video_every          = 100_000,
    video_length         = 200,
    checkpoint_every     = 100_000,
    checkpoint_dir       = 'checkpoints_survivability',
    project              = 'survivability-sweep',
    extinct_threshold    = 10,
    extinct_patience     = 5,
)
# ─────────────────────────────────────────────────────────


def run_name(cfg):
    return f"lstm_kin_mp{cfg['infant_move_prob']:.2f}_s{cfg['seed']}"


def compute_metrics(state, infant_threshold):
    alive = (state.agents.alive > 0).astype(jnp.float32)
    n_alive = alive.sum()
    eps = 1e-10

    adult_mask = alive * (state.agents.time_alive >= infant_threshold).astype(jnp.float32)
    n_adults = adult_mask.sum()

    mean_energy = (state.agents.energy * alive).sum() / (n_alive + eps)
    selectivity = (
        state.agents.n_fed_offspring / (state.agents.n_fed_total + eps)
        - state.agents.n_faced_offspring / (state.agents.n_faced_agent + eps)
    )
    fed_mask = (state.agents.n_fed_total > 0).astype(jnp.float32) * alive
    mean_selectivity = (selectivity * fed_mask).sum() / (fed_mask.sum() + eps)
    mean_feeding = (state.agents.n_fed_total.astype(jnp.float32) * adult_mask).sum() / (n_adults + eps)
    infant_survival = float(state.total_survived_infancy) / (float(state.total_born) + 1e-10)

    return {
        'population':           int(n_alive),
        'mean_energy':          float(mean_energy),
        'mean_selectivity':     float(mean_selectivity),
        'mean_feeding_events':  float(mean_feeding),
        'infant_survival_rate': float(infant_survival),
    }


def render_frame(state):
    grid = np.array(state.state)
    rgb = np.ones((grid.shape[0], grid.shape[1], 3), dtype=np.float32)
    food    = np.clip(grid[:, :, 1], 0, 1)
    rgb[:, :, 0] -= food;    rgb[:, :, 2] -= food
    agents  = np.clip(grid[:, :, 0], 0, 1)
    rgb[:, :, 0] -= agents;  rgb[:, :, 1] -= agents;  rgb[:, :, 2] -= agents
    infants = np.clip(grid[:, :, 3], 0, 1)
    rgb[:, :, 2] -= infants * 0.6
    rgb = np.clip(rgb, 0, 1)
    rgb = np.repeat(rgb, 2, axis=0)
    rgb = np.repeat(rgb, 2, axis=1)
    return rgb


def already_done(cfg):
    path = os.path.join(cfg['checkpoint_dir'], f"{run_name(cfg)}_metrics.pkl")
    return os.path.exists(path)


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

    key = random.PRNGKey(cfg['seed'])
    state = env.reset(key)

    print(f"\n{'='*60}\nRun: {name}\n{'='*60}")

    vid = None; vid_path = None; vid_step = 0
    metrics_history = []
    extinct_count = 0

    for step in range(1, cfg['n_steps'] + 1):
        state, rewards, energy = env.step(state)

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

        if step % cfg['log_every'] == 0:
            m = compute_metrics(state, cfg['infant_threshold'])
            m['step'] = step
            metrics_history.append(m)
            wandb.log(m, step=step)
            print(
                f"step {step:>7} | pop={m['population']:>4} | "
                f"energy={m['mean_energy']:>6.1f} | "
                f"sel={m['mean_selectivity']:>7.4f} | "
                f"feed={m['mean_feeding_events']:>6.2f} | "
                f"surv={m['infant_survival_rate']:.3f}"
            )

            if m['population'] <= cfg['extinct_threshold']:
                extinct_count += 1
                if extinct_count >= cfg['extinct_patience']:
                    print(f"  >> Population extinct — stopping early at step {step}")
                    wandb.log({'extinct': True, 'extinct_step': step}, step=step)
                    break
            else:
                extinct_count = 0

        if step % cfg['checkpoint_every'] == 0:
            ckpt = os.path.join(cfg['checkpoint_dir'], f"{name}_step{step}.pkl")
            os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
            with open(ckpt, 'wb') as f:
                pickle.dump(state, f)

    if vid is not None:
        vid.close()
        wandb.log({'video': wandb.Video(vid_path, fps=20, format='mp4')}, step=step)

    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    with open(os.path.join(cfg['checkpoint_dir'], f"{run_name(cfg)}_metrics.pkl"), 'wb') as f:
        pickle.dump({'config': cfg, 'metrics': metrics_history}, f)

    wandb.finish()


def build_conditions():
    conditions = []
    for val in INFANT_MOVE_VALUES:
        for seed in SEEDS:
            conditions.append({**BASE_CONFIG, 'infant_move_prob': val, 'seed': seed})
    return conditions


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', default=None)
    parser.add_argument('--start_from', type=int, default=0)
    args = parser.parse_args()

    conditions = build_conditions()

    if args.checkpoint_dir:
        for cfg in conditions:
            cfg['checkpoint_dir'] = args.checkpoint_dir

    total = len(conditions)
    skipped = 0
    print(f"Total runs: {total}  ({len(INFANT_MOVE_VALUES)} values × {len(SEEDS)} seeds)")

    for i, cfg in enumerate(conditions):
        if i < args.start_from:
            continue
        name = run_name(cfg)
        if already_done(cfg):
            print(f"[{i+1}/{total}] SKIP: {name}")
            skipped += 1
            continue
        print(f"\n[{i+1}/{total}] {name}")
        run_condition(cfg)

    print(f"\nDone. {skipped}/{total} runs skipped.")


if __name__ == '__main__':
    main()
