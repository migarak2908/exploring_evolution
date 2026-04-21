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

# ── Ablation config — toggle these ──────────────────────
config = dict(
    # ablations
    use_lstm         = True,   # False for CNN-only ablation
    kin_recognition  = True,   # False to disable is_offspring channel
    proximal_reprod  = True,   # False for random birth placement
    # environment
    nb_agents        = 2500,   # array size; ~833 start alive (1/3)
    SX               = 100,
    SY               = 100,
    init_food        = 1000,   # ~p=0.1 initial food density on 100x100
    # energy / reproduction (paper defaults)
    energy_decay     = 0.1,
    max_ener         = 200.,
    food_value       = 20.,
    action_cost      = 1.0,
    feeding_transfer = 20.,
    energy_reproduce = 85.,
    energy_reproduce_cost = 30.,
    infant_threshold = 100,
    simple_regrowth  = True,
    # infant parameters
    infant_eat_prop  = 1.0,
    infant_eat_prob  = 1.0,
    infant_move_prob = 1.0,
    # training
    n_steps          = 500_000,
    log_every        = 500,
    video_every      = 50_000,
    video_length     = 200,
    checkpoint_every = 50_000,
    checkpoint_dir   = 'checkpoints',
    seed             = 0,
    # wandb
    project          = 'kin-selection',
)
# ────────────────────────────────────────────────────────


def run_name(cfg):
    parts = [
        'lstm'     if cfg['use_lstm']        else 'no-lstm',
        'kin'      if cfg['kin_recognition'] else 'no-kin',
        'prox'     if cfg['proximal_reprod'] else 'rand-birth',
    ]
    return '_'.join(parts)


def compute_metrics(state):
    alive = (state.agents.alive > 0).astype(jnp.float32)
    n_alive = alive.sum()
    eps = 1e-10

    mean_energy = (state.agents.energy * alive).sum() / (n_alive + eps)

    selectivity = (
        state.agents.n_fed_offspring / (state.agents.n_fed_total + eps)
        - state.agents.n_faced_offspring / (state.agents.n_faced_agent + eps)
    )
    fed_mask = (state.agents.n_fed_total > 0).astype(jnp.float32) * alive
    n_feeders = fed_mask.sum()
    mean_selectivity = (selectivity * fed_mask).sum() / (n_feeders + eps)

    mean_feeding = (state.agents.n_fed_total.astype(jnp.float32) * alive).sum() / (n_alive + eps)
    infant_survival = float(state.total_survived_infancy) / (float(state.total_born) + 1e-10)
    return {
        'population':          int(n_alive),
        'mean_energy':         float(mean_energy),
        'mean_selectivity':    float(mean_selectivity),
        'mean_feeding_events': float(mean_feeding),
        'infant_survival_rate':float(infant_survival),
    }


def render_frame(state):
    grid = np.array(state.state)
    rgb = np.ones((grid.shape[0], grid.shape[1], 3), dtype=np.float32)
    food    = np.clip(grid[:, :, 1], 0, 1)
    rgb[:, :, 0] -= food
    rgb[:, :, 2] -= food
    agents  = np.clip(grid[:, :, 0], 0, 1)
    rgb[:, :, 0] -= agents
    rgb[:, :, 1] -= agents
    rgb[:, :, 2] -= agents
    infants = np.clip(grid[:, :, 3], 0, 1)
    rgb[:, :, 2] -= infants * 0.6
    rgb = np.clip(rgb, 0, 1)
    rgb = np.repeat(rgb, 2, axis=0)
    rgb = np.repeat(rgb, 2, axis=1)
    return rgb


def save_checkpoint(state, step, cfg):
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    path = os.path.join(cfg['checkpoint_dir'], f"{run_name(cfg)}_step{step}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(state, f)
    print(f"  checkpoint saved: {path}")


def main(cfg):
    name = run_name(cfg)
    wandb.init(project=cfg['project'], name=name, config=cfg)

    env = Gridworld(
        nb_agents        = cfg['nb_agents'],
        SX               = cfg['SX'],
        SY               = cfg['SY'],
        init_food        = cfg['init_food'],
        use_lstm         = cfg['use_lstm'],
        kin_recognition  = cfg['kin_recognition'],
        proximal_reprod  = cfg['proximal_reprod'],
        energy_decay     = cfg['energy_decay'],
        max_ener         = cfg['max_ener'],
        food_value       = cfg['food_value'],
        action_cost      = cfg['action_cost'],
        feeding_transfer = cfg['feeding_transfer'],
        energy_reproduce = cfg['energy_reproduce'],
        energy_reproduce_cost = cfg['energy_reproduce_cost'],
        infant_threshold = cfg['infant_threshold'],
        simple_regrowth  = cfg['simple_regrowth'],
        infant_eat_prop  = cfg['infant_eat_prop'],
        infant_eat_prob  = cfg['infant_eat_prob'],
        infant_move_prob = cfg['infant_move_prob'],
    )

    key = random.PRNGKey(cfg['seed'])
    state = env.reset(key)

    print(f"Run: {name}  |  params/agent: {env.model.num_params}")

    vid = None
    vid_path = None
    vid_step = 0
    metrics_history = []

    for step in range(1, cfg['n_steps'] + 1):
        state, rewards, energy = env.step(state)

        # ── video capture ──────────────────────────────────
        if step % cfg['video_every'] == 1:
            vid_step = step
            vid_path = os.path.join(cfg['checkpoint_dir'], f"{name}_step{step}.mp4")
            os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
            vid = VideoWriter(vid_path, fps=20.0)

        if vid is not None:
            vid.add(render_frame(state))
            if (step - vid_step + 1) >= cfg['video_length']:
                vid.close()
                wandb.log({'video': wandb.Video(vid_path, fps=20, format='mp4')}, step=step)
                vid = None

        # ── metrics ───────────────────────────────────────
        if step % cfg['log_every'] == 0:
            metrics = compute_metrics(state)
            metrics['step'] = step
            metrics_history.append(metrics)
            wandb.log(metrics, step=step)
            print(
                f"step {step:>7} | "
                f"pop={metrics['population']:>4} | "
                f"energy={metrics['mean_energy']:>6.1f} | "
                f"selectivity={metrics['mean_selectivity']:>7.4f} | "
                f"infant_surv={metrics['infant_survival_rate']:.3f}"
            )

        # ── checkpoint ────────────────────────────────────
        if step % cfg['checkpoint_every'] == 0:
            save_checkpoint(state, step, cfg)

    if vid is not None:
        vid.close()
        wandb.log({'video': wandb.Video(vid_path, fps=20, format='mp4')}, step=cfg['n_steps'])

    save_checkpoint(state, cfg['n_steps'], cfg)

    # save metrics history for offline analysis
    metrics_path = os.path.join(cfg['checkpoint_dir'], f"{name}_metrics.pkl")
    with open(metrics_path, 'wb') as f:
        pickle.dump({'config': cfg, 'metrics': metrics_history}, f)

    wandb.finish()


if __name__ == '__main__':
    main(config)
