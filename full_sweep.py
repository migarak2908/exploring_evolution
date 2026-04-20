import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import wandb

from EcoEvoJax.source.gridworld import Gridworld
from EcoEvoJax.source.utils import VideoWriter

# ── Base config ─────────────────────────────────────────
base_config = dict(
    # environment
    nb_agents            = 2500,
    SX                   = 100,
    SY                   = 100,
    init_food            = 1000,
    # energy / reproduction (paper defaults)
    energy_decay         = 0.1,
    max_ener             = 200.,
    food_value           = 20.,
    action_cost          = 1.0,
    feeding_transfer     = 20.,
    energy_reproduce     = 85.,
    energy_reproduce_cost= 30.,
    infant_threshold     = 100,
    # training
    n_steps              = 500_000,
    log_every            = 500,
    video_every          = 50_000,   # capture a clip every N steps
    video_length         = 200,      # frames per clip
    checkpoint_every     = 50_000,
    checkpoint_dir       = 'checkpoints',
    seed                 = 0,
    # wandb
    project              = 'kin-selection',
)

# ── Ablation conditions to run ───────────────────────────
# Add/remove rows to toggle which conditions run.
# Start with just LSTM vs no-LSTM; uncomment others when ready.
ablations = [
    dict(use_lstm=True,  kin_recognition=True, proximal_reprod=True),
    dict(use_lstm=False, kin_recognition=True, proximal_reprod=True),
    # dict(use_lstm=True,  kin_recognition=False, proximal_reprod=True),
    # dict(use_lstm=False, kin_recognition=False, proximal_reprod=True),
    # dict(use_lstm=True,  kin_recognition=True,  proximal_reprod=False),
    # dict(use_lstm=False, kin_recognition=True,  proximal_reprod=False),
    # dict(use_lstm=True,  kin_recognition=False, proximal_reprod=False),
    # dict(use_lstm=False, kin_recognition=False, proximal_reprod=False),
]
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

    mean_energy      = (state.agents.energy * alive).sum() / (n_alive + eps)
    selectivity      = (
        state.agents.n_fed_offspring / (state.agents.n_fed_total + eps)
        - state.agents.n_faced_offspring / (state.agents.n_faced_agent + eps)
    )
    mean_selectivity  = (selectivity * alive).sum() / (n_alive + eps)
    mean_feeding      = (state.agents.n_fed_total.astype(jnp.float32) * alive).sum() / (n_alive + eps)
    infant_survival   = state.agents.survived_infancy.sum() / state.agents.survived_infancy.shape[0]
    mean_nb_offspring = (state.agents.nb_offspring.astype(jnp.float32) * alive).sum() / (n_alive + eps)

    return {
        'population':           int(n_alive),
        'mean_energy':          float(mean_energy),
        'mean_selectivity':     float(mean_selectivity),
        'mean_feeding_events':  float(mean_feeding),
        'infant_survival_rate': float(infant_survival),
        'mean_nb_offspring':    float(mean_nb_offspring),
    }


def render_frame(state):
    """Build an RGB frame from the grid state.
    Colour scheme:
      - white background
      - green: food (channel 1)
      - black: agents (channel 0)
      - orange tint: infants (channel 3)
    """
    grid = np.array(state.state)

    rgb = np.ones((grid.shape[0], grid.shape[1], 3), dtype=np.float32)

    # food → green
    food = np.clip(grid[:, :, 1], 0, 1)
    rgb[:, :, 0] -= food
    rgb[:, :, 2] -= food

    # agents → black
    agents = np.clip(grid[:, :, 0], 0, 1)
    rgb[:, :, 0] -= agents
    rgb[:, :, 1] -= agents
    rgb[:, :, 2] -= agents

    # infants → orange tint (red channel boost, reduce blue)
    infants = np.clip(grid[:, :, 3], 0, 1)
    rgb[:, :, 2] -= infants * 0.6

    rgb = np.clip(rgb, 0, 1)

    # upscale 2x for visibility
    rgb = np.repeat(rgb, 2, axis=0)
    rgb = np.repeat(rgb, 2, axis=1)

    return rgb


def save_checkpoint(state, step, name, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{name}_step{step}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(state, f)
    print(f"  checkpoint saved: {path}")


def run_condition(cfg, ablation):
    full_cfg = {**cfg, **ablation}
    name = run_name(full_cfg)

    wandb.init(project=full_cfg['project'], name=name, config=full_cfg, reinit=True)

    env = Gridworld(
        nb_agents             = full_cfg['nb_agents'],
        SX                    = full_cfg['SX'],
        SY                    = full_cfg['SY'],
        init_food             = full_cfg['init_food'],
        use_lstm              = full_cfg['use_lstm'],
        kin_recognition       = full_cfg['kin_recognition'],
        proximal_reprod       = full_cfg['proximal_reprod'],
        energy_decay          = full_cfg['energy_decay'],
        max_ener              = full_cfg['max_ener'],
        food_value            = full_cfg['food_value'],
        action_cost           = full_cfg['action_cost'],
        feeding_transfer      = full_cfg['feeding_transfer'],
        energy_reproduce      = full_cfg['energy_reproduce'],
        energy_reproduce_cost = full_cfg['energy_reproduce_cost'],
        infant_threshold      = full_cfg['infant_threshold'],
    )

    key = random.PRNGKey(full_cfg['seed'])
    state = env.reset(key)

    print(f"\n{'='*60}")
    print(f"Run: {name}  |  params/agent: {env.model.num_params}")
    print(f"{'='*60}")

    vid = None
    vid_path = None
    vid_step = 0

    for step in range(1, full_cfg['n_steps'] + 1):
        state, rewards, energy = env.step(state)

        # ── video capture ──────────────────────────────────
        if step % full_cfg['video_every'] == 1:
            vid_step = step
            vid_path = os.path.join(
                full_cfg['checkpoint_dir'], f"{name}_step{step}.mp4"
            )
            os.makedirs(full_cfg['checkpoint_dir'], exist_ok=True)
            vid = VideoWriter(vid_path, fps=20.0)

        if vid is not None:
            vid.add(render_frame(state))
            if (step - vid_step + 1) >= full_cfg['video_length']:
                vid.close()
                wandb.log({'video': wandb.Video(vid_path, fps=20, format='mp4')}, step=step)
                vid = None

        # ── metrics ───────────────────────────────────────
        if step % full_cfg['log_every'] == 0:
            metrics = compute_metrics(state)
            wandb.log(metrics, step=step)
            print(
                f"step {step:>7} | "
                f"pop={metrics['population']:>4} | "
                f"energy={metrics['mean_energy']:>6.1f} | "
                f"selectivity={metrics['mean_selectivity']:>7.4f} | "
                f"infant_surv={metrics['infant_survival_rate']:.3f}"
            )

        # ── checkpoint ────────────────────────────────────
        if step % full_cfg['checkpoint_every'] == 0:
            save_checkpoint(state, step, name, full_cfg['checkpoint_dir'])

    # close any open video writer at end of run
    if vid is not None:
        vid.close()
        wandb.log({'video': wandb.Video(vid_path, fps=20, format='mp4')}, step=full_cfg['n_steps'])

    save_checkpoint(state, full_cfg['n_steps'], name, full_cfg['checkpoint_dir'])
    wandb.finish()


def main():
    for ablation in ablations:
        run_condition(base_config, ablation)


if __name__ == '__main__':
    main()
