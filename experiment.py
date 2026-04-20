import os
import pickle
import jax
import jax.numpy as jnp
from jax import random
import wandb

from EcoEvoJax.source.gridworld import Gridworld

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
    # training
    n_steps          = 500_000,
    log_every        = 500,
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
    mean_selectivity = (selectivity * alive).sum() / (n_alive + eps)

    mean_feeding = (state.agents.n_fed_total.astype(jnp.float32) * alive).sum() / (n_alive + eps)
    infant_survival = state.agents.survived_infancy.sum() / state.agents.survived_infancy.shape[0]
    mean_nb_offspring = (state.agents.nb_offspring.astype(jnp.float32) * alive).sum() / (n_alive + eps)

    return {
        'population':          int(n_alive),
        'mean_energy':         float(mean_energy),
        'mean_selectivity':    float(mean_selectivity),
        'mean_feeding_events': float(mean_feeding),
        'infant_survival_rate':float(infant_survival),
        'mean_nb_offspring':   float(mean_nb_offspring),
    }


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
    )

    key = random.PRNGKey(cfg['seed'])
    state = env.reset(key)

    print(f"Run: {name}  |  params/agent: {env.model.num_params}")

    for step in range(1, cfg['n_steps'] + 1):
        state, rewards, energy = env.step(state)

        if step % cfg['log_every'] == 0:
            metrics = compute_metrics(state)
            wandb.log(metrics, step=step)
            print(
                f"step {step:>7} | "
                f"pop={metrics['population']:>4} | "
                f"energy={metrics['mean_energy']:>6.1f} | "
                f"selectivity={metrics['mean_selectivity']:>7.4f} | "
                f"infant_surv={metrics['infant_survival_rate']:.3f}"
            )

        if step % cfg['checkpoint_every'] == 0:
            save_checkpoint(state, step, cfg)

    save_checkpoint(state, cfg['n_steps'], cfg)
    wandb.finish()


if __name__ == '__main__':
    main(config)
