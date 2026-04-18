# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple
from PIL import Image
from PIL import ImageDraw
import numpy as np

import jax
import jax.numpy as jnp
from jax import random
from flax.struct import dataclass
import math



AGENT_VIEW = 5
GRID_CHANNELS = 6
OBS_CHANNELS = 5
NUM_ACTIONS = 7

DX = jnp.array([0, 1, 0, -1])
DY = jnp.array([1, 0, -1, 0])

from source.agent import MetaRnnPolicy_bcppr
from source.agent import metaRNNPolicyState_bcppr
from evojax.policy.base import PolicyState
from evojax.task.base import TaskState
from evojax.task.base import VectorizedTask

_offsets = sorted([(dx, dy) for dx in range(-5, 6) for dy in range(-5,6) if not (dx == 0 and dy == 0)],
                 key= lambda o: o[0]**2 + o[1]**2)

SEARCH_OFFSETS = jnp.array(_offsets)

@dataclass
class AgentStates(object):
    posx: jnp.uint16
    posy: jnp.uint16
    orientation: jnp.uint16
    params: jnp.ndarray
    policy_states: PolicyState
    energy: jnp.ndarray
    time_good_level: jnp.uint16
    time_alive: jnp.uint16
    time_under_level: jnp.uint16
    alive: jnp.int8
    nb_food: jnp.ndarray
    nb_offspring: jnp.uint16
    uid: jnp.uint32
    parent_id: jnp.uint32


@dataclass
class State(TaskState):
    obs: jnp.int8
    last_actions: jnp.int8
    rewards: jnp.int8
    state: jnp.int32
    agents: AgentStates
    steps: jnp.int32
    key: jnp.ndarray
    next_uid: jnp.uint32


def get_ob(state: jnp.ndarray, pos_x: jnp.int32, pos_y: jnp.int32) -> jnp.ndarray:
    obs = (jax.lax.dynamic_slice(jnp.pad(state, ((AGENT_VIEW, AGENT_VIEW), (AGENT_VIEW, AGENT_VIEW), (0, 0))),
                                 (pos_x - AGENT_VIEW + AGENT_VIEW, pos_y - AGENT_VIEW + AGENT_VIEW, 0),
                                 (2 * AGENT_VIEW + 1, 2 * AGENT_VIEW + 1, OBS_CHANNELS)))

    return obs


def get_init_state_fn(key: jnp.ndarray, SX, SY, posx, posy, pos_food_x, pos_food_y, niches_scale=200) -> jnp.ndarray:
    grid = jnp.zeros((SX, SY, GRID_CHANNELS))

    # initialise agents
    grid = grid.at[posx, posy, 0].add(1)

    # initialise food
    grid = grid.at[pos_food_x, pos_food_y, 1].set(1)

    # initialise walls
    grid = grid.at[0, :, 2].set(1)
    grid = grid.at[-1, :, 2].set(1)
    grid = grid.at[:, 0, 2].set(1)
    grid = grid.at[:, -1, 2].set(1)

    # initialise infants
    grid = grid.at[posx, posy, 3].set(1)

    # initialise parent_ids

    grid = grid.at[:, :, 4].set(0)

    # initialise_gradient

    new_array = jnp.clip(
        np.asarray([(math.pow(niches_scale, el) - 1) / (niches_scale - 1) for el in np.arange(0, SX) / SX]), 0,
        1)

    for col in range(SY - 1):
        new_col = jnp.clip(
            np.asarray([(math.pow(niches_scale, el) - 1) / (niches_scale - 1) for el in np.arange(0, SX) / SX]), 0, 1)

        new_array = jnp.append(new_array, new_col)
    new_array = jnp.transpose(jnp.reshape(new_array, (SY, SX)))
    grid = grid.at[:, :, 5].set(new_array)

    return (grid)


get_obs_vector = jax.vmap(get_ob, in_axes=(None, 0, 0), out_axes=0)


class Gridworld(VectorizedTask):
    """gridworld task."""

    def __init__(self,
                 nb_agents: int = 100,
                 obs_channels: int = OBS_CHANNELS,
                 action_space: int = NUM_ACTIONS,
                 init_food=16000,
                 SX=300,
                 SY=100,
                 reproduction_on=True,
                 proximal_reprod=True,
                 place_resources=False,
                 place_agent=False,
                 use_lstm=True,
                 params=None,
                 test: bool = False,
                 energy_decay=0.05,
                 max_age: int = 1000,
                 time_reproduce: int = 150,
                 time_death: int = 40,
                 max_ener=3.,
                 regrowth_scale=0.002,
                 niches_scale=200,
                 spontaneous_regrow=1 / 200000,
                 wall_kill=True,
                 infant_move_prob=1.0,
                 infant_eat_prop=1.0,
                 infant_eat_prob=1.0,
                 infant_threshold=100,
                 feeding_transfer=0.2
                 ):
        # self.obs_shape = (AGENT_VIEW, AGENT_VIEW, obs_channels)
        # self.obs_shape=11*5*4
        self.act_shape = tuple([action_space, ])
        self.test = test
        self.nb_agents = nb_agents
        self.SX = SX
        self.SY = SY
        self.energy_decay = energy_decay
        self.model = MetaRnnPolicy_bcppr(input_dim=((AGENT_VIEW * 2 + 1), (AGENT_VIEW * 2 + 1), obs_channels), hidden_dim=4,
                                         output_dim=action_space, encoder_layers=[], hidden_layers=[8], use_lstm=use_lstm)

        self.energy_decay = energy_decay
        self.max_age = max_age
        self.time_reproduce = time_reproduce
        self.time_death = time_death
        self.max_ener = max_ener

        self.regrowth_scale = regrowth_scale
        self.niches_scale = niches_scale
        self.spontaneous_regrow = spontaneous_regrow
        self.place_agent = place_agent
        self.place_resources = place_resources
        self.use_lstm = use_lstm
        self.params = params
        self.reproduction_on = reproduction_on
        self.proximal_reprod = proximal_reprod

        self.infant_move_prob = infant_move_prob
        self.infant_eat_prop = infant_eat_prop
        self.infant_eat_prob = infant_eat_prob
        self.infant_threshold = infant_threshold
        self.feeding_transfer = feeding_transfer

        def reset_fn(key):

            if self.place_agent:
                next_key, key = random.split(key)
                posx = random.randint(next_key, (nb_agents,), int(2 / 5 * SX), int(3 / 5 * SX))
                next_key, key = random.split(key)
                posy = random.randint(next_key, (nb_agents,), int(2 / 5 * SX), int(3 / 5 * SX))
                next_key, key = random.split(key)

            else:
                next_key, key = random.split(key)
                posx = random.randint(next_key, (self.nb_agents,), 1, (SX - 1))
                next_key, key = random.split(key)
                posy = random.randint(next_key, (self.nb_agents,), 1, (SY - 1))
                next_key, key = random.split(key)

            if self.place_resources:
                # lab environments have a custom location of resources
                N = 5  # minimum distance from agents
                N_wall = 5  # minimum distance from wall

                pos_food_x = jnp.concatenate(
                    (random.randint(next_key, (int(init_food / 4),), int(1 / 2 * SX) + N, (SX - 1 - N_wall)),
                     random.randint(next_key, (int(init_food / 4),), N_wall, int(1 / 2 * SX) - N),
                     random.randint(next_key, (int(init_food / 4),), 1 + N_wall, (SX - 1 - N_wall)),
                     random.randint(next_key, (int(init_food / 4),), 1 + N_wall, (SX - 1 - N_wall))))

                next_key, key = random.split(key)
                pos_food_y = jnp.concatenate(
                    (random.randint(next_key, (int(init_food / 4),), 1 + N_wall, SY - 1 - N_wall),
                     random.randint(next_key, (int(init_food / 4),), 1 + N_wall, SY - 1 - N_wall),
                     random.randint(next_key, (int(init_food / 4),), int(1 / 2 * SY) + N,
                                    (SY - 1 - N_wall)),
                     random.randint(next_key, (int(init_food / 4),), N_wall, int(1 / 2 * SY) - N)))
                next_key, key = random.split(key)

            else:
                pos_food_x = random.randint(next_key, (init_food,), 1, (SX - 1))
                next_key, key = random.split(key)
                pos_food_y = random.randint(next_key, (init_food,), 1, (SY - 1))
                next_key, key = random.split(key)

            grid = get_init_state_fn(key, SX, SY, posx, posy, pos_food_x, pos_food_y, niches_scale)

            next_key, key = random.split(key)

            if self.params is None:
                params = jax.random.normal(
                    next_key,
                    (self.nb_agents, self.model.num_params,),
                ) / 100
            else:
                params =self.params

            policy_states = self.model.reset_b(jnp.zeros(self.nb_agents, ))

            next_key, key = random.split(key)

            orientation = random.randint(next_key, (nb_agents,), 0, 4)
            uid = jnp.arange(1, nb_agents+1)
            parent_id = jnp.full((nb_agents,), 0)
            next_uid = nb_agents+1

            agents = AgentStates(posx=posx, posy=posy, orientation=orientation,
                                 energy=self.max_ener * jnp.ones((self.nb_agents,)),
                                 time_good_level=jnp.zeros((self.nb_agents,), dtype=jnp.uint16), params=params,
                                 policy_states=policy_states,
                                 time_alive=jnp.zeros((self.nb_agents,), dtype=jnp.uint16),
                                 time_under_level=jnp.zeros((self.nb_agents,), dtype=jnp.uint16),
                                 alive=jnp.ones((self.nb_agents,), dtype=jnp.uint16).at[0:2 * self.nb_agents // 3].set(
                                     0),
                                 nb_food=jnp.zeros((self.nb_agents,)),
                                 nb_offspring=jnp.zeros((self.nb_agents,), dtype=jnp.uint16),
                                 uid=uid,
                                 parent_id=parent_id
                                 )

            raw_obs = get_obs_vector(grid, posx, posy)
            is_offspring = (raw_obs[:, :, :, 4] == uid[:, None, None]).astype(jnp.int32)
            obs = raw_obs.at[:, :, :, 4].set(is_offspring)

            return State(state=grid, obs=obs, last_actions=jnp.zeros((self.nb_agents, NUM_ACTIONS)),
                         rewards=jnp.zeros((self.nb_agents, 1)), agents=agents,
                         steps=jnp.zeros((), dtype=int), key=next_key, next_uid=next_uid)

        self._reset_fn = jax.jit(reset_fn)


        def reproduce(params, posx, posy, energy, time_good_level, key, policy_states, time_alive, alive, nb_food,
                      nb_offspring, action_int, uid, parent_id, next_uid, grid):

            reproducer_mask = (time_good_level > self.time_reproduce) * action_int[:, 6] * (alive > 0)
            dead_mask = (1 - alive)

            reprod_rank = jnp.cumsum(reproducer_mask) * reproducer_mask
            dead_rank = jnp.cumsum(dead_mask) * dead_mask

            match = (dead_rank[:, None] == reprod_rank[None, :]) & \
                    (dead_rank[:, None] > 0) & \
                    (dead_rank[:, None] <= jnp.minimum(dead_rank.max(), reprod_rank.max()))


            parent_idx = jnp.argmax(match, axis=1)
            is_filled = match.any(axis=1)

            if self.proximal_reprod:
                candidate_x = jnp.clip(posx[:, None] + SEARCH_OFFSETS[None, :, 0], 1, SX - 2)
                candidate_y = jnp.clip(posy[:, None] + SEARCH_OFFSETS[None, :, 1], 1, SY - 2)
                occupied = grid[candidate_x, candidate_y, 0] > 0
                first_empty = jnp.argmin(occupied, axis=1)
                spawn_x = candidate_x[jnp.arange(nb_agents), first_empty]
                spawn_y = candidate_y[jnp.arange(nb_agents), first_empty]

            else:
                next_key, key = jax.random.split(key)
                spawn_x = random.randint(next_key, (nb_agents,), 1, SX - 2)
                next_key, key = jax.random.split(key)
                spawn_y = random.randint(next_key, (nb_agents,), 1, SY - 2)

            offspring_x = spawn_x[parent_idx]
            offspring_y = spawn_y[parent_idx]
            is_filled = is_filled & (grid[offspring_x, offspring_y, 0] == 0)

            same_cell = (offspring_x[:, None] == offspring_x[None, :]) & \
                        (offspring_y[:, None] == offspring_y[None, :]) & \
                        is_filled[:, None] & is_filled[None, :]

            is_loser = (same_cell & (jnp.arange(nb_agents)[:, None] > jnp.arange(nb_agents)[None, :])).any(axis=1)

            is_filled = is_filled & ~is_loser

            n_births = is_filled.sum()

            parent_uid = uid[parent_idx]
            offspring_uid = next_uid + jnp.where(is_filled, jnp.cumsum(is_filled) - 1, 0)
            uid = jnp.where(is_filled, offspring_uid, uid)
            parent_id = jnp.where(is_filled, parent_uid, parent_id)
            next_uid = next_uid + n_births.astype(jnp.uint32)

            next_key, key = random.split(key)
            if self.reproduction_on:
                mutated = params[parent_idx] + 0.02 * jax.random.normal(next_key, params.shape)
                params = jnp.where(is_filled[:, None], mutated, params)

            posx = jnp.where(is_filled, offspring_x, posx)
            posy = jnp.where(is_filled, offspring_y, posy)
            energy = jnp.where(is_filled, self.max_ener, energy)
            time_good_level = jnp.where(is_filled, 0, time_good_level)
            alive = jnp.where(is_filled, 1, alive)
            time_alive = jnp.where(is_filled, 0, time_alive)
            nb_food = jnp.where(is_filled, 0, nb_food)
            nb_offspring = jnp.where(is_filled, 0, nb_offspring)

            policy_states = metaRNNPolicyState_bcppr(
                lstm_h=jnp.where(is_filled[:, None], jnp.zeros_like(policy_states.lstm_h), policy_states.lstm_h),
                lstm_c=jnp.where(is_filled[:, None], jnp.zeros_like(policy_states.lstm_c), policy_states.lstm_c),
                keys=policy_states.keys
            )

            actually_reproduced = reproducer_mask & (reprod_rank <= n_births)
            nb_offspring = nb_offspring + actually_reproduced.astype(jnp.int32)

            time_good_level = jnp.where(actually_reproduced, 0, time_good_level)

            return (params, posx, posy, energy, time_good_level, policy_states, time_alive,
                    alive, nb_food, nb_offspring, uid, parent_id, next_uid)


        def step_fn(state):
            key = state.key
            next_key, key = random.split(key)

            # model selection of action
            actions_logit, policy_states = self.model.get_actions(state, state.agents.params,
                                                                  state.agents.policy_states)
            actions = jax.nn.one_hot(jax.random.categorical(next_key, actions_logit * 50, axis=-1), NUM_ACTIONS)
            # actions=jax.nn.one_hot(jnp.argmax(actions_logit,axis=-1),5)

            grid = state.state
            energy = state.agents.energy
            alive = state.agents.alive

            # obtain actions
            action_int = actions.astype(jnp.int32)

            # identify infants
            is_infant = state.agents.time_alive < self.infant_threshold

            # feeding mechanism

            curr_pos_x = state.agents.posx
            curr_pos_y = state.agents.posy

            feed_pos_x = curr_pos_x + DX[state.agents.orientation]
            feed_pos_y = curr_pos_y + DY[state.agents.orientation]

            x_match = (feed_pos_x[:, None] == curr_pos_x[None, :])
            y_match = (feed_pos_y[:, None] == curr_pos_y[None, :])

            success_feed = x_match & y_match

            can_feed = (alive[:, None] > 0) * (action_int[:, 5][:, None])
            can_receive = (alive[None, :] > 0)

            valid_feed = success_feed & can_feed & can_receive

            energy = jnp.where(valid_feed.any(axis=1), energy - self.feeding_transfer, energy)
            energy = jnp.where(valid_feed.any(axis=0), energy + self.feeding_transfer, energy)


            # move agent

            next_key, key = jax.random.split(key)
            infant_move = jnp.where(is_infant, jax.random.bernoulli(next_key, self.infant_move_prob, (nb_agents,)) ,1)

            posx = state.agents.posx + (DX[state.agents.orientation] * action_int[:, 1] * infant_move)
            posy = state.agents.posy + (DY[state.agents.orientation] * action_int[:, 1] * infant_move)

            orientation = (state.agents.orientation + action_int[:, 2] - action_int[:, 3]) % 4

            # wall

            hit_wall = state.state[posx, posy, 2] > 0
            if (wall_kill):
                alive = jnp.where(hit_wall, 0, alive)
            posx = jnp.where(hit_wall, state.agents.posx, posx)
            posy = jnp.where(hit_wall, state.agents.posy, posy)

            posx = jnp.clip(posx, 0, SX - 1)
            posy = jnp.clip(posy, 0, SY - 1)
            grid = grid.at[state.agents.posx, state.agents.posy, 0].set(0)

            #precedence and takes the position.
            next_key, key = jax.random.split(key)
            priority = jax.random.uniform(next_key, (nb_agents, ), 0.0, 1.0)

            candidate_posx = (posx[:, None] == posx[None, :])
            candidate_posy = (posy[:, None] == posy[None, :])
            overlap = candidate_posx & candidate_posy & (alive[:, None] > 0) & (alive[None, :] > 0)

            not_self = jnp.arange(nb_agents)[:, None] != jnp.arange(nb_agents)[None, :]
            is_loser = overlap & not_self & (priority[:, None] < priority[None, :])
            is_loser = is_loser.any(axis=1)

            posx = jnp.where(is_loser, state.agents.posx, posx)
            posy = jnp.where(is_loser, state.agents.posy, posy)

            # add only the alive

            grid = grid.at[posx, posy, 0].add(1 * (alive > 0))

            # collect food

            rewards = (alive > 0) * (grid[posx, posy, 1] > 0) * (1 / (grid[posx, posy, 0] + 1e-10))
            rewards = rewards * action_int[:, 4]

            next_key, key = jax.random.split(key)
            ie_prop = self.infant_eat_prop
            ie_prob = self.infant_eat_prob

            eat_success = jax.random.bernoulli(next_key, ie_prob, shape=(self.nb_agents,))
            rewards = jnp.where(is_infant,  rewards * eat_success * ie_prop, rewards)

            grid = grid.at[posx, posy, 1].add(jnp.where((rewards > 0.0), -1, 0))

            grid = grid.at[:, :, 1].set(jnp.clip(grid[:, :, 1], 0, 1))

            nb_food = state.agents.nb_food + rewards

            # regrow

            num_neighbs = jax.scipy.signal.convolve2d(grid[:, :, 1], jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
                                                      mode="same")
            scale = grid[:, :, 5]
            scale_constant = regrowth_scale
            next_key, key = random.split(key)

            if scale_constant:

                num_neighbs = jnp.where(num_neighbs == 0, 0, num_neighbs)
                num_neighbs = jnp.where(num_neighbs == 1, 0.01 / 5, num_neighbs)
                num_neighbs = jnp.where(num_neighbs == 2, 0.01 / scale_constant, num_neighbs)
                num_neighbs = jnp.where(num_neighbs == 3, 0.05 / scale_constant, num_neighbs)
                num_neighbs = jnp.where(num_neighbs > 3, 0, num_neighbs)
                num_neighbs = jnp.multiply(num_neighbs, scale)
                num_neighbs = jnp.where(num_neighbs > 0, num_neighbs, 0)
                # num_neighbs = num_neighbs + self.spontaneous_regrow * scale
                num_neighbs = num_neighbs + self.spontaneous_regrow
                # num_neighbs=num_neighbs.at[350:356,98:102].set(1/40)

                num_neighbs = jnp.clip(num_neighbs - grid[:, :, 2], 0, 1)

                grid = grid.at[:, :, 1].add(random.bernoulli(next_key, num_neighbs))

            ####
            steps = state.steps + 1

            # decay of energy and clipping
            energy = energy - self.energy_decay + rewards
            energy = jnp.clip(energy, -1000, self.max_ener)

            time_good_level = jnp.where(energy > 0, (state.agents.time_good_level + 1) * alive, 0)

            time_alive = state.agents.time_alive

            # look if still alive

            time_alive = jnp.where(alive > 0, time_alive + 1, 0)

            # Update infants
            grid = grid.at[state.agents.posx, state.agents.posy, 3].set(0)
            grid = grid.at[posx, posy, 3].add(is_infant.astype(jnp.int32) * (alive > 0))
            grid = grid.at[:, :, 3].set(jnp.clip(grid[:, :, 3], 0, 1))

            # compute reproducer and go through the function only if there is one
            reproducer = jnp.where(time_good_level > self.time_reproduce, 1, 0) * action_int[:, 6]
            uid, parent_id, next_uid = state.agents.uid, state.agents.parent_id, state.next_uid
            next_key, key = random.split(key)


            params, posx, posy, energy, time_good_level, policy_states, time_alive, alive, nb_food, nb_offspring, uid, parent_id, next_uid = jax.lax.cond(
                reproducer.sum() > 0, reproduce, lambda y, z, a, b, c, d, e, f, g, h, i, j, k, l, m, n: (y, z, a, b, c, e, f, g, h, i, k, l, m),
                *(
                    state.agents.params, posx, posy, energy, time_good_level, next_key, policy_states,
                    time_alive, alive, nb_food, state.agents.nb_offspring, action_int, uid, parent_id, next_uid, grid))

            time_under_level = jnp.where(energy < 0, state.agents.time_under_level + 1, 0)
            alive = jnp.where(jnp.logical_or(time_alive > self.max_age, time_under_level > self.time_death), 0, alive)

            grid = grid.at[state.agents.posx, state.agents.posy, 4].set(0)
            grid = grid.at[posx, posy, 4].set((parent_id * (alive > 0)).astype(jnp.float32))
            done = False
            steps = jnp.where(done, jnp.zeros((), jnp.int32), steps)

            raw_obs = get_obs_vector(grid, posx, posy)
            is_offspring = (raw_obs[:, :, :, 4] == uid[:, None, None]).astype(jnp.int32)
            obs = raw_obs.at[:, :, :, 4].set(is_offspring)


            cur_state = State(state=grid, obs=obs, last_actions=actions,
                              rewards=jnp.expand_dims(rewards, -1),
                              agents=AgentStates(posx=posx, posy=posy, orientation=orientation, energy=energy, time_good_level=time_good_level,
                                                 params=params, policy_states=policy_states,
                                                 time_alive=time_alive, time_under_level=time_under_level, alive=alive,
                                                 nb_food=nb_food, nb_offspring=nb_offspring, uid=uid, parent_id=parent_id),
                              steps=steps, key=key, next_uid=next_uid)
            # keep it in case we let agent several trials
            state = jax.lax.cond(
                done, lambda x: reset_fn(state.key), lambda x: x, cur_state)

            return state, rewards, energy

        self._step_fn = jax.jit(step_fn)

    def reset(self, key: jnp.ndarray) -> State:
        return self._reset_fn(key)

    def step(self,
             state: State,
             ) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        return self._step_fn(state)


# TODO 7: Metrics
#   - Infant survival rate: track fraction of infants (time_alive < threshold) that reach adulthood
#   - Feeding amount: total energy transferred via feed action per step
#   - Feeding selectivity: implement equation 2 from Taylor-Davies et al.


