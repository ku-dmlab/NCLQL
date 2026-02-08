from typing import NamedTuple, Tuple

import jax, jax.numpy as jnp
import optax
import haiku as hk

from relax.algorithm.base import Algorithm
from relax.network.nclql import NCLQLAgent, NCLQLParams
from relax.utils.experience import Experience
from relax.utils.typing_utils import Metric


class NCLQLOptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState

class NCLQLTrainState(NamedTuple):
    params: NCLQLParams
    opt_state: NCLQLOptStates
    step: int

class NCLQL(Algorithm):
    def __init__(self, agent: NCLQLAgent, params: NCLQLParams, *, gamma: float = 0.99, lr: float = 1e-4,
                 tau: float = 0.005, reward_scale: float = 0.2):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.optim = optax.adam(lr)
        self.reward_scale = reward_scale
        self.state = NCLQLTrainState(
            params=params,
            opt_state=NCLQLOptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2)
            ),
            step=jnp.int32(0),
        )

        @jax.jit
        def stateless_update(
            key: jax.Array, state: NCLQLTrainState, data: Experience
        ) -> Tuple[NCLQLTrainState, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            q1_params, q2_params, target_q1_params, target_q2_params = state.params
            q1_opt_state, q2_opt_state = state.opt_state
            step = state.step
            batch_size = obs.shape[0]
            key, key1, key2, key3 = jax.random.split(key, 4)

            reward *= self.reward_scale

            # compute target q
            next_action = self.agent.get_action(key1, (q1_params, q2_params), next_obs)
            L_minus_1 = jnp.full((batch_size,), self.agent.L - 1)
            q1_target = self.agent.q(target_q1_params, next_obs, next_action, L_minus_1)
            q2_target = self.agent.q(target_q2_params, next_obs, next_action, L_minus_1)
            q_target = jnp.minimum(q1_target, q2_target)
            q_backup = reward + (1 - done) * self.gamma * q_target

            # compute q loss
            def q_td_loss_fn(q_params: hk.Params) -> jax.Array:
                q = self.agent.q(q_params, obs, action, L_minus_1)
                q_loss = jnp.mean((q - q_backup) ** 2)
                return q_loss

            # compute qt loss
            l = jax.random.randint(key2, (batch_size,), minval=0, maxval=self.agent.L-1) # to do not include L-1
            noise = jax.random.normal(key3, (batch_size, self.agent.act_dim))
            sigmas = self.agent.ald.sigma_schedule()
            sigmas_l = sigmas[l][:, None]   
            a_l = action + sigmas_l * noise
            # a_l = jnp.clip(a_l, -1.0, 1.0)

            q1 = self.agent.q(q1_params, obs, action, L_minus_1)
            q2 = self.agent.q(q2_params, obs, action, L_minus_1)
            q_cat = jnp.stack([q1, q2], axis=0)
            q_mean = jnp.mean(q_cat, axis=0)

            def q_t_loss_fn(q_params: hk.Params) -> jax.Array:
                q_t = self.agent.q(q_params, obs, a_l, l)
                q_loss = jnp.mean(((q_t - q_mean)) ** 2) 
                return q_loss

            def param_update(optim, params, grads, opt_state):
                update, new_opt_state = optim.update(grads, opt_state)
                new_params = optax.apply_updates(params, update)
                return new_params, new_opt_state

            q1_loss, q1_grads = jax.value_and_grad(q_td_loss_fn)(q1_params)
            q2_loss, q2_grads = jax.value_and_grad(q_td_loss_fn)(q2_params)
            q1_params, q1_opt_state = param_update(self.optim, q1_params, q1_grads, q1_opt_state)
            q2_params, q2_opt_state = param_update(self.optim, q2_params, q2_grads, q2_opt_state)

            qt1_loss, qt1_grads = jax.value_and_grad(q_t_loss_fn)(q1_params)
            qt2_loss, qt2_grads = jax.value_and_grad(q_t_loss_fn)(q2_params)
            q1_params, q1_opt_state = param_update(self.optim, q1_params, qt1_grads, q1_opt_state)
            q2_params, q2_opt_state = param_update(self.optim, q2_params, qt2_grads, q2_opt_state)

            target_q1_params = optax.incremental_update(q1_params, target_q1_params, self.tau)
            target_q2_params = optax.incremental_update(q2_params, target_q2_params, self.tau)

            state = NCLQLTrainState(
                params=NCLQLParams(q1_params, q2_params, target_q1_params, target_q2_params),
                opt_state=NCLQLOptStates(q1_opt_state, q2_opt_state),
                step=step + 1,
            )
            info = {
                "td_q1_loss": q1_loss,
                "td_q2_loss": q2_loss,
                "q1": jnp.mean(q1),
                "q2": jnp.mean(q2),
                "qt1_loss": qt1_loss,
                "qt2_loss": qt2_loss,
            }
            return state, info

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action)
    
    def get_policy_params(self):
        return self.state.params.q1, self.state.params.q2
