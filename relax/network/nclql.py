from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import jax, jax.numpy as jnp
import haiku as hk

from relax.network.blocks import Activation, NoiseConditionedQNet
from relax.utils.jax_utils import random_key_from_data

@dataclass(frozen=True)
class AnnealedLangevinDynamics:
    L: int
    T: int
    q_grad_norm: bool
    w: float = 1.0
    sigma_max: float = 0.1
    sigma_min: float = 0.001
    step_lr: float = 0.001

    def sigma_schedule(self) -> float:
        sigmas = jnp.exp(jnp.linspace(jnp.log(self.sigma_max), jnp.log(self.sigma_min), self.L))
        return sigmas

    def sample(self, key: jax.Array, model, shape: Tuple[int, ...]) -> jax.Array:
        sigmas = self.sigma_schedule()
        x_key, noise_key = jax.random.split(key)
        x = jax.random.uniform(x_key, shape, minval=-1.0, maxval=1.0)
        noise = jax.random.normal(noise_key, (self.L, self.T, *shape))

        l_seq = jnp.repeat(jnp.arange(self.L), self.T)
        noise_seq = noise.reshape((self.L * self.T, *shape))

        def step(x, inputs):
            l, eps = inputs
            step_size = self.step_lr * (sigmas[l] / sigmas[-1]) ** 2

            def grad_fn(x):
                q1, q2 = model(x, l)
                return q1.sum() + q2.sum()
            grad_x = jax.grad(grad_fn)(x)

            if self.q_grad_norm:
                grad_x = grad_x / (jnp.linalg.norm(grad_x, axis=-1, keepdims=True) + 1e-8)

            x = x + 0.5 * step_size * self.w * grad_x + jnp.sqrt(step_size) * eps
            x = jnp.clip(x, -1.0, 1.0)
            return x, None

        x, _ = jax.lax.scan(step, x, (l_seq, noise_seq))
        return x
    

class NCLQLParams(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    
@dataclass
class NCLQLAgent:
    q: Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]
    q_grad_norm: bool
    w: float
    act_dim: int
    num_particles: int
    T: int
    L: int
    sigma_max: float
    sigma_min: float
    step_lr: float

    @property
    def ald(self) -> AnnealedLangevinDynamics:
        return AnnealedLangevinDynamics(self.L, self.T, self.q_grad_norm, self.w, self.sigma_max, self.sigma_min, self.step_lr)

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        """for data collection"""

        q1_params, q2_params = policy_params

        original_obs_ndim = obs.ndim
        obs = obs[None, :] if obs.ndim == 1 else obs  

        def model_fn(x, t):
            q1 = self.q(q1_params, obs, x, t)
            q2 = self.q(q2_params, obs, x, t)
            return q1, q2    
        
        def q1s_for_actions(actions):
            L_minus_1 = jnp.full((obs.shape[0],), self.L - 1)
            return jax.vmap(self.q, in_axes=(None, 0, 0, 0))(q1_params, obs, actions, L_minus_1)
        def q2s_for_actions(actions):
            L_minus_1 = jnp.full((obs.shape[0],), self.L - 1)
            return jax.vmap(self.q, in_axes=(None, 0, 0, 0))(q2_params, obs, actions, L_minus_1)
        
        key, noise_key = jax.random.split(key)
        if self.num_particles == 1:
            act = self.ald.sample(noise_key, model_fn, (*obs.shape[:-1], self.act_dim))
        else:
            keys = jax.random.split(noise_key, self.num_particles)
            acts = jax.vmap(self.ald.sample, in_axes=(0, None, None))(keys, model_fn, (*obs.shape[:-1], self.act_dim))
            
            q1s = jax.vmap(q1s_for_actions)(acts) 
            q2s = jax.vmap(q2s_for_actions)(acts) 
            qs = jnp.minimum(q1s, q2s)
            q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
            act = jnp.take_along_axis(acts, q_best_ind[..., None], axis=0).squeeze(axis=0)

        if original_obs_ndim == 1:
            act = act.squeeze(0)

        return act


    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        """for evaluation and not deterministic"""
        key = random_key_from_data(obs)
        return self.get_action(key, policy_params, obs)


def create_NCLQL_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.mish,
    T: int = 2,
    L: int = 10,
    sigma_max: float = 0.1,
    sigma_min: float = 0.001,
    step_lr: float = 0.0001,
    w: float = 500,
    q_grad_norm: bool = True,
    num_particles: int = 1,
) -> Tuple[NCLQLAgent, NCLQLParams]:
    q = hk.without_apply_rng(hk.transform(lambda obs, act, i: NoiseConditionedQNet(hidden_sizes, activation)(obs, act, i)))

    @jax.jit
    def init(key, obs, act):
        q1_key, q2_key = jax.random.split(key, 2)
        q1_params = q.init(q1_key, obs, act, 0)
        q2_params = q.init(q2_key, obs, act, 0)
        target_q1_params = q1_params
        target_q2_params = q2_params
        return NCLQLParams(q1_params, q2_params, target_q1_params, target_q2_params)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = NCLQLAgent(q=q.apply, q_grad_norm=q_grad_norm, w=w, act_dim=act_dim, T=T, L=L,
                    sigma_max=sigma_max, sigma_min=sigma_min, step_lr=step_lr, num_particles=num_particles)
    return net, params