import argparse
import os.path
from pathlib import Path
import time
from functools import partial
import yaml

import jax, jax.numpy as jnp


from relax.algorithm.nclql import NCLQL
from relax.network.nclql import create_NCLQL_net
from relax.buffer import TreeBuffer
from relax.trainer.off_policy import OffPolicyTrainer
from relax.env import create_env, create_vector_env
from relax.utils.experience import Experience
from relax.utils.fs import PROJECT_ROOT
from relax.utils.random_utils import seeding
from relax.utils.log_diff import log_git_details

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, default="NCLQL")
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--suffix", type=str, default="test_NCLQL")
    parser.add_argument("--buffer_size", type=int, default=int(1e6))
    parser.add_argument("--num_vec_envs", type=int, default=5)
    parser.add_argument("--hidden_num", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--start_step", type=int, default=int(3e4))
    parser.add_argument("--total_step", type=int, default=int(5e6))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--num_particles", type=int, default=1)
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--w", type=float, default=500)
    parser.add_argument("--q_grad_norm", type=bool, default=True)
    parser.add_argument("--T", type=int, default=2)
    parser.add_argument("--L", type=int, default=10)
    parser.add_argument("--sigma_max", type=float, default=0.1)
    parser.add_argument("--sigma_min", type=float, default=0.001)
    parser.add_argument("--step_lr", type=float, default=0.0001)
    parser.add_argument("--project_name", type=str, default="project")
    args = parser.parse_args()

    if args.debug:
        from jax import config
        config.update("jax_disable_jit", True)

    master_seed = args.seed
    master_rng, _ = seeding(master_seed)
    env_seed, env_action_seed, eval_env_seed, buffer_seed, init_network_seed, train_seed = map(
        int, master_rng.integers(0, 2**32 - 1, 6)
    )
    init_network_key = jax.random.key(init_network_seed)
    train_key = jax.random.key(train_seed)
    del init_network_seed, train_seed

    if args.num_vec_envs > 0:
        env, obs_dim, act_dim = create_vector_env(args.env, args.num_vec_envs, env_seed, env_action_seed, mode="futex")
    else:
        env, obs_dim, act_dim = create_env(args.env, env_seed, env_action_seed)
    eval_env = None

    hidden_sizes = [args.hidden_dim] * args.hidden_num

    buffer = TreeBuffer.from_experience(obs_dim, act_dim, size=args.buffer_size, seed=buffer_seed)

    gelu = partial(jax.nn.gelu, approximate=False)

    def mish(x: jax.Array):
        return x * jnp.tanh(jax.nn.softplus(x))
    
    print(f"Algorithm: {args.alg}")

    if args.alg == "NCLQL":
        agent, params = create_NCLQL_net(init_network_key, obs_dim, act_dim, hidden_sizes, mish,
                                        T=args.T,
                                        L=args.L,
                                        sigma_max=args.sigma_max,
                                        sigma_min=args.sigma_min,
                                        step_lr=args.step_lr,
                                        w=args.w,
                                        q_grad_norm=args.q_grad_norm,
                                        num_particles=args.num_particles)
        algorithm = NCLQL(agent, params, lr=args.lr)
    else:
        raise ValueError(f"Invalid algorithm {args.alg}!")
    
    exp_dir = PROJECT_ROOT / "logs" / args.env / (args.alg + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + f'_s{args.seed}_{args.suffix}')
    trainer = OffPolicyTrainer(
        env=env,
        algorithm=algorithm,
        buffer=buffer,
        start_step=args.start_step,
        total_step=args.total_step,
        sample_per_iteration=1,
        evaluate_env=eval_env,
        save_policy_every=int(args.total_step / 40),
        warmup_with="random",
        log_path=exp_dir,
        config=vars(args),
        project_name=args.project_name,
    )

    trainer.setup(Experience.create_example(obs_dim, act_dim, trainer.batch_size))
    log_git_details(log_file=os.path.join(exp_dir, f'{args.alg}.diff'))
    
    # Save the arguments to a YAML file
    args_dict = vars(args)
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file)
    trainer.run(train_key)
