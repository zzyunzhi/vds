import os.path as osp

import argparse
from gym.wrappers import FlattenObservation
from baselines.bench import Monitor
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.wrappers import ClipActionsWrapper
from baselines import logger
from baselines.common import retro_wrappers

import tensorflow as tf
import numpy as np
import random


def init_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', type=str, default='FetchReach-v1')
    parser.add_argument('--env_type', help='environment type: maze, robotics, ant', default='robotics')
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--num_timesteps', type=float, default=1e6),
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=2, type=int)
    parser.add_argument('--num_cpu', help='Number of CPUs', default=1, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_interval', help='save inteval, per epoch for regular training, per n_batches for ve tranining', default=0, type=int)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--allow_run_as_root', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--bind_to_core', default=False, action='store_true')
    return parser


def set_global_seeds(i):
    try:
        import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    myseed = i  + 1000 * rank if i is not None else None
    logger.info('setting global seed', myseed)

    tf.set_random_seed(myseed)
    np.random.seed(myseed)
    random.seed(myseed)

def make_vec_env(
        make_wrapped_env,
        num_env, seed,
        mpi_rank,
        monitor_log_dir,
        flatten_dict_observations=False,
        reward_scale=1.0,
):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    seed = seed + 10000 * mpi_rank if seed is not None else None
    set_global_seeds(seed)

    def make_env_factory(subrank):
        def make_env():

            # wrapped_env -> flatten_observation -> monitor -> clip_action -> scale_reward

            env = make_wrapped_env()
            env.seed(seed + subrank if seed is not None else None)

            if flatten_dict_observations:# and isinstance(env.observation_space, gym.spaces.Dict):
                env = FlattenObservation(env)

            env = Monitor(env,
                          osp.join(monitor_log_dir, str(mpi_rank) + '.' + str(subrank)),  # training and eval write to same file?
                          allow_early_resets=True)
            env = ClipActionsWrapper(env)

            if reward_scale != 1:
                env = retro_wrappers.RewardScaler(env, reward_scale)

            return env
        return make_env
    return DummyVecEnv([make_env_factory(i) for i in range(num_env)])

