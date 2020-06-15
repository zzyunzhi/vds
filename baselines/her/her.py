import os
import sys

import click
import numpy as np
import json
import joblib
from mpi4py import MPI

from baselines import logger
import baselines.common.tf_util as U
from baselines.common import set_global_seeds, tf_util
from baselines.common.mpi_moments import mpi_moments
from baselines.common.mpi_fork import mpi_fork_run_as_root, mpi_fork
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker

def mpi_average(value):
    if not isinstance(value, list):
        value = [value]
    if not any(value):
        value = [0.]
    return mpi_moments(np.array(value))[0]


def train(*, policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, save_interval,
          save_path, demo_file, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    if save_path:
        save_path = os.path.join(save_path, 'itr_{}.pkl')

    logger.info("Training...")
    to_dump = dict(policy=policy)

    if policy.bc_loss == 1: policy.init_demo_buffer(demo_file) #initialize demo buffer if training with demonstrations

    # num_timesteps = n_epochs * n_cycles * rollout_length * number of rollout workers
    for epoch in range(n_epochs):
        # train
        rollout_worker.clear_history()
        critic_loss_history, actor_loss_history = [], []
        for _ in range(n_cycles):
            episode = rollout_worker.generate_rollouts()
            policy.store_episode(episode)
            for _ in range(n_batches):
                critic_loss, actor_loss = policy.train()
                critic_loss_history.append(critic_loss)
                actor_loss_history.append(actor_loss)
            policy.update_target_net()

        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # record timesteps
        logger.record_tabular('timesteps', policy.buffer.get_transitions_stored())

        # record loss
        logger.record_tabular('train/critic_loss', np.mean(critic_loss_history))
        logger.record_tabular('train/actor_loss', np.mean(actor_loss_history))

        # record logs
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs('ddpg'):
            logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()

        if rank == 0 and save_interval > 0 and epoch % save_interval == 0 and save_path:
            joblib.dump(to_dump, save_path.format(epoch), compress=3)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]

    return policy


def learn(*, network, env, total_timesteps, num_cpu, allow_run_as_root,
    seed=None,
    eval_env=None,
    replay_strategy='future',
    save_interval=5,
    clip_return=True,
    demo_file=None,
    override_params=None,
    load_path=None,
    save_path=None,
    **kwargs
):
    rank = MPI.COMM_WORLD.Get_rank()
    logger.info('before mpi_fork: rank', rank, 'num_cpu', MPI.COMM_WORLD.Get_size())

    if num_cpu > 1:
        if allow_run_as_root:
            whoami = mpi_fork_run_as_root(num_cpu)
        else:
            whoami = mpi_fork(num_cpu)
        if whoami == 'parent':
            logger.info('parent exiting with code 0...')
            sys.exit(0)

        U.single_threaded_session().__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    num_cpu = MPI.COMM_WORLD.Get_size()
    logger.info('after mpi_fork: rank', rank, 'num_cpu', num_cpu)

    override_params = override_params or {}

    # Seed everything.
    rank_seed = seed + 1000000 * rank if seed is not None else None
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    env_name = env.spec.id
    params['env_name'] = env_name
    params['replay_strategy'] = replay_strategy
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter
    params['rollout_batch_size'] = env.num_envs
    params['num_cpu'] = num_cpu
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
         json.dump(params, f)
    params = config.prepare_params(params)

    if demo_file is not None:
        params['bc_loss'] = 1
    params.update(kwargs)

    config.log_params(params, logger=logger)

    if num_cpu == 1:
        logger.warn()
        logger.warn('*** Warning ***')
        logger.warn(
            'You are running HER with just a single MPI worker. This will work, but the ' +
            'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
            'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
            'are looking to reproduce those results, be aware of this. Please also refer to ' +
            'https://github.com/openai/baselines/issues/314 for further details.')
        logger.warn('****************')
        logger.warn()

    dims = config.configure_dims(params)
    policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return)
    if load_path is not None:
        tf_util.load_variables(load_path)

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    eval_env = eval_env or env

    rollout_worker = RolloutWorker(env, policy, dims, logger, monitor=True, **rollout_params)
    evaluator = RolloutWorker(eval_env, policy, dims, logger, **eval_params)

    n_cycles = params['n_cycles']
    n_epochs = total_timesteps // n_cycles // rollout_worker.T // rollout_worker.rollout_batch_size
    logger.info("actual total timesteps : {}".format(n_epochs * n_cycles * rollout_worker.T * rollout_worker.rollout_batch_size))

    return train(
        save_path=save_path, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        save_interval=save_interval, demo_file=demo_file)


@click.command()
@click.option('--env', type=str, default='FetchReach-v1', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--total_timesteps', type=int, default=int(5e5), help='the number of timesteps to run')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.option('--demo_file', type=str, default = 'PATH/TO/DEMO/DATA/FILE.npz', help='demo data file path')
def main(**kwargs):
    learn(**kwargs)


if __name__ == '__main__':
    main()
