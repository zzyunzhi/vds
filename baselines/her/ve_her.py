import os
import sys
import time

import numpy as np
import json
import joblib
from itertools import count
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

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


def train(*, policy, value_ensemble, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, ve_n_batches,
          save_interval, save_path, plotter):
    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0

    if save_path:
        save_path = os.path.join(save_path, 'itr_{}.pkl')

    logger.info("Training...")
    to_dump = dict(value_ensemble=value_ensemble, policy=policy)
    goal_history = []

    # while not value_ensemble.buffer_full:
    #     rollout_worker.clear_history()
    #     episode = rollout_worker.generate_rollouts()
    #     value_ensemble.store_episode(episode)

    # num_timesteps = n_epochs * n_cycles * rollout_length * number of rollout workers
    for epoch in range(n_epochs):
        # train
        rollout_worker.clear_history()
        time_rollout, time_ve, time_train = 0, 0, 0
        ve_loss_history, critic_loss_history, actor_loss_history = [], [], []
        for _ in range(n_cycles):
            t = time.time()
            episode = rollout_worker.generate_rollouts()
            time_rollout += time.time() - t

            # store goals for visualizatino
            goal_history.append(episode['g'][:, -1, :])  # (rollout_batch_size, goal_dim)

            # train the value ensemble
            # label u_2 because value_ensemble doesn't have access to the policy
            if value_ensemble.size_ensemble > 0:
                t = time.time()
                value_ensemble.store_episode(episode)
                for _ in range(ve_n_batches):
                    ve_loss = value_ensemble.train(policy=policy)
                    ve_loss_history.append(ve_loss)
                value_ensemble.update_target_net()
                time_ve += time.time() - t

            # train the policy
            t = time.time()
            policy.store_episode(episode)
            for _ in range(n_batches):
                critic_loss, actor_loss = policy.train()
                critic_loss_history.append(critic_loss)
                actor_loss_history.append(actor_loss)
            policy.update_target_net()
            time_train += time.time() - t

        # test
        evaluator.clear_history()
        t = time.time()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()
        time_eval = time.time() - t

        # record total timesteps
        logger.record_tabular('timesteps', policy.buffer.get_transitions_stored())

        # record loss
        logger.record_tabular('ve/loss', np.mean(ve_loss_history))
        logger.record_tabular('train/critic_loss', np.mean(critic_loss_history))
        logger.record_tabular('train/actor_loss', np.mean(actor_loss_history))

        # record logs
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in value_ensemble.logs('ve'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs('ddpg'):
            logger.record_tabular(key, mpi_average(val))

        # record time
        logger.record_tabular('time_rollout', time_rollout)
        logger.record_tabular('time_ve', time_ve)
        logger.record_tabular('time_train', time_train)
        logger.record_tabular('time_eval', time_eval)

        if rank == 0:
            logger.dump_tabular()
            if plotter is not None:
                goal_history = np.concatenate(goal_history, axis=0)
                plotter(epoch, goal_history)
            goal_history = []

        if rank == 0 and save_interval > 0 and epoch % save_interval == 0 and save_path:
            # to_dump['samples'] = rollout_worker.venv.reset_history()
            joblib.dump(to_dump, save_path.format(epoch), compress=3)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]

    return policy, value_ensemble


def train_ve(*, policy, value_ensemble, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, ve_n_batches,
          save_interval, save_path):
    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        rank = 0

    if save_path:
        save_path = os.path.join(save_path, 'itr_{}.pkl')

    logger.info("Training with freezing policy...")
    to_dump = dict(value_ensemble=value_ensemble)

    while not value_ensemble.buffer_full:
        rollout_worker.clear_history()
        episode = rollout_worker.generate_rollouts()
        value_ensemble.store_episode(episode)

    # num_timesteps = n_epochs * n_cycles * rollout_length * number of rollout workers + ve_buffer_size

    training_counter = count()
    for cycle in range(n_epochs*n_cycles):
        # train
        rollout_worker.clear_history()
        episode = rollout_worker.generate_rollouts()
        # train the value ensemble
        value_ensemble.store_episode(episode)
        for batch in range(ve_n_batches):
            training_epoch = next(training_counter)

            t = time.time()
            ve_loss = value_ensemble.train()

            if save_interval > 0 and training_epoch % save_interval == 0:
                # record total timesteps
                logger.record_tabular('timesteps', value_ensemble.buffer_get_transitions_stored())

                # record loss
                logger.record_tabular('ve/loss', ve_loss)

                # record time
                logger.record_tabular('time_ve', time.time() - t)

                # record logs
                logger.record_tabular('cycle', cycle)
                logger.record_tabular('batch', batch)
                for key, val in value_ensemble.logs('ve'):
                    logger.record_tabular(key, mpi_average(val))

                if rank == 0:
                    logger.dump_tabular()

        if rank == 0 and save_path:
            joblib.dump(to_dump, save_path.format(cycle), compress=3)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]

    return policy, value_ensemble

def play(*, env, policy):

    logger.log("Running trained models")
    obs = env.reset()

    episode_rew = np.zeros(env.num_envs)
    while True:
        actions, _, _, _ = policy.step(obs)

        obs, rew, done, _ = env.step(actions)
        episode_rew += rew
        done_any = done.any() if isinstance(done, np.ndarray) else done
        if done_any:
            for i in np.nonzero(done)[0]:
                print('rew={}'.format(rew[i]))
                print('episode_rew={}'.format(episode_rew[i]))
                episode_rew[i] = 0
            while True: pass

        env.render()


def learn(*, env_type, env, eval_env, plotter_env, total_timesteps, num_cpu, allow_run_as_root, bind_to_core,
    seed=None,
    save_interval=5,
    clip_return=True,
    override_params=None,
    load_path=None,
    save_path=None,
    policy_pkl=None,
):

    rank = MPI.COMM_WORLD.Get_rank()
    logger.info('before mpi_fork: rank', rank, 'num_cpu', MPI.COMM_WORLD.Get_size())

    if num_cpu > 1:
        if allow_run_as_root:
            whoami = mpi_fork_run_as_root(num_cpu, bind_to_core=bind_to_core)
        else:
            whoami = mpi_fork(num_cpu, bind_to_core=bind_to_core)
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
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter
    params['rollout_batch_size'] = env.num_envs
    params['num_cpu'] = num_cpu
    params['env_type'] = env_type
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
         json.dump(params, f)
    params = config.prepare_ve_params(params)

    dims = config.configure_dims(params)
    policy, value_ensemble, sample_disagreement_goals_fun, sample_uniform_goals_fun = \
        config.configure_ve_ddpg(dims=dims, params=params, clip_return=clip_return, policy_pkl=policy_pkl)

    if policy_pkl is not None:
        env.set_sample_goals_fun(sample_dummy_goals_fun)
    else:
        env.envs_op("update_goal_sampler", goal_sampler=sample_disagreement_goals_fun)
        eval_env.envs_op("update_goal_sampler", goal_sampler=sample_uniform_goals_fun)
        if plotter_env is not None:
            plotter_env.envs_op("update_goal_sampler", goal_sampler=sample_uniform_goals_fun)

    if load_path is not None:
        tf_util.load_variables(os.path.join(load_path, 'final_policy_params.joblib'))
        return play(env=env, policy=policy)

    rollout_params, eval_params, plotter_params = config.configure_rollout_worker_params(params)

    rollout_worker = RolloutWorker(env, policy, dims, logger, monitor=True, **rollout_params)

    n_cycles = params['n_cycles']
    n_epochs = total_timesteps // n_cycles // rollout_worker.T // rollout_worker.rollout_batch_size
    params['n_epochs'] = n_epochs
    params['total_timesteps'] = n_epochs * n_cycles * rollout_worker.T * rollout_worker.rollout_batch_size

    config.log_params(params, logger=logger)

    if policy_pkl is not None:
        train_fun = train_ve
        evaluator = None
    else:
        train_fun = train
        # construct evaluator
        # assert eval_env.sample_goals_fun is None
        # eval_env.set_sample_goals_fun(sample_dummy_goals_fun)
        evaluator = RolloutWorker(eval_env, policy, dims, logger, **eval_params)
        if plotter_env is not None:
            raise NotImplementedError
            # from baselines.misc.html_report import HTMLReport
            # plotter_worker = RolloutWorker(plotter_env, policy, dims, logger, **plotter_params)
            # rank = MPI.COMM_WORLD.Get_rank()
            # report = HTMLReport(os.path.join(save_path, f'report-{rank}.html'), images_per_row=8)
            #
            # # report.add_header("{}".format(EXPERIMENT_TYPE))
            # # report.add_text(format_dict(v))
            # plotter = config.configure_plotter(policy, value_ensemble, plotter_worker, params, report)
        else:
            plotter = None

    return train_fun(
        save_path=save_path, policy=policy, value_ensemble=value_ensemble, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'], ve_n_batches=params['ve_n_batches'],
        save_interval=save_interval, plotter=plotter)
