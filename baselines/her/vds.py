import os
import time

import numpy as np
import json
import joblib
from mpi4py import MPI

from baselines import logger
from baselines.common.mpi_moments import mpi_moments
import baselines.her.experiment.vds_config as config
from baselines.her.rollout import RolloutWorker


def get_mpi_moments(value):
    if not isinstance(value, list):
        value = [value]
    if not any(value):
        value = [0.]
    return mpi_moments(np.array(value))


def mpi_average(value):
    data = get_mpi_moments(value)
    # assert data[2] == 2
    # return round(float(data[0]), 2), round(float(data[1]), 2)
    return data[0]


def mpi_sum(value):
    data = get_mpi_moments(value)
    return data[0] * data[2]


def train(*, policy, value_ensemble, #goal_sampler_factory,
        rollout_worker, evaluator, plotter,
          n_epochs, n_test_rollouts, n_cycles, n_batches, ve_n_batches, update_goal_sampler_interval,
          save_interval, save_path):

    rank = MPI.COMM_WORLD.Get_rank()

    if save_path:
        save_path = os.path.join(save_path, 'itr_{}.pkl')

    print("start training...")
    to_dump = dict(valiue_ensemble=value_ensemble, policy=policy, goal_history=None)
    goal_history = []

    # reused_states = None
    # _, goal_sampler_info = goal_sampler_factory(reused_states=reused_states)
    # reused_states = goal_sampler_info["all_states"]  # use fixed presampler to test mpi average

    # while not value_ensemble.buffer_full:
    #     rollout_worker.clear_history()
    #     episode = rollout_worker.generate_rollouts()
    #     value_ensemble.store_episode(episode)

    for epoch in range(n_epochs):
        # train
        rollout_worker.clear_history()
        time_rollout, time_ve, time_train = 0, 0, 0
        ve_loss_history, critic_loss_history, actor_loss_history = [], [], []

        for _ in range(n_cycles):
            t = time.time()
            episode = rollout_worker.generate_rollouts()
            time_rollout += time.time() - t

            # store goals for visualization
            if rank == 0 and save_interval > 0 and save_path:
                goal_history.append(episode['g'][:, -1, :])  # (rollout_batch_size, goal_dim)

            # train the value ensemble
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

        # if epoch % update_goal_sampler_interval == 0:
        #
        #     # # train the value ensemble
        #     # t = time.time()
        #     # for _ in range(ve_n_batches * min(epoch+1, update_goal_sampler_interval)):
        #     #     ve_loss = value_ensemble.train()
        #     #     ve_loss_history.append(ve_loss)
        #     # value_ensemble.update_target_net()
        #     # time_ve += time.time() - t
        #
        #     goal_sampler, goal_sampler_info = goal_sampler_factory(reused_states=reused_states)
        #     reused_states = goal_sampler_info['reused_states']
        #     rollout_worker.envs_op('update_goal_sampler', goal_sampler=goal_sampler)

        # test
        evaluator.clear_history()
        t = time.time()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()
        time_eval = time.time() - t

        # record total timesteps
        logger.record_tabular('timesteps', mpi_sum(policy.buffer.get_transitions_stored()))

        # record loss
        logger.record_tabular('ve/loss', mpi_average(np.mean(ve_loss_history)))
        logger.record_tabular('train/critic_loss', mpi_average(np.mean(critic_loss_history)))
        logger.record_tabular('train/actor_loss', mpi_average(np.mean(actor_loss_history)))

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
        logger.record_tabular('time_rollout', mpi_average(time_rollout))
        logger.record_tabular('time_ve', mpi_average(time_ve))
        if ve_n_batches > 0:
            logger.record_tabular('time_ve_per_batch', mpi_average(time_ve/ve_n_batches))
        logger.record_tabular('time_train', mpi_average(time_train))
        logger.record_tabular('time_train_per_batch', mpi_average(time_train/n_batches))
        logger.record_tabular('time_eval', mpi_average(time_eval))

        if rank == 0:
            logger.dump_tabular()
        #     if plotter is not None:
        #         goal_history = np.concatenate(goal_history, axis=0)
        #         plotter(epoch, goal_history)
        # goal_history = []

        if rank == 0 and save_interval > 0 and epoch % save_interval == 0 and save_path:
            goal_history = np.concatenate(goal_history, axis=0)
            logger.log(f'dumping at epoch {epoch}, history length {len(goal_history)}')

            to_dump['goal_history'] = goal_history
            goal_history = []
            # to_dump['samples'] = rollout_worker.venv.reset_history()
            joblib.dump(to_dump, save_path.format(epoch), compress=3)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]

    return dict(policy=policy, value_ensemble=value_ensemble)


def learn(*, venv, eval_venv, plotter_venv,
          params,
          save_interval=5,
          save_path=None,
          ):

    dims = config.configure_dims(params)
    logger.info("env dims", dims)

    policy, value_ensemble, sample_disagreement_goals_fun, sample_uniform_goals_fun = \
        config.configure_all(dims=dims, params=params)
    venv.envs_op("update_goal_sampler", goal_sampler=sample_disagreement_goals_fun)
    eval_venv.envs_op("update_goal_sampler", goal_sampler=sample_uniform_goals_fun)

    # policy, value_ensemble, goal_sampler_factory, feasible_goal_uniform_sampler = config.configure_all(dims=dims, params=params)
    #
    # # if debug:
    # #     goal_sampler = lambda: np.asarray((2,2))  # FIXME: is this feasible??
    # #     venv.envs_op('update_goal_sampler', goal_sampler=goal_sampler)
    # #     eval_venv.envs_op('update_goal_sampler', goal_sampler=goal_sampler)
    # #     plotter_venv.envs_op('update_goal_sampler', goal_sampler=goal_sampler)
    # # else:
    # warmup_states = goal_sampler_factory()[1]["all_states"]
    # assert venv.env_op('sample_goal') is None
    # venv.envs_op('update_goal_sampler', goal_sampler=lambda: warmup_states[np.random.choice(len(warmup_states))])
    # eval_venv.envs_op('update_goal_sampler', goal_sampler=feasible_goal_uniform_sampler)
    # if plotter_venv is not None:
    #     plotter_venv.envs_op('update_goal_sampler', goal_sampler=feasible_goal_uniform_sampler)
    #
    # # assert env.sample_goals_fun is None
    # # if policy_pkl is not None:
    # #     env.set_sample_goals_fun(sample_dummy_goals_fun)
    # # else:
    # #     env.set_sample_goals_fun(sample_disagreement_goals_fun)
    #
    # # if load_path is not None:
    # #     tf_util.load_variables(os.path.join(load_path, 'final_policy_params.joblib'))
    # #     return play(env=env, policy=policy)

    config.prepare_rollout_worker_params(params)
    rollout_worker = RolloutWorker(venv, policy, dims, logger, **params['rollout_params'])
    evaluator = RolloutWorker(eval_venv, policy, dims, logger, **params['eval_params'])

    if plotter_venv is not None:
        raise NotImplementedError
        # from baselines.misc.html_report import HTMLReport
        # plotter_worker = RolloutWorker(plotter_venv, policy, dims, logger, **params['plotter_params'])
        # rank = MPI.COMM_WORLD.Get_rank()
        # report = HTMLReport(os.path.join(save_path, f'report-{rank}.html'), images_per_row=8)
        #
        # # report.add_header("{}".format(EXPERIMENT_TYPE))
        # # report.add_text(format_dict(v))
        # plotter = config.configure_plotter(policy, value_ensemble, plotter_worker, params, report)
    else:
        plotter = None

    n_cycles = params['n_cycles']
    n_epochs = params['timesteps_per_cpu'] // n_cycles // rollout_worker.T // rollout_worker.rollout_batch_size
    params['n_epochs'] = n_epochs
    params['timesteps_per_cpu'] = n_epochs * n_cycles * rollout_worker.T * rollout_worker.rollout_batch_size

    config.log_params(params, logger=logger)

    # if policy_pkl is not None:
    #     train_fun = train_ve
    #     evaluator = None
    # else:

    # construct evaluator
    # assert eval_venv.sample_goals_fun is None
    # eval_venv.set_sample_goals_fun(sample_dummy_goals_fun)

    return train(
        save_path=save_path, policy=policy, value_ensemble=value_ensemble, rollout_worker=rollout_worker,
        # goal_sampler_factory=goal_sampler_factory,
        plotter=plotter,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'], ve_n_batches=params['ve_n_batches'],
        update_goal_sampler_interval=params['update_goal_sampler_interval'],
        save_interval=save_interval)
