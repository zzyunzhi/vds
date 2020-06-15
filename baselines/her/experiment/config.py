import os
import numpy as np
import gym

from baselines import logger
from baselines.her.ddpg import DDPG
from baselines.her.goal_sampler import make_goal_sampler_factory_random_init_ob
from baselines.her.value_ensemble_v1 import ValueEnsemble
from baselines.her.her_sampler import make_sample_her_transitions
from baselines.bench.monitor import Monitor

DEFAULT_ENV_PARAMS = {
    'FetchReach-v1': {
        'n_cycles': 10,
    },
}


DEFAULT_PARAMS = {
    # env
    'max_u': 1.,  # max absolute value of actions on different coordinates
    # ddpg
    'layers': 3,  # number of layers in the critic/actor networks
    'hidden': 256,  # number of neurons in each hidden layers
    'network_class': 'baselines.her.actor_critic:ActorCritic',
    'Q_lr': 0.001,  # critic learning rate
    'pi_lr': 0.001,  # actor learning rate
    'buffer_size': int(1E6),  # for experience replay
    'polyak': 0.95,  # polyak averaging coefficient
    'action_l2': 1.0,  # quadratic penalty on actions (before rescaling by max_u)
    'clip_obs': 200.,
    'relative_goals': False,
    # training
    'n_cycles': 50,  # per epoch
    'n_batches': 40,  # training batches per cycle
    'batch_size': 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'n_test_rollouts': 10,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    'test_with_polyak': False,  # run test episodes with the target network
    # exploration
    'random_eps': 0.3,  # percentage of time a random action is taken
    'noise_eps': 0.2,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    # HER
    'replay_strategy': 'future',  # supported modes: future, none
    'replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future
    # normalization
    'norm_eps': 0.01,  # epsilon used for observation normalization
    'norm_clip': 5,  # normalized observations are cropped to this values

    'bc_loss': 0, # whether or not to use the behavior cloning loss as an auxilliary loss
    'q_filter': 0, # whether or not a Q value filter should be used on the Actor outputs
    'num_demo': 100, # number of expert demo episodes
    'demo_batch_size': 128, #number of samples to be used from the demonstrations buffer, per mpi thread 128/1024 or 32/256
    'prm_loss_weight': 0.001, #Weight corresponding to the primary loss
    'aux_loss_weight':  0.0078, #Weight corresponding to the auxilliary loss also called the cloning loss

    # value ensemble
    'size_ensemble': 3,
    # 'inference_clip_pos_returns': True,
    've_buffer_size': int(1E6),
    've_lr': 0.001,
    've_n_batches': 100,
    've_batch_size': int(1E3),
    've_use_Q': True,  # not used by value_ensemble_v1
    've_use_double_network': True,
    'disagreement_fun_name': 'std',

    # HER for value ensemble
    've_replay_strategy': 'none',  # supported modes: future, none
    've_replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future

    # goal sampler
    'n_candidates': 1000,
}


CACHED_ENVS = {}


def cached_make_env(make_env):
    """
    Only creates a new environment from the provided function if one has not yet already been
    created. This is useful here because we need to infer certain properties of the env, e.g.
    its observation and action spaces, without any intend of actually using it.
    """
    if make_env not in CACHED_ENVS:
        env = make_env()
        CACHED_ENVS[make_env] = env
        env.goal = env.unwrapped._sample_goal()
        env.reset(reset_goal=False)
    return CACHED_ENVS[make_env]


def prepare_params(kwargs):
    # DDPG params
    ddpg_params = dict()
    env_name = kwargs['env_name']

    def make_env(subrank=None):
        env = gym.make(env_name)
        if kwargs['env_type'] == 'goal':
            from baselines.envs.goal_sampler_env_wrapper import GoalSamplerEnvWrapper
            env = GoalSamplerEnvWrapper(env)
        if subrank is not None and logger.get_dir() is not None:
            try:
                from mpi4py import MPI
                mpi_rank = MPI.COMM_WORLD.Get_rank()
            except ImportError:
                MPI = None
                mpi_rank = 0
                logger.warn('Running with a single MPI process. This should work, but the results may differ from the ones publshed in Plappert et al.')

            max_episode_steps = env._max_episode_steps
            env =  Monitor(env,
                           os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(subrank)),
                           allow_early_resets=True)
            # hack to re-expose _max_episode_steps (ideally should replace reliance on it downstream)
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        return env

    kwargs['make_env'] = make_env
    tmp_env = cached_make_env(kwargs['make_env'])
    if kwargs['env_type'] == 'goal':
        kwargs['T'] = tmp_env.env._max_episode_steps
    else:
        kwargs['T'] = tmp_env._max_episode_steps

    kwargs['max_u'] = np.array(kwargs['max_u']) if isinstance(kwargs['max_u'], list) else kwargs['max_u']
    kwargs['gamma'] = 1. - 1. / kwargs['T']
    if 'lr' in kwargs:
        kwargs['pi_lr'] = kwargs['lr']
        kwargs['Q_lr'] = kwargs['lr']
        del kwargs['lr']
    for name in ['buffer_size', 'hidden', 'layers',
                 'network_class',
                 'polyak',
                 'batch_size', 'Q_lr', 'pi_lr',
                 'norm_eps', 'norm_clip', 'max_u',
                 'action_l2', 'clip_obs', 'relative_goals']:
        ddpg_params[name] = kwargs[name]
        kwargs['_' + name] = kwargs[name]
        del kwargs[name]
    kwargs['ddpg_params'] = ddpg_params

    return kwargs


def prepare_ve_params(kwargs):
    # policy params
    kwargs = prepare_params(kwargs)
    ddpg_params = kwargs['ddpg_params']

    # value ensemble params
    ve_params = dict()

    for name in ['size_ensemble']:#, 'inference_clip_pos_returns']:
        ve_params[name] = kwargs[name]
        kwargs['_' + name] = kwargs[name]
        del kwargs[name]

    for name in ['buffer_size', 'lr', 'batch_size', 'use_Q', 'use_double_network']:
        ve_params[name] = kwargs[f've_{name}']
        kwargs[f'_ve_{name}'] = kwargs[f've_{name}']
        del kwargs[f've_{name}']

    # following ddpg params
    for name in ['hidden', 'layers', 'norm_eps', 'norm_clip',
                 'max_u', 'clip_obs', 'relative_goals']:
        ve_params[name] = ddpg_params[name]
    kwargs['ve_params'] = ve_params

    # goal sampler params
    gs_params = dict()
    for name in ['n_candidates', 'disagreement_fun_name']:
        gs_params[name] = kwargs[name]
        kwargs['_' + name] = kwargs[name]
        del kwargs[name]
    kwargs['gs_params'] = gs_params

    return kwargs


def log_params(params, logger=logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))


def configure_her(params):
    env = cached_make_env(params['make_env'])
    # env.reset()

    def reward_fun(ag_2, g, info):  # vectorized
        return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)

    # Prepare configuration for HER.
    her_params = {
        'reward_fun': reward_fun,
    }
    for name in ['replay_strategy', 'replay_k']:
        her_params[name] = params[name]
        params['_' + name] = her_params[name]
        del params[name]
    sample_her_transitions = make_sample_her_transitions(**her_params)

    return sample_her_transitions


def configure_ve_her(params):
    env = cached_make_env(params['make_env'])
    # try:
    #     env.reset()
    # except NotImplementedError:
    #     env.get_reset_obs()

    def reward_fun(ag_2, g, info):  # vectorized
        return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)

    # Prepare configuration for HER.
    ddpg_her_params = {
        'reward_fun': reward_fun,
    }
    for name in ['replay_strategy', 'replay_k']:
        ddpg_her_params[name] = params[name]
        params['_' + name] = ddpg_her_params[name]
        del params[name]
    ddpg_sample_transitions = make_sample_her_transitions(**ddpg_her_params)

    ve_her_params = {
        'reward_fun': reward_fun,
    }
    for name in ['replay_strategy', 'replay_k']:
        ve_her_params[name] = params[f've_{name}']
        params[f'_ve_{name}'] = ve_her_params[name]
        del params[f've_{name}']
    ve_sample_transitions = make_sample_her_transitions(**ve_her_params)

    return ddpg_sample_transitions, ve_sample_transitions

def configure_disagreement(params, value_ensemble, policy):
    env = cached_make_env(params['make_env'])
    # env.get_reset()

    disagreement_params = dict(
        # 'static_init_obs': env.static_init_obs,
        sample_goals_fun=lambda size: [env.unwrapped._sample_goal() for _ in range(size)],
        policy=policy,
        value_ensemble=value_ensemble,
    )
    gs_params = params['gs_params']
    sample_disagreement_goals = make_goal_sampler_factory_random_init_ob(**disagreement_params,
        n_candidates=gs_params["n_candidates"], disagreement_fun_name=gs_params["disagreement_fun_name"]
    )

    sample_uniform_goals = make_goal_sampler_factory_random_init_ob(**disagreement_params,
        n_candidates=1, disagreement_fun_name='uniform',
    )

    return sample_disagreement_goals, sample_uniform_goals


def simple_goal_subtract(a, b):
    assert a.shape == b.shape
    return a - b


def configure_rollout_worker_params(params):
    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'compute_Q': False,
        'T': params['T'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'compute_Q': True,
        'T': params['T'],
    }
    plotter_params = {
        'exploit': True,
        'compute_Q': False,
        'use_target_net': params['test_with_polyak'],
        'T': params["T"],
        'history_len': 100000,
        'rollout_batch_size': 1,
    }

    params['_test_with_polyak'] = params['test_with_polyak']
    del params['test_with_polyak']

    for name in ['rollout_batch_size', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]
        params['_' + name] = params[name]
        del params[name]

    return rollout_params, eval_params, plotter_params

#
# def configure_plotter(policy, value_ensemble, plotter_worker, params, report):
#     import functools
#     from baselines.envs.visualization.utils import make_plotter, plot_heatmap
#
#     env = cached_make_env(params['make_env'])
#     goals, plotter_info = env.get_grid_goals(sampling_res=5, feasible=True)  # use 0 for Point-v0
#
#     # TODO: plot all goals here
#
#     plot_heatmap_fun = functools.partial(plot_heatmap, show_heatmap=False, **plotter_info)
#
#     return make_plotter(
#         init_ob=env.reset(reset_goal=False),
#         policy=policy,
#         value_ensemble=value_ensemble,
#         goals=goals,
#         disagreement_str='std', # params['gs_params']['disagreement_fun_name'],
#         plotter_worker=plotter_worker,
#         gamma=params['gamma'],
#         report=report,
#         plot_heatmap_fun=plot_heatmap_fun,
#         eval_policy=False,
#     )


def configure_ddpg(dims, params, reuse=False, use_mpi=True, clip_return=True):
    sample_her_transitions = configure_her(params)
    # Extract relevant parameters.
    gamma = params['gamma']
    rollout_batch_size = params['rollout_batch_size']
    ddpg_params = params['ddpg_params']

    input_dims = dims.copy()

    # DDPG agent
    # env = cached_make_env(params['make_env'])
    # env.reset()
    ddpg_params.update({'input_dims': input_dims,  # agent takes an input observations
                        'T': params['T'],
                        'scope': 'ddpg',
                        'clip_pos_returns': True,  # clip positive returns
                        'clip_return': (1. / (1. - gamma)) if clip_return else np.inf,  # max abs of return
                        'rollout_batch_size': rollout_batch_size,
                        'subtract_goals': simple_goal_subtract,
                        'sample_transitions': sample_her_transitions,
                        'gamma': gamma,
                        'bc_loss': params['bc_loss'],
                        'q_filter': params['q_filter'],
                        'num_demo': params['num_demo'],
                        'demo_batch_size': params['demo_batch_size'],
                        'prm_loss_weight': params['prm_loss_weight'],
                        'aux_loss_weight': params['aux_loss_weight'],
                        })
    ddpg_params['info'] = {
        'env_name': params['env_name'],
    }
    policy = DDPG(reuse=reuse, **ddpg_params, use_mpi=use_mpi)
    return policy


def configure_dims(params):
    env = cached_make_env(params['make_env'])
    # env.reset()
    obs, _, _, info = env.step(env.action_space.sample())

    dims = {
        'o': obs['observation'].shape[0],
        'u': env.action_space.shape[0],
        'g': obs['desired_goal'].shape[0],
    }
    for key, value in info.items():
        value = np.array(value)
        if value.ndim == 0:
            value = value.reshape(1)
        dims['info_{}'.format(key)] = value.shape[0]
    return dims


def configure_ve_ddpg(dims, params, reuse=False, use_mpi=True, clip_return=True, policy_pkl=None):
    # Extract relevant parameters.
    gamma = params['gamma']
    rollout_batch_size = params['rollout_batch_size']
    # env = cached_make_env(params['make_env'])
    # env.get_reset_obs()
    # env.reset()

    ddpg_sample_transitions, ve_sample_transitions = configure_ve_her(params)

    # DDPG agent
    if policy_pkl is not None:
        # load froze policy
        import joblib
        logger.info('loading policy...')
        data = joblib.load(policy_pkl)
        policy = data['policy']

    else:
        ddpg_params = params['ddpg_params']
        ddpg_params.update({'input_dims': dims.copy(),  # agent takes an input observations
                            'T': params['T'],
                            'scope': 'ddpg',
                            'clip_pos_returns': True,  # clip positive returns
                            'clip_return': (1. / (1. - gamma)) if clip_return else np.inf,  # max abs of return
                            'rollout_batch_size': rollout_batch_size,
                            'subtract_goals': simple_goal_subtract,
                            'sample_transitions': ddpg_sample_transitions,
                            'gamma': gamma,
                            'bc_loss': params['bc_loss'],
                            'q_filter': params['q_filter'],
                            'num_demo': params['num_demo'],
                            'demo_batch_size': params['demo_batch_size'],
                            'prm_loss_weight': params['prm_loss_weight'],
                            'aux_loss_weight': params['aux_loss_weight'],
                            })
        ddpg_params['info'] = {
            'env_name': params['env_name'],
        }
        policy = DDPG(reuse=reuse, **ddpg_params, use_mpi=use_mpi)

    ve_params = params['ve_params']
    ve_params.update({
        'input_dims': dims.copy(),
        'T': params['T'],
        'scope': 've' if policy_pkl is None else 've-trainable',  # a hack to avoid duplicate vars when policy_pkl is loaded
        'rollout_batch_size': rollout_batch_size,
        'subtract_goals': simple_goal_subtract,
        'clip_pos_returns': True,  # following ddpg configuration
        'clip_return': (1. / (1. - gamma)) if clip_return else np.inf,  # following ddpg configuration
        'sample_transitions': ve_sample_transitions,
        'gamma': gamma,
        # TODO: tmp hack below
        'polyak': ddpg_params['polyak'],
    })
    value_ensemble = ValueEnsemble(reuse=reuse, **ve_params)

    # goal sampling function to be passed in vector env
    sample_disagreement_goals_fun, sample_uniform_goals_fun = configure_disagreement(
        params,
        value_ensemble=value_ensemble,
        policy=policy
    )

    return policy, value_ensemble, sample_disagreement_goals_fun, sample_uniform_goals_fun
