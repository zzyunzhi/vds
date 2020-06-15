import functools

from baselines.her.ddpg import DDPG
from baselines.her.goal_sampler import *
from baselines.her.value_ensemble_v1 import ValueEnsemble
from baselines.her.her_sampler import make_sample_her_transitions
# from baselines.envs.visualization.ant_maze_visualizer import make_plotter, plot_maze_heatmap
# from baselines.envs.visualization.utils import make_plotter, plot_heatmap
from collections import defaultdict


DEFAULT_ENV_PARAMS = defaultdict(dict)

DEFAULT_PARAMS = {
    # ddpg
    # 'max_u': 1.,  # max absolute value of actions on different coordinates
    # ddpg/network
    'layers': 3,  # number of layers in the critic/actor networks
    'hidden': 256,  # number of neurons in each hidden layers
    'network_class': 'baselines.her.actor_critic:ActorCritic',
    'Q_lr': 0.001,  # critic learning rate
    'pi_lr': 0.001,  # actor learning rate
    'polyak': 0.95,  # polyak averaging coefficient
    'action_l2': 1.0,  # quadratic penalty on actions (before rescaling by max_u)
    # ddpg/clipping_and_normalization
    'clip_obs': 200.,
    'clip_pos_returns': True,
    'clip_return': True,
    'norm_eps': 0.01,  # epsilon used for observation normalization
    'norm_clip': 5,  # normalized observations are cropped to this values
    # ddpg/imitation_learning
    'relative_goals': False,  # TODO
    'bc_loss': 0, # whether or not to use the behavior cloning loss as an auxilliary loss
    'q_filter': 0, # whether or not a Q value filter should be used on the Actor outputs
    'num_demo': 0,#100, # number of expert demo episodes
    'demo_batch_size': 0,#128, #number of samples to be used from the demonstrations buffer, per mpi thread 128/1024 or 32/256
    'prm_loss_weight': 0.001, #Weight corresponding to the primary loss
    'aux_loss_weight':  0.0078, #Weight corresponding to the auxilliary loss also called the cloning loss
    # ddpg/replay_buffer
    'buffer_size': int(1E6),  # for experience replay
    'batch_size': 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.

    # training
    'n_cycles': 50,  # per epoch
    # 'rollout_batch_size': 2,  # per mpi thread  # this will be overwritten by num_env in her
    'n_batches': 40,  # training batches per cycle
    'n_test_rollouts': 10,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    've_n_batches': 40,
    'update_goal_sampler_interval': 1,  # rollout_batch_size * num_cpu * update_goal_sampler * interval goals will be sampled for one distribution
    # epoch == 0: ve trains for ve_n_batches; later when epoch % interval == 0, ve trains for ve_n_batches * interval batches

    # rollout worker
    'test_with_polyak': False,  # run test episodes with the target network
    'random_eps': 0.3,  # percentage of time a random action is taken
    'noise_eps': 0.2,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u

    # ddpg/HER
    'replay_strategy': 'future',  # supported modes: future, none
    'replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future

    # value ensemble
    'size_ensemble': 3,
    've_buffer_size': int(1E6),  # should at least hold n_cycles * batch_rollout_size * max_timesteps * update_goal_sampler_interval = 50 * 2 * 50 * 5
    've_lr': 0.001,
    've_batch_size': 256,
    've_use_Q': True,
    've_use_double_network': True,  # always use double Q

    # ve/HER
    've_replay_strategy': 'none',  # supported modes: future, none
    've_replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future

    # goal and goal sampler
    'presampler_str': 'feasible',
    'presample_size': 1000,
    'n_reused_states': 0,
    'disagreement_str': 'std',  # 'top100' # used for rollout_batch_size * num_cpu * n_cycles * update_goal_sampler_interval = 2 * 2 * 50 * 3 = 600 samples
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
        if False:
            env.current_goal = np.zeros(2)  # TODO: addd this back for ant envs
        CACHED_ENVS[make_env] = env
    return CACHED_ENVS[make_env]


def prepare_ddpg_params(kwargs):
    ddpg_params = dict()

    for name in ['buffer_size', 'hidden', 'layers',
                 'network_class', 'clip_pos_returns',
                 'polyak',
                 'batch_size', 'Q_lr', 'pi_lr',
                 'norm_eps', 'norm_clip', 'max_u',
                 'action_l2', 'clip_obs', 'relative_goals',
                 'bc_loss', 'q_filter', 'num_demo', 'demo_batch_size', 'prm_loss_weight', 'aux_loss_weight',
                 ]:
        ddpg_params[name] = kwargs[name]
        # kwargs['_' + name] = kwargs[name]
        del kwargs[name]

    ddpg_params['clip_return'] = (1. / (1. - kwargs['gamma'])) if kwargs['clip_return'] else np.inf
    del kwargs['clip_return']

    kwargs['ddpg_params'] = ddpg_params

def prepare_ve_params(kwargs):
    # policy params
    prepare_ddpg_params(kwargs)
    ddpg_params = kwargs['ddpg_params']

    # value ensemble params
    ve_params = dict()

    for name in ['size_ensemble']:
        ve_params[name] = kwargs[name]
        # kwargs['_' + name] = kwargs[name]
        del kwargs[name]

    for name in ['buffer_size', 'lr', 'batch_size', 'use_Q', 'use_double_network']:
        ve_params[name] = kwargs[f've_{name}']
        # kwargs[f'_ve_{name}'] = kwargs[f've_{name}']
        del kwargs[f've_{name}']

    # following ddpg params
    for name in ['hidden', 'layers', 'norm_eps', 'norm_clip', 'polyak',
                 'max_u', 'clip_obs', 'relative_goals', 'clip_return', 'clip_pos_returns']:
        ve_params[name] = ddpg_params[name]

    kwargs['ve_params'] = ve_params


def log_params(params, logger=logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))


def configure_ve_her(params):
    env = cached_make_env(params['make_env'])
    env.reset(reset_goal=False)

    def reward_fun(ag_2, g, info):  # vectorized
        return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)

    # Prepare configuration for HER.
    ddpg_her_params = {
        'reward_fun': reward_fun,
    }
    for name in ['replay_strategy', 'replay_k']:
        ddpg_her_params[name] = params[name]
        # params['_' + name] = ddpg_her_params[name]
        del params[name]
    params['ddpg_her_params'] = ddpg_her_params
    ddpg_sample_transitions = make_sample_her_transitions(**ddpg_her_params)

    ve_her_params = {
        'reward_fun': reward_fun,
    }
    for name in ['replay_strategy', 'replay_k']:
        ve_her_params[name] = params[f've_{name}']
        # params[f'_ve_{name}'] = ve_her_params[name]
        del params[f've_{name}']
    params['ve_her_params'] = ve_her_params
    ve_sample_transitions = make_sample_her_transitions(**ve_her_params)

    return ddpg_sample_transitions, ve_sample_transitions


def simple_goal_subtract(a, b):
    assert a.shape == b.shape
    return a - b


def prepare_rollout_worker_params(params):
    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'compute_Q': False,
        'T': params['T'],
        'rollout_batch_size': params['rollout_batch_size'],
    }
    eval_params = {
        'exploit': True,
        'compute_Q': True,
        'T': params['T'],
        'rollout_batch_size': params['rollout_batch_size'],
        'use_target_net': params['test_with_polyak'],
    }
    plotter_params = {
        'exploit': True,
        'compute_Q': False,
        'use_target_net': False,
        'T': params['T'],
        'history_len': 10000,
        'rollout_batch_size': 1,
    }

    del params['test_with_polyak']

    for name in ['noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        # eval_params[name] = params[name]  # will be 0
        del params[name]

    params['rollout_params'] = rollout_params
    params['eval_params'] = eval_params
    params['plotter_params'] = plotter_params

#
# def configure_plotter(policy, value_ensemble, plotter_worker, params, report):
#     env = cached_make_env(params['make_env'])
#     goals, plotter_info = env.get_grid_goals(sampling_res=2, feasible=True)  # use 0 for Point-v0
#
#     # TODO: plot all goals here
#
#     plot_heatmap_fun = functools.partial(plot_heatmap, show_heatmap=False, **plotter_info)
#
#     return make_plotter(
#         init_ob=env.init_ob,
#         policy=policy,
#         value_ensemble=value_ensemble,
#         goals=goals,
#         disagreement_str=params['goal_params']['disagreement_str'],
#         plotter_worker=plotter_worker,
#         gamma=params['gamma'],
#         report=report,
#         plot_heatmap_fun=plot_heatmap_fun,
#     )
#
#     # env = cached_make_env(params['make_env'])
#     # goals, info = find_maze_empty_space(env, sampling_res=2)
#     # plot_heatmap_fun = functools.partial(plot_maze_heatmap, show_heatmap=False,
#     #                                      maze_id=info['maze_id'],
#     #                                      center=params['goal_params']['goal_center'],
#     #                                      limit=params['goal_params']['goal_range'])
#     #
#     # return make_plotter(
#     #     init_ob=env.init_ob,
#     #     policy=policy,
#     #     value_ensemble=value_ensemble,
#     #     goals=goals,
#     #     spacing=info['spacing'],
#     #     disagreement_str=params['goal_params']['disagreement_str'],
#     #     plotter_worker=plotter_worker,
#     #     gamma=params['gamma'],
#     #     report=report,
#     #     plot_heatmap_fun=plot_heatmap_fun,
#     # )


def configure_dims(params):
    env = cached_make_env(params['make_env'])
    # try:
    env.reset(reset_goal=False)
    # except NotImplementedError:
    #     env.get_reset_obs()
    #     env.reset_goal(goal=env.sample_goals(1)[0])

    a = env.action_space.sample()
    ob, _, _, info = env.step(action=a)

    dims = {
        'o': ob['observation'].size,
        'u': a.size,
        'g': ob['desired_goal'].size,
    }
    for key, value in info.items():
        # value = np.array(value)
        # if value.ndim == 0:
            # value = value.reshape(1)
        dims['info_{}'.format(key)] = np.asarray(value).size
    return dims


def configure_goal_presampler(kwargs):
    goal_params = dict()
    for name in ['goal_center', 'goal_range', 'presample_size', 'disagreement_str', 'presampler_str', 'n_reused_states']:
        goal_params[name] = kwargs[name]
        # kwargs['_' + name] = kwargs[name]
        del kwargs[name]
    kwargs['goal_params'] = goal_params

    if goal_params['presampler_str'] == 'uniform':
        goal_presampler, info = make_uniform_goal_presampler(state_center=goal_params['goal_center'], state_range=goal_params['goal_range'])
    elif goal_params['presampler_str'] == 'normal':
        goal_presampler, info = make_normal_goal_presampler(state_center=goal_params['goal_center'], state_range=goal_params['goal_range'])
    elif goal_params['presampler_str'] == 'feasible':
        env = cached_make_env(kwargs['make_env'])
        goal_presampler, info = make_grid_goal_presampler(env=env, sampling_res=3, uniform_noise=True, feasible=True)
    else:
        raise NotImplementedError

    logger.info(f'configure goal presampler, number of states = {len(info["all_states"])}')

    # TODO: plot all goals here

    return goal_presampler


def configure_all(dims, params, reuse=False, policy_pkl=None):
    env = cached_make_env(params['make_env'])
    env.reset(reset_goal=False)#get_reset_obs()

    params['T'] = env.spec.max_episode_steps
    params['gamma'] = 1. - 1. / params['T']
    params['max_u'] = env.action_space.high
    # params['goal_range'] = env.goal_range
    # params['goal_center'] = env.goal_center

    # Extract relevant parameters.
    prepare_ve_params(params)
    ddpg_sample_transitions, ve_sample_transitions = configure_ve_her(params)

    # DDPG agent
    if policy_pkl is not None:
        # load froze policy
        import joblib
        logger.info('loading policy...')
        data = joblib.load(policy_pkl)
        policy = data['policy']

    else:
        policy = DDPG(
            reuse=reuse,
            input_dims=dims.copy(),
            scope='ddpg',
            T=params['T'],
            gamma=params['gamma'],
            rollout_batch_size=params['rollout_batch_size'],
            sample_transitions=ddpg_sample_transitions,
            subtract_goals=simple_goal_subtract,
            **params['ddpg_params'],
        )

    value_ensemble = ValueEnsemble(
        reuse=reuse,
        input_dims=dims.copy(),
        scope='ve' if policy_pkl is None else 've-trainable',
        T=params['T'],
        gamma=params['gamma'],
        rollout_batch_size=params['rollout_batch_size'],
        sample_transitions=ve_sample_transitions,
        subtract_goals=simple_goal_subtract,
        **params['ve_params']
    )

    if False:
        goal_presampler = configure_goal_presampler(params)
        goal_params = params['goal_params']
        goal_sampler_factory = make_goal_sampler_factory(
            init_ob=env.init_ob, goal_presampler=goal_presampler,
            value_ensemble=value_ensemble, policy=policy,
            presample_size=goal_params['presample_size'],
            disagreement_str=goal_params['disagreement_str'],
            n_reused_states=goal_params['n_reused_states'],
        )

        # for evaluation: sample from grid intersections with uniform probability
        # number of grids determined by sampling_res
        feasible_grid_goal_presampler, _ = make_grid_goal_presampler(env=env, sampling_res=3, uniform_noise=False, feasible=True)
        feasible_uniform_grid_goal_sampler, _ = make_uniform_goal_sampler(feasible_grid_goal_presampler)
        # TODO: plot all goals here
        return policy, value_ensemble, goal_sampler_factory, feasible_uniform_grid_goal_sampler


    # goal sampling function to be passed in vector env
    from baselines.her.experiment.config import configure_disagreement
    params['gs_params'] = dict(n_candidates=params['presample_size'], disagreement_fun_name=params['disagreement_str'])
    print(params['disagreement_str'])
    sample_disagreement_goals_fun, sample_uniform_goals_fun = configure_disagreement(
        params,
        value_ensemble=value_ensemble,
        policy=policy
    )
    return policy, value_ensemble, sample_disagreement_goals_fun, sample_uniform_goals_fun

