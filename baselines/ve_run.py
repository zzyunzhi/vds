import sys
import os.path as osp
import tensorflow as tf

from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module

from mpi4py import MPI

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None


def train(args, extra_args):
    env_id = args.env
    env_type = args.env_type

    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)
    eval_env = build_env(args)
    plotter_env = build_env(args) if args.debug else None

    logger.info('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    from baselines.her.ve_her import learn
    policy, value_ensemble = learn(
        env_type=env_type,
        env=env,
        eval_env=eval_env,
        plotter_env=plotter_env,
        num_cpu=args.num_cpu,
        allow_run_as_root=args.allow_run_as_root,
        bind_to_core=args.bind_to_core,
        save_path=args.log_path,
        seed=args.seed,
        total_timesteps=int(args.num_timesteps),
        policy_pkl=None,#args.policy_pkl,
        save_interval=args.save_interval,
        override_params=alg_kwargs,
    )

    return policy, value_ensemble, env


def build_env(args):
    # env_id = ENV_NAME_TO_REGISTRY.get(args.env, "GoalSampling" + args.env)
    env_id = args.env
    env_type = args.env_type

    config = tf.ConfigProto(allow_soft_placement=True,
                           intra_op_parallelism_threads=1,
                           inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    get_session(config=config)

    # flatten_dict_observations = args.alg not in {'her'}
    env = make_vec_env(env_id, env_type, args.num_env or 1, args.seed,
                       reward_scale=args.reward_scale, flatten_dict_observations=False,
                       force_dummy=args.force_dummy)
    return env


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    logger.info(args, extra_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])

    policy, value_ensemble, env = train(args, extra_args)

    if args.log_path is not None and rank == 0:
        save_path = osp.expanduser(args.log_path)
        policy.save(osp.join(save_path, 'final_policy_params.joblib'))
        value_ensemble.save(osp.join(save_path, 'final_ve_params.joblib'))

    env.close()


if __name__ == '__main__':
    main(sys.argv)
