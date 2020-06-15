import json
import sys
import os
from mpi4py import MPI  # TODO: import here or in main?
from baselines import logger
from baselines.run import parse_cmdline_kwargs, configure_logger
from baselines.her.vds import learn

# from baselines.envs.goal_env_wrapper import GoalExplorationEnv
from baselines.common.mpi_fork import mpi_fork_run_as_root, mpi_fork
import baselines.common.tf_util as U
import gym
from baselines.run_util import init_arg_parser, make_vec_env
from baselines.run_util import set_global_seeds
import baselines.her.experiment.vds_config as config


def main(args):
    # print("\n\n\n\n\nXXX")
    # print(sys.path)
    # import baselines
    # print(baselines.__file__())
    # for varname in ['PMI_RANK', 'OMPI_COMM_WORLD_RANK']:
    #     if varname in os.environ:
    #         print(varname, int(os.environ[varname]))
    # print("parsing args...")

    arg_parser = init_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)

    # if args.num_cpu > 1:
    if args.allow_run_as_root:
        whoami = mpi_fork_run_as_root(args.num_cpu, bind_to_core=args.bind_to_core)
    else:
        whoami = mpi_fork(args.num_cpu, bind_to_core=args.bind_to_core)
    if whoami == 'parent':
        print('parent exiting with code 0...')
        sys.exit(0)

    U.single_threaded_session().__enter__()

    rank = MPI.COMM_WORLD.Get_rank()

    # assert MPI.COMM_WORLD.Get_size() == args.num_cpu, MPI.COMM_WORLD.Get_size()

    # configure logger
    # rank = MPI.COMM_WORLD.Get_rank()  # FIXME: how to log when rank != 0??
    # if rank == 0:
    configure_logger(args.log_path, format_strs=[])
    logger.info(f"main: {rank} / {MPI.COMM_WORLD.Get_size()}")
    logger.info(f"logger dir: {logger.get_dir()}")

    extra_args = parse_cmdline_kwargs(unknown_args)
    logger.info(args, extra_args)

    # else:
    #     configure_logger(log_path=None)  # or still args.log_path?

    # raise RuntimeError(f"tf session: {tf.get_default_session()}, {MPI.COMM_WORLD.Get_rank()} / {MPI.COMM_WORLD.Get_size()}")

    def make_wrapped_env():
        env = gym.make(args.env)
        if args.env_type == 'maze':
            pass
        elif args.env_type == 'robotics':
            from baselines.envs.goal_sampler_env_wrapper import GoalSamplerEnvWrapper
            env = GoalSamplerEnvWrapper(env)
        elif args.env_type == 'ant':
            env = GoalExplorationEnv(env=env, only_feasible=True, extend_dist_rew=0, inner_weight=0, goal_weight=1)
        else:
            raise NotImplementedError(args.env_type)
        # FIXME: if resample space is feasible, can set only_feasible = False to avoid unnecessary computation
        return env

    venv_kwargs = dict(
        make_wrapped_env=make_wrapped_env,
        seed=args.seed,
        reward_scale=args.reward_scale,
        flatten_dict_observations=False,
        mpi_rank=rank,
        monitor_log_dir=args.log_path,  # FIXME
    )
    venv = make_vec_env(num_env=args.num_env, **venv_kwargs)
    eval_venv = make_vec_env(num_env=args.num_env, **venv_kwargs)
    if args.debug:
        plotter_venv = make_vec_env(num_env=1, **venv_kwargs)
    else:
        plotter_venv = None

    # Seed everything.
    rank_seed = args.seed + 1000000 * rank if args.seed is not None else None
    set_global_seeds(rank_seed)
    logger.info(f'setting global rank: {rank_seed} ')

    # Prepare params.
    params = dict()
    params.update(config.DEFAULT_PARAMS)
    params.update(config.DEFAULT_ENV_PARAMS[args.env])
    params.update(**extra_args)  # makes it possible to override any parameter

    # if args.debug:
    #     params['n_cycles'] = 2
    #     params['n_batches'] = 2
    #     params['ve_n_batches'] = 2
    #     params['size_ensemble'] = 2

    # env settings
    params['env_name'] = args.env
    params['num_cpu'] = args.num_cpu
    params['rollout_batch_size'] = args.num_env
    params['timesteps_per_cpu'] = int(args.num_timesteps)

    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)

    params['make_env'] = make_wrapped_env

    learn_fun_return = learn(
        venv=venv,
        eval_venv=eval_venv,
        plotter_venv=plotter_venv,
        params=params,
        save_path=args.log_path,
        save_interval=args.save_interval,
    )

    if rank == 0:
        save_path = os.path.expanduser(logger.get_dir())
        for k, v in learn_fun_return.items():
            v.save(os.path.join(save_path, f"final-{k}.joblib"))

    venv.close()
    eval_venv.close()
    if plotter_venv is not None:
        plotter_venv.close()


if __name__ == '__main__':
    main(sys.argv)
