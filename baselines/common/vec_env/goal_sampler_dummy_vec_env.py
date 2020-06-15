import numpy as np
from .vec_env import VecEnv
from .util import copy_obs_dict, dict_to_obs, obs_space_info

class GoalSamplerDummyVecEnv(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """
    def __init__(self, env_fns, save_samples=False):
        """
        Arguments:

        env_fns: iterable of callables      functions that build environments
        """
        self.envs = [fn() for fn in env_fns]
        self.sample_goals_fun = None  # to be set later
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = { k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys }
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews  = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.spec = self.envs[0].spec

        self.save_samples = save_samples
        self.samples_history = []

        # # to save goal samples into pickle files
        # if save_path:
        #     self._save_path = os.path.join(save_path, "samples_itr_{}.pkl")
        # else:
        #     self._save_path = None
        # self._reset_counter = 0

    def set_sample_goals_fun(self, sample_goals_fun):
        self.sample_goals_fun = sample_goals_fun

    def step_async(self, actions):
        listify = True
        try:
            if len(actions) == self.num_envs:
                listify = False
        except TypeError:
            pass

        if not listify:
            self.actions = actions
        else:
            assert self.num_envs == 1, "actions {} is either not a list or has a wrong size - cannot match to {} environments".format(actions, self.num_envs)
            self.actions = [actions]

    def step_wait(self):
        for e in range(self.num_envs):
            action = self.actions[e]
            obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[e].step(action)
            if self.buf_dones[e]:
                obs = self.envs[e].get_reset_obs()  # now desired_goal is set to None
            self._save_obs(e, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
                self.buf_infos.copy())

    def _get_reset_obs(self):
        # must be called before reset_goal
        for e in range(self.num_envs):
            obs = self.envs[e].get_reset_obs()
            self._save_obs(e, obs, ignore_keys=('desired_goal'))
        return self._obs_from_buf()

    def _reset_goal(self, goals):
        # must be called after get_reset_obs
        for e in range(self.num_envs):
            obs = self.envs[e].reset_goal(goal=goals[e])
            self._save_obs(e, obs)
        return self._obs_from_buf()

    def reset(self):
        # does not call self.envs[e].reset()
        # sample goals with goal_sampler instead
        obs_dict = self._get_reset_obs()
        # save_path = self._save_path.format(self._reset_counter) if self._save_path is not None else None
        # self._reset_counter += 1
        # goals = self.sample_goals_fun(obs_dict, save_path=save_path)
        goals = self.sample_goals_fun(obs_dict, save_path=None)
        if self.save_samples:
            self.samples_history.append(goals)
        return self._reset_goal(goals)

    def _save_obs(self, e, obs, ignore_keys=()):
        for k in self.keys:
            if k is None:
                self.buf_obs[k][e] = obs
            else:
                if k not in ignore_keys:
                    self.buf_obs[k][e] = obs[k]

    def _obs_from_buf(self):
        return dict_to_obs(copy_obs_dict(self.buf_obs))

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, mode='human'):
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)

    def reset_history(self):
        if self.save_samples:
            samples = np.concatenate(self.samples_history, axis=0)
            self.samples_history = []
            return samples
