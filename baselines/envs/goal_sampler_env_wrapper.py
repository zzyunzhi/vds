from gym import Wrapper
import numpy as np


class GoalSamplerEnvWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.sample_goal_fun = None

    def update_goal_sampler(self, goal_sampler):
        self.sample_goal_fun = goal_sampler

    def reset(self, *, reset_goal=True):
        self.env._elapsed_steps = 0  # this is a hack for gym.wrappers.time_limit
        # below: self.unwrapped.reset() (gym.RobotEnv)
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self.unwrapped._reset_sim()
        if reset_goal:
            obs = self.unwrapped._get_obs()
            self.unwrapped.goal = self.sample_goal_fun(obs_dict=obs)
        else:
            self.unwrapped.goal = self.unwrapped._sample_goal().copy()
        obs = self.unwrapped._get_obs()
        return obs

    def get_grid_goals(self, sampling_res, feasible=True):
        """ This function is NOT used. """
        # FIXME: only for FetchEnv
        points_per_dim = 2**sampling_res
        spacing = self.env.target_range * 2 / points_per_dim

        # if self.has_object:
        #     raise NotImplementedError
        #     goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        #     goal += self.target_offset
        #     goal[2] = self.height_offset
        #     if self.target_in_the_air and self.np_random.uniform() < 0.5:
        #         goal[2] += self.np_random.uniform(0, 0.45)
        # else:
        #     goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)

        # return goal.copy()

        if self.env.has_object:
            target_center = self.env.initial_gripper_xpos[:3] + self.env.target_offset
        else:
            target_center = self.env.initial_gripper_xpos[:3]
        target_center_2d_coords = np.asarray(target_center)[:2]

        pos_lim_low = target_center_2d_coords - self.env.target_range
        pos_lim_high = target_center_2d_coords + self.env.target_range

        # if self.has_object:
        #     self.zz_candidates = self.env.height_offset + np.linspace(0, 0.45, z_discretization)
        # else:
        #     self.zz_candidates = target_center[2] + np.linspace(-self.env.target_range, self.env.target_range, z_discretization)
        #
        # self.is_feasible = 1

        x = np.linspace(pos_lim_low[0], pos_lim_high[0], num=points_per_dim, endpoint=False) + spacing / 2
        y = np.linspace(pos_lim_low[1], pos_lim_high[1], num=points_per_dim, endpoint=False) + spacing / 2
        xx, yy = np.meshgrid(x, y)
        # zz = self.zz_candidates[z_idx] * np.ones(points_per_dim*points_per_dim)
        zz = 0 * np.ones(points_per_dim*points_per_dim)
        # candidates = np.asarray(list(zip(xx.ravel(), yy.ravel(), np.ones(n_candidates) * target_center_height)))
        candidates = np.asarray(list(zip(xx.ravel(), yy.ravel(), zz)))
        assert len(candidates) == points_per_dim*points_per_dim
        plotter_info = dict(
            limit=(self.env.target_range, self.env.target_range),
            center=target_center_2d_coords,
            spacing=(spacing, spacing),
        )
        return candidates, plotter_info