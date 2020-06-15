import copy

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from baselines.envs.maze.maze_layouts import maze_layouts

"""
Maze environments are adapted from
https://github.com/kzl/aop/tree/master/envs/ContinualParticleMaze.py
"""


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class ParticleMazeEnv(gym.GoalEnv):
    def __init__(self, grid_name, reward_type='sparse'):
        self.seed()
        self._init_grid(grid_name)

        self.reward_type = reward_type
        self.dt = 0.1
        self.num_collision_steps = 10
        self.distance_threshold = 0.1

        self.goal = self._sample_goal()
        obs = self._get_obs()

        self.action_space = spaces.Box(-1., 1., shape=(2,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

        self._init_obs = copy.deepcopy(obs)
        self.sample_goal_fun = None

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = np.linalg.norm(achieved_goal - goal, axis=-1)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def update_goal_sampler(self, goal_sampler):
        self.sample_goal_fun = goal_sampler

    def reset(self, *, reset_goal=True):
        self._reset_agent()
        self.goal = None
        obs = self._get_obs()
        if reset_goal:
            self.goal = self.sample_goal_fun(obs_dict=obs)
        else:
            self.goal = self._sample_goal()
        obs = self._get_obs()
        return obs

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _init_grid(self, grid_name):
        self.grid = maze_layouts[grid_name]
        self.grid = self.grid.replace('\n', '')
        self.grid_size = grid_size = int(np.sqrt(len(self.grid)))

        self.grid_chars = (np.array(list(self.grid)) != 'S').reshape((grid_size, grid_size))
        self.start_ind, = np.argwhere(self.grid_chars == False)
        self.grid = self.grid.replace('S', ' ')

        self.grid = (np.array(list(self.grid)) != ' ').reshape((grid_size, grid_size))
        self.grid_wall_index = np.argwhere(self.grid == True)
        self.grid_free_index = np.argwhere(self.grid != True)

        self._reset_agent()

    def step(self, action):
        ind = self.get_index(self.x)
        assert not self.grid[ind[0], ind[1]], self.x

        action = np.clip(action, self.action_space.low, self.action_space.high)
        ddt = self.dt / self.num_collision_steps

        for _ in range(self.num_collision_steps):
            x_new = self.x + action * ddt
            ind = self.get_index(x_new)

            if self.grid[ind[0], ind[1]]:
                # bounce back
                # self.x -= action * ddt
                break
            else:
                self.x = x_new

        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def _reset_agent(self):
        self.x = self.get_coords(self.start_ind)

    def _get_obs(self):
        return dict(
            desired_goal=self.goal,
            achieved_goal=self.x,
            observation=self.x
        )

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _sample_goal(self):
        goal_ind = self.grid_free_index[np.random.choice(len(self.grid_free_index))]
        return (goal_ind + self.np_random.uniform(low=0, high=1, size=2)) / self.grid_size * 2 - 1

    def sample_goals(self, num_samples):
        goal_ind = self.grid_free_index[np.random.choice(len(self.grid_free_index), num_samples)]
        return (goal_ind + self.np_random.uniform(low=0, high=1, size=(num_samples, 2))) / self.grid_size * 2 - 1

    def get_coords(self, ind):
        return (ind + 0.5) / self.grid_size * 2 - 1

    def get_index(self, coords):
        return np.clip((coords + 1) * 0.5 * self.grid_size + 0, 0, self.grid_size-1).astype(np.int)

    def close(self):
        pass

