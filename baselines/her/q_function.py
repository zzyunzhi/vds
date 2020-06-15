import tensorflow as tf
from baselines.her.util import store_args, nn


class VFunction:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), the action (u),
                the next observation (o_2), the next goal (g_2)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.o_2_tf = inputs_tf['o_2']
        self.g_2_tf = inputs_tf['g_2']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        o_2 = self.o_stats.normalize(self.o_2_tf)
        g_2 = self.g_stats.normalize(self.g_2_tf)

        # Networks.
        with tf.variable_scope('V'):
            input_V = tf.concat(axis=1, values=[o, g])
            self.V_tf = nn(input_V, [self.hidden] * self.layers + [1])
            input_V_2 = tf.concat(axis=1, values=[o_2, g_2])
            self.V_2_tf = nn(input_V_2, [self.hidden] * self.layers + [1], reuse=True)


class QFunction:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), the action (u),
                the next observation (o_2), the next goal (g_2), the next action (u_2)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']
        self.o_2_tf = inputs_tf['o_2']
        self.g_2_tf = inputs_tf['g_2']
        self.u_2_tf = inputs_tf['u_2']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        o_2 = self.o_stats.normalize(self.o_2_tf)
        g_2 = self.g_stats.normalize(self.g_2_tf)

        # Networks.
        with tf.variable_scope('V'):
            input_V = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            self.V_tf = nn(input_V, [self.hidden] * self.layers + [1])
            input_V_2 = tf.concat(axis=1, values=[o_2, g_2, self.u_2_tf / self.max_u])
            self.V_2_tf = nn(input_V_2, [self.hidden] * self.layers + [1], reuse=True)


class DoubleQFunction:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), the action (u),
                the next observation (o_2), the next goal (g_2), the next action (u_2)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)

        # Networks.
        with tf.variable_scope('V'):
            input_V = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            self.V_tf = nn(input_V, [self.hidden] * self.layers + [1])

