"""
Note: This is the proposed actor-critic-based DRL algorithm, named Deep Stochastic Online Scheduling (DSOS).
The DSOS is based on the actor-critic framework with two Deep neural networks (DNNs) as the approximators.
The Actor applies stochastic policy.
We assume the stochastic policy follows Guassian distribution.
The Critic applies temporal difference (TD) learning.
"""

import tensorflow as tf

GAMMA = 0.9
class Actor(object):
    def __init__(self, n_features, n_actions, lr):
        self.sess = tf.Session()
        self.n_actions = n_actions
        self.s = tf.placeholder(tf.float32, [None, n_features], name="state")
        self.a = tf.placeholder(tf.float32, [None, n_actions], name="act")
        self.td_error = tf.placeholder(tf.float32, None, name="td_error")

        with tf.variable_scope('get_v'):
            normal_dist, self.action, _ = self.build_a(self.s, trainable=True, reuse=False)
        with tf.variable_scope('get_exp_v'):
            log_prob = normal_dist.log_prob(self.a)  # loss without advantage
            self.exp_v = log_prob * self.td_error  # advantage (TD_error) guided loss
            # self.exp_v += 0.01 * self.normal_dist.entropy()  # Add cross entropy cost to encourage exploration
        with tf.variable_scope('train_a'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)    # min(v) = max(-v)
        self.sess.run(tf.global_variables_initializer())

    def build_a(self, s, trainable, reuse):
        with tf.variable_scope('NN_actor', reuse=reuse):
            l1 = tf.layers.dense(inputs=s, units=500, activation=tf.nn.tanh,
                bias_initializer=tf.constant_initializer(0),
                trainable=trainable,
                name='l1')
            l2 = tf.layers.dense(inputs=l1, units=300, activation=tf.nn.tanh,
                bias_initializer=tf.constant_initializer(0),
                trainable=trainable,
                name='l2')
            mu = tf.layers.dense(inputs=l2, units=self.n_actions, activation=tf.nn.tanh,
                bias_initializer=tf.constant_initializer(0),
                trainable=trainable,
                name='mu')
            sigma = tf.layers.dense(inputs=l2, units=self.n_actions, activation=tf.nn.sigmoid,
                bias_initializer=tf.constant_initializer(0),
                trainable=trainable,
                name='sigma')

            normal_dist = tf.distributions.Normal(mu, sigma)
            action_inf = normal_dist.sample(1)[0][0]
            action = tf.clip_by_value(action_inf, -2, 2)[0]
            params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='NN_actor')
            self.saver = tf.train.Saver()
            return normal_dist, action, params

    def choose_action(self, s):
        return self.sess.run(self.action, {self.s: s})

    def learn(self, s, a, td):
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        self.sess.run(self.train_op, feed_dict)

class Critic(object):
    def __init__(self, n_features, lr):
        self.sess = tf.Session()
        self.s = tf.placeholder(tf.float32, [None, n_features], name="state_c")
        self.s_ = tf.placeholder(tf.float32, [None, n_features], name="state_c_next")
        self.r = tf.placeholder(tf.float32, None, name='r')
        with tf.variable_scope('get_v'):
            v, _ = self.build_c(self.s, trainable=True, reuse=False)
            v_, _ = self.build_c(self.s_, trainable=True, reuse=True)
        with tf.variable_scope('squared_TD_error'):
            self.td_error = tf.reduce_mean(self.r + GAMMA * v_ - v)
            self.loss = tf.square(self.td_error)
        with tf.variable_scope('train_c'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

    def build_c(self, s, trainable, reuse):
        with tf.variable_scope('NN_critic', reuse=reuse):
            l1 = tf.layers.dense(inputs=s, units=500, activation=tf.nn.relu,
                bias_initializer=tf.constant_initializer(0),
                trainable=trainable,
                name='l1_c')
            l2 = tf.layers.dense(inputs=l1, units=300, activation=tf.nn.relu,
                bias_initializer=tf.constant_initializer(0),
                trainable=trainable,
                name='l2_c')
            v = tf.layers.dense(inputs=l2, units=1, activation=None,
                trainable=trainable,
                name='V_c')

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='NN_critic')
        self.saver = tf.train.Saver()
        return v, params

    def get_td_error(self, s, r, s_):
        feed_dict = {self.s: s, self.r: r, self.s_: s_}
        td_error = self.sess.run(self.td_error, feed_dict)
        return td_error

    def learn(self, s, r, s_):
        feed_dict = {self.s: s, self.r: r, self.s_: s_}
        self.sess.run(self.train_op, feed_dict)
