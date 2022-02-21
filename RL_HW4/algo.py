import random
from collections import defaultdict

import numpy as np

from abc import abstractmethod
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class QAgent:
    def __init__(self, ):
        pass

    @abstractmethod
    def select_action(self, ob):
        pass

class MyQAgent(QAgent):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.learningRate = alpha
        self.discountFactor = gamma
        self.qTable = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    def select_action(self, ob):
        stateQ = self.qTable[str(ob)]
        max = []
        maxValue = stateQ[0]
        max.append(0)
        for i in range(1, 4):
            if stateQ[i] > maxValue:
                max.clear()
                maxValue = stateQ[i]
                max.append(i)
            elif stateQ[i] == maxValue:
                max.append(i)
        return random.choice(max)

    def learn(self, ob, action, reward, nextOb):
        oldQ = self.qTable[str(ob)][action]
        newQ = reward + self.discountFactor * max(self.qTable[str(nextOb)])
        self.qTable[str(ob)][action] += self.learningRate * (newQ - oldQ)
        # self.qTable[str(ob)][action] = np.clip(self.qTable[str(ob)][action], -100, 100)


class Model:
    def __init__(self, width, height, policy):
        self.width = width
        self.height = height
        self.policy = policy
        pass

    @abstractmethod
    def store_transition(self, s, a, r, s_):
        pass

    @abstractmethod
    def sample_state(self):
        pass

    @abstractmethod
    def sample_action(self, s):
        pass

    @abstractmethod
    def predict(self, s, a):
        pass

class DynaModel(Model):
    def __init__(self, width, height, policy):
        Model.__init__(self, width, height, policy)
        self.buffer = []
        self.qTable = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    def store_transition(self, s, a, r, s_):
        self.buffer.append([s, a, r, s_])

    def sample_state(self):
        idx = np.random.randint(0, len(self.buffer))
        return self.buffer[idx][0], idx

    def sample_action(self, s):
        return self.policy.select_action(s)

    def predict(self, s, a):
        max = 0.0
        s_list= [self.sample_state()[0]]
        for item in self.buffer:
            if str(item[0]) == str(s) and item[1] == a:
                if item[2] > max:
                    s_list.clear()
                    s_list.append(item[3])
                    max = item[2]
                elif item[2] == max:
                    s_list.append(item[3])
        return random.choice(s_list)

    def train_transition(self):
        s, a, r, s_ = random.choice(self.buffer)
        oldQ = self.qTable[str(s)][a]
        newQ = r + self.policy.discountFactor * max(self.qTable[str(s_)])
        self.qTable[str(s)][a] += self.policy.learningRate * (newQ - oldQ)


class NetworkModel(Model):
    def __init__(self, width, height, policy):
        Model.__init__(self, width, height, policy)
        self.x_ph = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='x')
        self.x_next_ph = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='x_next')
        self.a_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='a')
        self.r_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='r')
        h1 = tf.layers.dense(tf.concat([self.x_ph, self.a_ph], axis=-1), units=256, activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, units=256, activation=tf.nn.relu)
        self.next_x = tf.layers.dense(h2, units=3, activation=tf.nn.tanh) * 1.3 + self.x_ph
        self.x_mse = tf.reduce_mean(tf.square(self.next_x - self.x_next_ph))
        self.opt_x = tf.train.RMSPropOptimizer(learning_rate=1e-5).minimize(self.x_mse)
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.variables_initializer(tf.global_variables()))
        self.buffer = []
        self.sensitive_index = []

    def norm_s(self, s):
        return s

    def de_norm_s(self, s):
        s = np.clip(np.round(s), 0, self.width - 1).astype(np.int32)
        s[2] = np.clip(s[2], 0, 1).astype(np.int32)
        return s

    def store_transition(self, s, a, r, s_):
        s = self.norm_s(s)
        s_ = self.norm_s(s_)
        self.buffer.append([s, a, r, s_])
        if s[-1] - s_[-1] != 0:
            self.sensitive_index.append(len(self.buffer) - 1)

    def train_transition(self, batch_size):
        s_list = []
        a_list = []
        r_list = []
        s_next_list = []
        for _ in range(batch_size):
            idx = np.random.randint(0, len(self.buffer))
            s, a, r, s_ = self.buffer[idx]
            s_list.append(s)
            a_list.append([a])
            r_list.append(r)
            s_next_list.append(s_)

        '''
        if len(self.sensitive_index) > 0:
            for _ in range(batch_size):
                idx = np.random.randint(0, len(self.sensitive_index))
                idx = self.sensitive_index[idx]
                s, a, r, s_ = self.buffer[idx]
                s_list.append(s)
                a_list.append([a])
                r_list.append(r)
                s_next_list.append(s_)
        '''
        x_mse = self.sess.run([self.x_mse,  self.opt_x], feed_dict={
            self.x_ph: s_list, self.a_ph: a_list, self.x_next_ph: s_next_list
        })[:1]
        return x_mse

    def sample_state(self):
        idx = np.random.randint(0, len(self.buffer))
        s, a, r, s_ = self.buffer[idx]
        return self.de_norm_s(s), idx

    def sample_action(self, s):
        return self.policy.select_action(s)

    def predict(self, s, a):
        s_ = self.sess.run(self.next_x, feed_dict={self.x_ph: [s], self.a_ph: [[a]]})
        return self.de_norm_s(s_[0])
