import numpy as np
import tensorflow as tf
import math
from model_base import Model

class ModelMdn(Model):
    def __init__(self):
        NHIDDEN = 24
        STDEV = 0.5
        self.KMIX = 24  # number of mixtures
        NOUT = self.KMIX * 3  # pi, mu, stdev
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="x")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="y")
        Wh = tf.Variable(tf.random_normal([1, NHIDDEN], stddev=STDEV, dtype=tf.float32))
        bh = tf.Variable(tf.random_normal([1, NHIDDEN], stddev=STDEV, dtype=tf.float32))
        Wo = tf.Variable(tf.random_normal([NHIDDEN, NOUT], stddev=STDEV, dtype=tf.float32))
        bo = tf.Variable(tf.random_normal([1, NOUT], stddev=STDEV, dtype=tf.float32))
        hidden_layer = tf.nn.tanh(tf.matmul(self.x, Wh) + bh)
        self.output = tf.matmul(hidden_layer, Wo) + bo
        out_pi, out_sigma, out_mu = self._get_mixture_coef(self.output)
        self.lossfunc = self._get_lossfunc(out_pi, out_sigma, out_mu, self.y)
        self.train_op = tf.train.AdamOptimizer().minimize(self.lossfunc)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())

    def fit(self, x_, y_):
        self.NEPOCH = 10000
        self.loss = np.zeros(self.NEPOCH)  # store the training progress here.
        for i in range(self.NEPOCH):
            if i % 1000 == 0:
                print(i)
            self.sess.run(self.train_op, feed_dict={self.x: x_, self.y: y_})
            loss = self.sess.run(self.lossfunc, feed_dict={self.x: x_, self.y: y_})
            self.loss[i] = loss


    def predict(self, x_):
        out_pi, out_sigma, out_mu = self.sess.run(self._get_mixture_coef(self.output),feed_dict={self.x: x_})
        return self._generate_ensemble(out_pi, out_mu, out_sigma)

    def _get_mixture_coef(self, output):
        out_pi = tf.placeholder(dtype=tf.float32, shape=[None, self.KMIX], name="mixparam")
        out_sigma = tf.placeholder(dtype=tf.float32, shape=[None, self.KMIX], name="mixparam")
        out_mu = tf.placeholder(dtype=tf.float32, shape=[None, self.KMIX], name="mixparam")
        out_pi, out_sigma, out_mu = tf.split(output, 3, 1)
        max_pi = tf.reduce_max(out_pi, 1, keep_dims=True)
        out_pi = tf.subtract(out_pi, max_pi)
        out_pi = tf.exp(out_pi)
        normalize_pi = tf.reciprocal(tf.reduce_sum(out_pi, 1, keep_dims=True))
        out_pi = tf.multiply(normalize_pi, out_pi)
        out_sigma = tf.exp(out_sigma)
        return out_pi, out_sigma, out_mu

    def _tf_normal(self, y, mu, sigma):
        oneDivSqrtTwoPI = 1 / math.sqrt(2 * math.pi)  # normalisation factor for gaussian, not needed.
        result = tf.subtract(y, mu)
        result = tf.multiply(result, tf.reciprocal(sigma))
        result = -tf.square(result) / 2
        return tf.multiply(tf.exp(result), tf.reciprocal(sigma)) * oneDivSqrtTwoPI

    def _get_lossfunc(self, out_pi, out_sigma, out_mu, y):
        result = self._tf_normal(y, out_mu, out_sigma)
        result = tf.multiply(result, out_pi)
        result = tf.reduce_sum(result, 1, keep_dims=True)
        result = -tf.log(result)
        return tf.reduce_mean(result)
    def _get_pi_idx(self, x, pdf):
        N = pdf.size
        accumulate = 0
        for i in range(0, N):
            accumulate += pdf[i]
            if (accumulate >= x):
                return i
        print('error with sampling ensemble')
        return -1

    def _generate_ensemble(self, out_pi, out_mu, out_sigma, M = 10):
        NTEST = len(out_mu)
        result = np.random.rand(NTEST, M) # initially random [0, 1]
        rn = np.random.randn(NTEST, M) # normal random matrix (0.0, 1.0)
        mu = 0
        std = 0
        # transforms result into random ensembles
        for j in range(0, M):
            for i in range(0, NTEST):
                idx = self._get_pi_idx(result[i, j], out_pi[i])
                mu = out_mu[i, idx]
                std = out_sigma[i, idx]
                result[i, j] = mu + rn[i, j]*std
        return result