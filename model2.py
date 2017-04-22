import math
import tensorflow as tf
import numpy as np
from model_base import Model
class Model2(Model):
    def __init__(self):
        NHIDDEN = 24
        STDEV = 0.5
        NOUT = 2  # mu, stdev
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="x")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="y")
        Wh = tf.Variable(tf.random_normal([1, NHIDDEN], stddev=STDEV, dtype=tf.float32))
        bh = tf.Variable(tf.random_normal([1, NHIDDEN], stddev=STDEV, dtype=tf.float32))
        Wo = tf.Variable(tf.random_normal([NHIDDEN, NOUT], stddev=STDEV, dtype=tf.float32))
        bo = tf.Variable(tf.random_normal([1, NOUT], stddev=STDEV, dtype=tf.float32))
        hidden_layer = tf.nn.tanh(tf.matmul(self.x, Wh) + bh)
        self.output = tf.matmul(hidden_layer, Wo) + bo
        out_sigma, out_mu = self._get_mixture_coef(self.output)
        self.lossfunc = self._get_lossfunc(out_sigma, out_mu, self.y)
        self.train_op = tf.train.AdamOptimizer().minimize(self.lossfunc)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())

    def fit(self, x_, y_):
        self.NEPOCH = 10000
        self.loss = np.zeros(self.NEPOCH)  # store the training progress here.
        for i in range(self.NEPOCH):
            self.sess.run(self.train_op, feed_dict={self.x: x_, self.y: y_})
            loss = self.sess.run(self.lossfunc, feed_dict={self.x: x_, self.y: y_})
            print(loss)
            self.loss[i] = loss

    def predict(self, x_):
        out_sigma, out_mu = self.sess.run(self._get_mixture_coef(self.output),feed_dict={self.x: x_})
        return self._generate_ensemble(out_mu, out_sigma)

    def _get_mixture_coef(self, output):
        out_sigma, out_mu = tf.split(output, 2, 1)
        out_sigma = tf.exp(out_sigma)
        return out_sigma, out_mu

    def _tf_normal(self, y, mu, sigma):
        oneDivSqrtTwoPI = 1 / math.sqrt(2 * math.pi)  # normalisation factor for gaussian, not needed.
        result = tf.subtract(y, mu)
        result = tf.multiply(result, tf.reciprocal(sigma))
        result = -tf.square(result) / 2
        return tf.multiply(tf.exp(result), tf.reciprocal(sigma)) * oneDivSqrtTwoPI

    def _get_lossfunc(self, out_sigma, out_mu, y):
        result = self._tf_normal(y, out_mu, out_sigma)
        result = tf.reduce_sum(result, 1, keep_dims=True)
        result = -tf.log(result+0.00001)
        return tf.reduce_mean(result)
    def _generate_ensemble(self, out_mu, out_sigma, M = 10):
        NTEST = len(out_mu)
        result = np.random.rand(NTEST, M) # initially random [0, 1]
        rn = np.random.randn(NTEST, M) # normal random matrix (0.0, 1.0)
        # transforms result into random ensembles
        for j in range(0, M):
            for i in range(0, NTEST):
                mu = out_mu[i,0]
                std = out_sigma[i,0]
                result[i, j] = mu + rn[i, j]*std
        return result