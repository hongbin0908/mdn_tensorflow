import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math
import keras
from keras.layers import Dense, Activation
from keras.initializers import random_normal
from keras.layers.advanced_activations import ELU

local_path = os.path.dirname(__file__)
root = os.path.join(local_path, '..')
sys.path.append(root)

def get_data1(nsample=1000):
    x_data = np.float32(np.random.uniform(-10.5, 10.5, (1, nsample))).T
    r_data = np.float32(np.random.normal(size=(nsample,1)))
    y_data = np.float32(np.sin(0.75*x_data)*7.0+x_data*0.5+r_data*1.0)
    return x_data, y_data

def get_data2(nsample=1000):
    tmp = get_data1(nsample)
    return (tmp[1], tmp[0])


class Model:
    def fit(self, x_data, y_data):
        pass
    def predict(self, x_data):
        pass

class Model1(Model):
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None,1])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None,1])
        NHIDDEN = 20
        W = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=1.0, dtype=tf.float32))
        b = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=1.0, dtype=tf.float32))
        W_out = tf.Variable(tf.random_normal([NHIDDEN,1], stddev=1.0, dtype=tf.float32))
        b_out = tf.Variable(tf.random_normal([1,1], stddev=1.0, dtype=tf.float32))
        hidden_layer = tf.nn.tanh(tf.matmul(self.x, W) + b)
        self.y_out = tf.matmul(hidden_layer,W_out) + b_out
        lossfunc = tf.nn.l2_loss(self.y_out-self.y)
        self.train_op = tf.train.RMSPropOptimizer(learning_rate=0.1, decay=0.8).minimize(lossfunc)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())
    def fit(self, x_data, y_data):
        NEPOCH = 1000
        for i in range(NEPOCH):
            self.sess.run(self.train_op,feed_dict={self.x: x_data, self.y: y_data})
    def predict(self, x_data):
        return self.sess.run(self.y_out,feed_dict={self.x: x_test})


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
        NTEST = x_test.size
        result = np.random.rand(NTEST, M) # initially random [0, 1]
        rn = np.random.randn(NTEST, M) # normal random matrix (0.0, 1.0)
        # transforms result into random ensembles
        for j in range(0, M):
            for i in range(0, NTEST):
                mu = out_mu[i,0]
                std = out_sigma[i,0]
                result[i, j] = mu + rn[i, j]*std
        return result
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
        NTEST = x_test.size
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

x_test = np.float32(np.arange(-10.5,10.5,0.1))
x_test = x_test.reshape(x_test.size,1)

#x_data, y_data = get_data1()
#model = Model1()
#model.fit(x_data, y_data)
#y_test = model.predict(x_test)
#
#plt.figure(figsize=(8, 8))
#plt.plot(x_data,y_data,'ro', x_test,y_test,'bo',alpha=0.3)
#plt.savefig(os.path.join(local_path, "mdn_ex2_fig1.png"))
#
x_data, y_data = get_data1()
model = Model2()
model.fit(x_data, y_data)
y_test = model.predict(x_test)

plt.figure(figsize=(8, 8))
plt.plot(x_data,y_data,'ro', x_test,y_test,'bo',alpha=0.3)
plt.savefig(os.path.join(local_path, "mdn_ex2_fig2.png"))

x_data, y_data = get_data2(2500)

modelMdn = ModelMdn()
modelMdn.fit(x_data, y_data)

plt.figure(figsize=(8, 8))
plt.plot(np.arange(100, modelMdn.NEPOCH,1), modelMdn.loss[100:], 'r-')
plt.savefig(os.path.join(local_path, "mdn_ex2_fig3.png"))


x_test = np.float32(np.arange(-15,15,0.1))
NTEST = x_test.size
x_test = x_test.reshape(NTEST,1) # needs to be a matrix, not a vector

y_test = modelMdn.predict(x_test)

plt.figure(figsize=(9, 9))
plt.plot(x_data,y_data,'ro', x_test,y_test,'bo',alpha=0.3)
plt.savefig(os.path.join(local_path, "mdn_ex2_fig4.png"))
