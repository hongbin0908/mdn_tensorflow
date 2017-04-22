import tensorflow as tf
from model_base import Model

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
        return self.sess.run(self.y_out,feed_dict={self.x: x_data})
