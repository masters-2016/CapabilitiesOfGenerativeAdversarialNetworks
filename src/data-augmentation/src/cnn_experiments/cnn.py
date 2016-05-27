import tensorflow as tf
import math, random, numpy

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape): 
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def random_batch(x, y, batch_size):
    xr = []
    yr = []

    for i in range(batch_size):
        idx = random.randint(0, len(x)-1)
        xr.append(x[idx])
        yr.append(y[idx])

    return numpy.array(xr), numpy.array(yr)


class CNN():

    def __init__(self):
        self.sess = tf.InteractiveSession()

        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])

        self.W_conv1 = weight_variable([5, 5, 1, 32])
        self.b_conv1 = bias_variable([32])

        self.x_image = tf.reshape(self.x, [-1,28,28,1])

        self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = max_pool_2x2(self.h_conv1)

        self.W_conv2 = weight_variable([5, 5, 32, 64])
        self.b_conv2 = bias_variable([64])

        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = max_pool_2x2(self.h_conv2)

        self.W_fc1 = weight_variable([7 * 7 * 64, 1024])
        self.b_fc1 = bias_variable([1024])

        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7*7*64])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

        self.keep_prob = tf.placeholder(tf.float32)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        self.W_fc2 = weight_variable([1024, 10])
        self.b_fc2 = bias_variable([10])

        self.y_conv=tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)

        self.loss = tf.reduce_mean(tf.abs(self.y_ - self.y_conv), reduction_indices=[1])
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        self.sess.run(tf.initialize_all_variables())


    def validate(self,x,y):
        loss = 0.0
        output = self.forward(x)

        for i in range(len(x)):
            target = y[i][0]
            actual = output[i][0]
            loss += abs(target - actual)

        return loss / float(len(x))

    def forward(self, x):
        return self.sess.run(self.y_conv, feed_dict={ self.x: x, self.keep_prob: 1.0 })

    def train(self, x, y):
        self.train_step.run(feed_dict={ self.x: x, self.y_: y, self.keep_prob: 1.0 })

        return self.validate(x,y)
