#-*- coding: utf-8 -*-
# Loading Dataset From File

import tensorflow as tf
import numpy as np

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# 파일로부터(ex> .csv) 데이터셋을 읽어들임
xy = np.loadtxt('./train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

# W[1],W[2] 1이 되고 b(W[0])가 0에 가까운 경우가 데이터에 가장 잘맞는 모델
W = init_weights([1,3])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#hypothesis = W1 * X1 + W2 * X2 + b
hypothesis = tf.matmul(W, X)

cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Learning Rate
alpha = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for step in xrange(4001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 20 == 0:
            print "["+str(step)+"]"
            print "Cost   :",sess.run(cost, feed_dict={X:x_data, Y:y_data})
            print "Weight :",sess.run(W)
