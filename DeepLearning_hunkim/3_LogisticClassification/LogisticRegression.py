#-*- coding: utf-8 -*-
# Logistic Classification (Binary Classfication)

import tensorflow as tf
import numpy as np

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

xy = np.loadtxt('train.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# len(x_data)는 데이터셋 즉 row의 개수가 아닌 variable의 개수인 column
# 여기서는 b , w0, w1로 3!
W = init_weights([1, len(x_data)])

# Hypothesis
h = tf.matmul(W, X)
hypothesis = tf.div(1., 1.+tf.exp(-h))

# Logistic Regression의 cost Function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))

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

    print "Model Test"
    print sess.run(hypothesis, feed_dict={X:[[1], [2], [2]]}) > 0.5
    print sess.run(hypothesis, feed_dict={X:[[1], [5], [5]]}) > 0.5
    print sess.run(hypothesis, feed_dict={X:[[1,1], [4,3], [3,5]]}) > 0.5
    # Flase
    # True
    # False True
