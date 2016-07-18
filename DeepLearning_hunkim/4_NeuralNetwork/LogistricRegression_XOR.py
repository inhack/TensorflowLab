#-*- coding: utf-8 -*-
# XOR with Logistic Regression

import tensorflow as tf
import numpy as np

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

xy = np.loadtxt('dataset.txt', unpack=True)
x_data = xy[0:-1]
y_data = xy[-1:]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 초기 weight을 랜덤값으로 정의
W = init_weights([1, len(x_data)])

# Hypothesis
h = tf.matmul(W, X)
hypothesis = tf.div(1., 1.+tf.exp(-h))

# cost function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))

# minimize the cost
alpha = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    # Learning!
    for step in xrange(1000):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 100 == 0:
            print "["+str(step)+"]"
            print "Cost   :",sess.run(cost, feed_dict={X:x_data, Y:y_data})
            print "Weight :",sess.run(W)

    # test the learning model
    print "Model Test"
    correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)

    # Accuracy 계산
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data})
    print "Accuracy :", accuracy.eval({X:x_data, Y:y_data})
