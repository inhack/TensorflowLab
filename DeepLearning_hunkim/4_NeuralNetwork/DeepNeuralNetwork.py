#-*- coding: utf-8 -*-
# XOR with Deep Neural Network

import tensorflow as tf
import numpy as np

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

xy = np.loadtxt('dataset.txt', unpack=True)
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4,1))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# DeepNN Weights
# 초기 weight을 랜덤값으로 정의
W1 = tf.Variable(tf.random_uniform([2,5], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([5,4], -1.0, 1.0))
W3 = tf.Variable(tf.random_uniform([4,1], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([5]), name="Bias1")
b2 = tf.Variable(tf.zeros([4]), name="Bias2")
b3 = tf.Variable(tf.zeros([1]), name="Bias3")

# Hypothesis (2 Layers)
L2 = tf.sigmoid(tf.matmul(X, W1) + b1)              # Layer 1 : 2 -> 5
L3 = tf.sigmoid(tf.matmul(L2, W2) + b2)             # Layer 2 : 5 -> 4
hypothesis = tf.sigmoid(tf.matmul(L3, W3) + b3)     # Layer 3 : 4 -> 1

# cost function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1.-hypothesis))

# minimize the cost
alpha = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    # Learning!
    for step in xrange(10001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 1000 == 0:
            print "["+str(step)+"]"
            print "Cost    :",sess.run(cost, feed_dict={X:x_data, Y:y_data})
            print "Weight1 :",sess.run(W1)
            print "Weight2 :",sess.run(W2)
            print "Weight3 :",sess.run(W3)

    # test the learning model
    print "Model Test"
    correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)

    # Accuracy 계산
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data})
    print "Accuracy :", accuracy.eval({X:x_data, Y:y_data})
