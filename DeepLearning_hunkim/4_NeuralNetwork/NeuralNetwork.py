#-*- coding: utf-8 -*-
# XOR with Neural Network (Sigmoid)

import tensorflow as tf
import numpy as np

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

xy = np.loadtxt('dataset.txt', unpack=True)
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4,1))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# NN Weights
# 초기 weight을 랜덤값으로 정의
# 아래와 같은 방식으로 weight을 정할 시 Accuracy가 0.5
#W1 = init_weights([2, 2])
#W2 = init_weights([2, 1])
W1 = tf.Variable(tf.random_uniform([2,2], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([2,1], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([2]), name="Bias1")
b2 = tf.Variable(tf.zeros([1]), name="Bias2")

# Hypothesis (2 Layers)
L2 = tf.sigmoid(tf.matmul(X, W1) + b1)              # Layer 1
hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)     # Layer 2

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

    # test the learning model
    print "Model Test"
    correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)

    # Accuracy 계산
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data})
    print "Accuracy :", accuracy.eval({X:x_data, Y:y_data})
