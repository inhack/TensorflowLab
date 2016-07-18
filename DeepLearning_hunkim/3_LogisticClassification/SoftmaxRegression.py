#-*- coding: utf-8 -*-
# Multinomial Logistic Classification

import tensorflow as tf
import numpy as np

xy = np.loadtxt('train2.txt', unpack=True, dtype='float32')
# Transpose the Matrix
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

print x_data

print y_data

X = tf.placeholder("float", [None,3])
Y = tf.placeholder("float", [None,3])

W = tf.Variable(tf.zeros([3,3]))

# tf.matmul(X, W) because of transpose
hypothesis = tf.nn.softmax(tf.matmul(X,W))

# Learning Rate
alpha = 0.001
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for step in range(10000):
        sess.run(optimizer,feed_dict={X:x_data,Y:y_data})
        if(step%1000==0):
            print "["+str(step)+"]"
            print "Cost   :",sess.run(cost, feed_dict={X:x_data, Y:y_data})
            print "Weight :",sess.run(W)

    print "Model Test"
    a = sess.run(hypothesis, feed_dict={X:[[1,11,7]]})
    print a, sess.run(tf.arg_max(a,1))
    b = sess.run(hypothesis, feed_dict={X:[[1,3,4]]})
    print b, sess.run(tf.arg_max(b,1))
    c = sess.run(hypothesis, feed_dict={X:[[1,1,0]]})
    print c, sess.run(tf.arg_max(c,1))
    all = sess.run(hypothesis, feed_dict={X:[[1,11,7],[1,3,4],[1,1,0]]})
    print all, sess.run(tf.arg_max(all,1))
