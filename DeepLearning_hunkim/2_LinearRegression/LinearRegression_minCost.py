#-*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt

X = [1., 2., 3.]
Y = [1., 2., 3.]

W_val = []
cost_val = []

m = n_sample = len(X)

W = tf.placeholder(tf.float32)

hypothesis = tf.mul(X, W)

cost = tf.reduce_sum(tf.pow(hypothesis-Y, 2))/(m)

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for i in range(-30, 50):
        print i * 0.1, sess.run(cost, feed_dict={W: i*0.1})
        W_val.append(i*0.1)
        cost_val.append(sess.run(cost, feed_dict={W: i*0.1}))

plt.plot(W_val, cost_val, 'ro')
plt.ylabel('Cost')
plt.xlabel('W')
plt.show()
