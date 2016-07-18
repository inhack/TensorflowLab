#-*- coding: utf-8 -*-
# Bias Integration into X

import tensorflow as tf

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# Bias값을 X에 넣어 식을 간단하게!
x_data = [[1., 1., 1., 1., 1.],
          [1., 0., 3., 0., 5.],
          [0., 2., 0., 4., 0.]]
y_data = [1., 2., 3., 4., 5.]

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
