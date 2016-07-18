#-*- coding: utf-8 -*-

import tensorflow as tf

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

x1_data = [1., 0., 3., 0., 5.]
x2_data = [0., 2., 0., 4., 0.]
y_data = [1., 2., 3., 4., 5.]


# W1과 W2가 1이 되고 b가 0에 가까운 경우가 데이터에 가장 잘맞는 모델
W1 = init_weights([1])
W2 = init_weights([1])
b = init_weights([1])

X1 = tf.placeholder(tf.float32)
X2 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = W1 * X1 + W2 * X2 + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Learning Rate
alpha = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for step in xrange(4001):
        sess.run(train, feed_dict={X1:x1_data, X2:x2_data, Y:y_data})
        if step % 20 == 0:
            print "["+str(step)+"]"
            print "Cost    :",sess.run(cost, feed_dict={X1:x1_data, X2:x2_data, Y:y_data})
            print "Weight1 :",sess.run(W1)
            print "Weight2 :",sess.run(W2)
            print "Bias    :",sess.run(b)
