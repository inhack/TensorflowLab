#-*- coding: utf-8 -*-
# Matrix Multiplication

import tensorflow as tf

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

x_data = [[1., 0., 3., 0., 5.],
          [0., 2., 0., 4., 0.]]
y_data = [1., 2., 3., 4., 5.]

# W[0],W[1] 1이 되고 b가 0에 가까운 경우가 데이터에 가장 잘맞는 모델
W = init_weights([1,2])
b = init_weights([1])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#hypothesis = W1 * X1 + W2 * X2 + b
hypothesis = tf.matmul(W, X) + b

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
            print "Bias   :",sess.run(b)

    # Test Model
    print "Model Test"
    #print sess.run(hypothesis, feed_dict={X:5})
    #print sess.run(hypothesis, feed_dict={X:2.5})
