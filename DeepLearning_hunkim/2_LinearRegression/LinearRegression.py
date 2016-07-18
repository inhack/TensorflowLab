#-*- coding: utf-8 -*-

import tensorflow as tf

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# Training Dataset
x_data = [1. ,2. ,3.]
y_data = [1. ,2. ,3.]

# Input
X = tf.placeholder(tf.float32)
# Output
Y = tf.placeholder(tf.float32)

# W는 1이 되고, b가 0이되는 경우가 데이터에 가장 잘맞는 모델
# W,b는 랜덤한 값으로 시작
W = init_weights([1])
b = init_weights([1])

# Hypothesis
hypothesis = W * X + b

# Cost Function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Learning Rate
alpha = tf.Variable(0.01)
# Gradent Descent(경사하강법)를 사용하여 cost를 최적화,최소화
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(cost)

# 실제로 계산이 이루어짐
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
    print sess.run(hypothesis, feed_dict={X:5})
    print sess.run(hypothesis, feed_dict={X:2.5})
