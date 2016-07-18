#-*- coding: utf-8 -*-
# Tensorboard for Deep NN

"""
* 5 steps of using Tensorboard

(1) From TF Graph, decide which node you want to annotate
 - 사용하는 데이터, 노드 중 출력하고 싶은 것을 선별
 - with tf.name_scope("test") as scope:
 - tf.histogram_summary("weights",W)
 - tf.scalar_summary("accuracy", accuracy)

(2) Merge all summaries
 - Tensorflow의 Operation을 실행시키기 전에, (1)에서 선별한 데이터를 모아서 Operation으로 만듬
 - merged = tf.merge_all_summaries()

(3) Create writer
 - 결과를 쓸 디렉토리를 지정
 - writer = tf.train.SummaryWriter("/tmp/mnist_logs", sess.graph_def)

(4) Run summary merge and add_summary
 - merge된 operation을 실행한 후, Writer에 전달
 - summary = sess.run(merged, ...)
 - writer.add_summary(summary)

(5) Launch Tensorboard
 - 실제 tensorboard를 실행
 - tensorboard --logdir=/tmp/mnist_logs

"""

import tensorflow as tf
import numpy as np

xy = np.loadtxt('dataset.txt', unpack=True)
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4,1))

X = tf.placeholder(tf.float32, name = 'X-input')
Y = tf.placeholder(tf.float32, name = 'Y-input')

W1 = tf.Variable(tf.random_uniform([2,5], -1.0, 1.0), 'Weitgh1')
W2 = tf.Variable(tf.random_uniform([5,4], -1.0, 1.0), 'Weitgh2')
W3 = tf.Variable(tf.random_uniform([4,1], -1.0, 1.0), 'Weitgh3')

b1 = tf.Variable(tf.zeros([5]), name="Bias1")
b2 = tf.Variable(tf.zeros([4]), name="Bias2")
b3 = tf.Variable(tf.zeros([1]), name="Bias3")

w1_hist = tf.histogram_summary("weights1", W1)                                      # (1)
w2_hist = tf.histogram_summary("weights2", W2)                                      # (1)
w3_hist = tf.histogram_summary("weights3", W3)                                      # (1)

b1_hist = tf.histogram_summary("biases1", b1)                                       # (1)
b2_hist = tf.histogram_summary("biases2", b2)                                       # (1)
b3_hist = tf.histogram_summary("biases3", b3)                                       # (1)

y_hist = tf.histogram_summary("y", Y)                                               # (1)

with tf.name_scope("layer2") as scope:                                              # (1)
    L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
with tf.name_scope("layer3") as scope:                                              # (1)
    L3 = tf.sigmoid(tf.matmul(L2, W2) + b2)
with tf.name_scope("layer4") as scope:                                              # (1)
    hypothesis = tf.sigmoid(tf.matmul(L3, W3) + b3)

with tf.name_scope("cost") as scope:                                                # (1)
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))
    cost_summ = tf.scalar_summary("cost", cost)                                     # (1)

with tf.name_scope("train") as scope:                                               # (1)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

with tf.Session() as sess:

    tf.initialize_all_variables().run()

    merged = tf.merge_all_summaries()                                               # (2)
    writer = tf.train.SummaryWriter("./logs/xor_logs", sess.graph)                  # (3)

#    for step in xrange(200000):
#        summary, _ = sess.run([merged, train], feed_dict = {X:x_data, Y:y_data})    # (4)
#        writer.add_summary(summary, step)                                           # (4)

    for step in xrange(20001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 2000 == 0:
            summary, _ = sess.run([merged,train], feed_dict={X:x_data, Y:y_data})    # (4)
            writer.add_summary(summary, step)                                        # (4)

    print "Model Test"
    correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)

    # Accuracy 계산
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data})
    print "Accuracy :", accuracy.eval({X:x_data, Y:y_data})

# $tensorboard --logdir=./logs/xor_logs
