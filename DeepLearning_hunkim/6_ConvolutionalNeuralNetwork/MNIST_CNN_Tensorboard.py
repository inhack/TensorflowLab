#-*- coding: utf-8 -*-
# MNIST with Convolutional Neural Network : 99 ~ 100%
# Tensorboard

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import datetime

batch_size = 128
test_size = 256

def init_weights(shape, varname):
    return tf.Variable(tf.random_normal(shape, stddev=0.01),name = varname)

# Create Convolutional NN Model
def model(X, W1, W2, W3, W4, W_O, p_keep_conv, p_keep_hidden):
    # feature extraction
    with tf.name_scope("layer1") as scope:
        l1a = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME'))        # l1a = (?, 28, 28, 32)
        l1 = tf.nn.max_pool(l1a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    # l1 = (? 14, 14, 32)
        l1 = tf.nn.dropout(l1, p_keep_conv)
    with tf.name_scope("layer2") as scope:
        l2a = tf.nn.relu(tf.nn.conv2d(l1, W2, strides=[1,1,1,1], padding='SAME'))       # l2a = (?, 14, 14 64)
        l2 = tf.nn.max_pool(l2a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    # l2 = (?, 7, 7, 64)
        l2 = tf.nn.dropout(l2, p_keep_conv)
    with tf.name_scope("layer3") as scope:
        l3a = tf.nn.relu(tf.nn.conv2d(l2, W3, strides=[1,1,1,1], padding='SAME'))       # l3a = (?, 7, 7, 128)
        l3 = tf.nn.max_pool(l3a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    # l3 = (?, 4, 4, 128)
    # fully connected layer
        l3 = tf.reshape(l3, [-1, W4.get_shape().as_list()[0]])      # reshapte to (?, 2048) / 4 * 4 * 128 = 2048
        l3 = tf.nn.dropout(l3, p_keep_conv)
    with tf.name_scope("layer4") as scope:
        l4 = tf.nn.relu(tf.matmul(l3, W4))
        l4 = tf.nn.dropout(l4, p_keep_hidden)
    with tf.name_scope("layer5") as scope:
        pyx = tf.matmul(l4, W_O)
        return pyx

starttime = datetime.datetime.now()

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)
teX = teX.reshape(-1, 28, 28, 1)

X = tf.placeholder("float", [None, 28, 28, 1], name="X-input")
Y = tf.placeholder("float", [None, 10], name="Y-input")

# filter의 사이즈는 원하는대로 설정하면 된다
# feature extraction
W1 = init_weights([3,3,1,32],"Weight1")    # filter : 3,3,1 / output layer 32
W2 = init_weights([3,3,32,64],"Weight2")
W3 = init_weights([3,3,64,128],"Weight3")
# fully connected layer
W4 = init_weights([128 * 4 * 4, 625],"Weight4")
W_O = init_weights([625, 10],"WeightO")    # output, label : 10

w1_hist = tf.histogram_summary("weights1", W1)
w2_hist = tf.histogram_summary("weights2", W2)
w3_hist = tf.histogram_summary("weights3", W3)
w4_hist = tf.histogram_summary("weights4", W4)
w5_hist = tf.histogram_summary("weightsO", W_O)

# for dropout
p_keep_conv = tf.placeholder("float", name="p_keep_conv")
p_keep_hidden = tf.placeholder("float", name="p_keep_hidden")

py_x = model(X, W1, W2, W3, W4, W_O, p_keep_conv, p_keep_hidden)


with tf.name_scope("cost") as scope:
    # py_x가 예측값, Y가 실제값
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    cost_summ = tf.scalar_summary("cost", cost)

with tf.name_scope("train_op") as scope:
    # cost 최적화 학습
    train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

# 학습된 모델로 테스트 및 정확도 측정
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(py_x, 1))
    acc_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.scalar_summary("accuracy", acc_op)

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("./logs/CNN_MNIST_logs", sess.graph)

    for i in range(5):
        # 0~128
        # 128~256
        # 256~384
        # 384 ~ 512
        # ...
        print i
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX), batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end], p_keep_conv:0.8, p_keep_hidden: 0.5})

        # Model Test
        summary, acc = sess.run([merged, acc_op], feed_dict={X: teX, Y: teY, p_keep_conv: 1.0, p_keep_hidden: 1.0})
        writer.add_summary(summary, i)

    endtime = datetime.datetime.now()
    print "Start :",starttime
    print "End   :",endtime
