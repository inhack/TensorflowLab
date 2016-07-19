#-*- coding: utf-8 -*-
# MNIST with Convolutional Neural Network : 99 ~ 100%
# CNN = feature extraction(subsampling(pooling), convolution) + classification(fully connected layer)

# filter의 크기대로 W와 입력값을 여러번 연산해야하는데 TF에서는 tf.nn.conv2d라는 유용한 함수를 제공
# 첫번째인자는 입력값(이미지)
# 두번째인자는 Weight
# 세번째인자는 strides로 필터를 옮기는 간격 [1,?,?,1]
# 네번째인자는 padding으로 SAME / VALID을 사용 가능 : TF가 알아서 패딩을 정해줌, 원래 이미지와 strides에 따른 Activation Maps을 생성해준다
#           strides가 1,1,1,1일 경우 원래 이미지와 같은 크기의 Activation Map 생성

# subsampling (pooing)
# 기존의 Conv Layer를 변환

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import datetime

batch_size = 128
test_size = 256

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# Create Convolutional NN Model
def model(X, W1, W2, W3, W4, W_O, p_keep_conv, p_keep_hidden):
    # feature extraction
    l1a = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME'))        # l1a = (?, 28, 28, 32)
    l1 = tf.nn.max_pool(l1a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    # l1 = (? 14, 14, 32)
    l1 = tf.nn.dropout(l1, p_keep_conv)
    l2a = tf.nn.relu(tf.nn.conv2d(l1, W2, strides=[1,1,1,1], padding='SAME'))       # l2a = (?, 14, 14 64)
    l2 = tf.nn.max_pool(l2a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    # l2 = (?, 7, 7, 64)
    l2 = tf.nn.dropout(l2, p_keep_conv)
    l3a = tf.nn.relu(tf.nn.conv2d(l2, W3, strides=[1,1,1,1], padding='SAME'))       # l3a = (?, 7, 7, 128)
    l3 = tf.nn.max_pool(l3a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    # l3 = (?, 4, 4, 128)
    # fully connected layer
    l3 = tf.reshape(l3, [-1, W4.get_shape().as_list()[0]])      # reshapte to (?, 2048) / 4 * 4 * 128 = 2048
    l3 = tf.nn.dropout(l3, p_keep_conv)
    l4 = tf.nn.relu(tf.matmul(l3, W4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)
    pyx = tf.matmul(l4, W_O)
    return pyx

starttime = datetime.datetime.now()

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)
teX = teX.reshape(-1, 28, 28, 1)

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

# filter의 사이즈는 원하는대로 설정하면 된다
# feature extraction
W1 = init_weights([3,3,1,32])    # filter : 3,3,1 / output layer 32
W2 = init_weights([3,3,32,64])
W3 = init_weights([3,3,64,128])
# fully connected layer
W4 = init_weights([128 * 4 * 4, 625])
W_O = init_weights([625, 10])    # output, label : 10

# for dropout
p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

py_x = model(X, W1, W2, W3, W4, W_O, p_keep_conv, p_keep_hidden)

# py_x가 예측값, Y가 실제값
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
# cost 최적화 학습
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
# 예측값
# argmax returns the index with the largest value across dimensions of a tensor.
predict_op = tf.argmax(py_x, 1)

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for i in range(100):
        # 0~128
        # 128~256
        # 256~384
        # 384 ~ 512
        # ...
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX), batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end], p_keep_conv:0.8, p_keep_hidden: 0.5})

        # Model Test
        test_indices = np.arange(len(teX))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print "["+str(i)+"]"
        print np.mean(np.argmax(teY[test_indices], axis=1) ==
                        sess.run(predict_op, feed_dict={X: teX[test_indices], Y: teY[test_indices], p_keep_conv: 1.0, p_keep_hidden: 1.0}))

    endtime = datetime.datetime.now()
    print "Start :",starttime
    print "End   :",endtime
