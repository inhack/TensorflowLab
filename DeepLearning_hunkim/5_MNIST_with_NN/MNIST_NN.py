#-*- coding: utf-8 -*-
# Deep Neural Network for MNIST         => Accuracy : 94~95%
# + Xavier                              => Accuracy : 97~98%
# + dropout (overfitting 방지를 위해)      => Accuracy : 98~99%

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

def xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt( 6.0 / (n_inputs+n_outputs) )
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt( 3.0 / (n_inputs+n_outputs) )
        return tf.truncated_normal_initializer(stddev=stddev)

# learning rate
alpha = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# 입력데이터인 mnist image 데이터는 28x28의 형태 (784)
X = tf.placeholder("float", [None, 784])
# Multinomial한 결과로 0~9의 값을 가지기 때문에 10
Y = tf.placeholder("float", [None, 10])

# 3 Layers
# Weight와 Bias의 초기값을 잘 지정해주면 더 좋은 효율을 낼 수 있다!
# Xavier Initialization 사용
"""
W1 = tf.Variable(tf.random_normal([784,256]))
W2 = tf.Variable(tf.random_normal([256,256]))
W3 = tf.Variable(tf.random_normal([256,10]))
B1 = tf.Variable(tf.random_normal([256]))
B2 = tf.Variable(tf.random_normal([256]))
B3 = tf.Variable(tf.random_normal([10]))
"""
W1 = tf.get_variable("W1", shape=[784, 256], initializer=xavier_init(784,256))
W2 = tf.get_variable("W2", shape=[256, 512], initializer=xavier_init(256,512))
W3 = tf.get_variable("W3", shape=[512, 256], initializer=xavier_init(512,256))
W4 = tf.get_variable("W4", shape=[256, 256], initializer=xavier_init(256,256))
W5 = tf.get_variable("W5", shape=[256, 10], initializer=xavier_init(256,10))
B1 = tf.Variable(tf.zeros([256]))
B2 = tf.Variable(tf.zeros([512]))
B3 = tf.Variable(tf.zeros([256]))
B4 = tf.Variable(tf.zeros([256]))
B5 = tf.Variable(tf.zeros([10]))

# Construct Deep Neural Network Model using ReLu(Hidden Layer)!!
"""
L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2))
hypothesis = tf.add(tf.matmul(L2, W3), B3)
"""
# overfitting 방지를 위하여 dropout을 사용
# droupout_rate는 학습 시에는 30%를 drop시켜 70%만 학습에 활용하게 함
# 실제 학습이 완료된 모델에서는 100% 모두 활용함
dropout_rate = tf.placeholder("float")

_L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), B1))
L1 = tf.nn.dropout(_L1, dropout_rate)

_L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2))
L2 = tf.nn.dropout(_L2, dropout_rate)

_L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), B3))
L3 = tf.nn.dropout(_L3, dropout_rate)

_L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4), B4))
L4 = tf.nn.dropout(_L4, dropout_rate)

hypothesis = tf.add(tf.matmul(L4, W5), B5)

# cost 함수를 정의할 시, 직접 구현한 cross entropy를 사용할 수도 있지만
# 아래와 같이 tensorflow가 제공하는 cross entropy 사용!
# L1, L2 다음의 hypothesis에서 softmax 값을 구하지 않았기 때문에, cost 함수 계산 시에 softmax_cross_entropy_with_logits을 사용
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y))
# 기존의 GradientDescentOptimizer를 사용하지 않고 AdamOptimizer를 사용
optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(cost)

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            sess.run(optimizer, feed_dict = {X: batch_xs, Y: batch_ys, dropout_rate: 0.7})

            avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y:batch_ys, dropout_rate: 0.7}) / total_batch

        if epoch % display_step == 0:
            print "Epoch :", '%04d' % (epoch+1)
            print "Cost  :", '{:.9f}'.format(avg_cost)

    print "Optimization Finished!"

    print ""
    print "Model Test"
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))

    # Calculate Accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy :", accuracy.eval({X: mnist.test.images, Y:mnist.test.labels, dropout_rate: 1})
