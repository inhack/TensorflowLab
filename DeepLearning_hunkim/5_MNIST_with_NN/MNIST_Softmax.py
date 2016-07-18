#-*- coding: utf-8 -*-
# Softmax Classifier for MNIST

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

# learning rate
alpha = 0.001
training_epochs = 30
batch_size = 100
display_step = 1

# 입력데이터인 mnist image 데이터는 28x28의 형태 (784)
X = tf.placeholder("float", [None, 784])
# Multinomial한 결과로 0~9의 값을 가지기 때문에 10
Y = tf.placeholder("float", [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

activation = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(activation), reduction_indices=1))  # Cross Entropy
optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            sess.run(optimizer, feed_dict = {X: batch_xs, Y: batch_ys})

            avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y:batch_ys}) / total_batch

        if epoch % display_step == 0:
            print "Epoch :", '%04d' % (epoch+1)
            print "Cost  :", '{:.9f}'.format(avg_cost)

    print "Optimization Finished!"

    print ""
    print "Model Test"
    correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(Y, 1))

    # Calculate Accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy :", accuracy.eval({X: mnist.test.images, Y:mnist.test.labels})
