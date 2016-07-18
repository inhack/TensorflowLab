#-*- coding: utf-8 -*-

import tensorflow as tf

# 함수의 파라미터와 비슷하게 타입을 미리 정해놓고 실행시점에 값을 설정
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a,b)
mul = tf.mul(a,b)

with tf.Session() as sess:
    print sess.run(add, feed_dict={a:2, b:3})
    print sess.run(mul, feed_dict={a:2, b:3})
