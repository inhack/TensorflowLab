#-*- coding: utf-8 -*-

import tensorflow as tf

sess = tf.Session()

a = tf.constant(2)
b = tf.constant(3)

c = a + b
d = a * b

# is not working well
print c
print d

# is working
print sess.run(c)
print sess.run(d)
