#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf

weight = tf.get_variable(shape=[10, 10], initializer=tf.truncated_normal_initializer(), name="weight2")
sess = tf.Session()
sess.run(tf.global_variables_initializer())
w = sess.run(weight)
print(w)