import tensorflow as tf
import numpy as np
import tf_util as U

# https://stackoverflow.com/questions/35488717/confused-about-conv2d-transpose?noredirect=1&lq=1

sess = tf.Session()
batch_size = 3
output_shape = [batch_size, 64, 64, 1]
strides = [1, 2, 2, 1]

w = tf.constant(0.1, shape=[4,4,1,32])

output = tf.constant(0.1, shape=output_shape)
expected_l = tf.nn.conv2d(output, w, strides=strides, padding = "SAME")
print expected_l.get_shape()