#! encoding: UTF-8

__author__ = "Xinpeng.Chen"

import os
import glob
import numpy as np
import cPickle as pickle

import tensorflow as tf

import ipdb


##################################################################################################
'''
The input is the voxel grid of size 20^3, followed by:
(1) a convolution layer with 20 feature maps of size  5 x 5 x 5 resulting 
    in 20 x 16 x 16 x 16 outputs

(2) a max pooling layer with 2 x 2 x 2-sized non-overlapping divisions
    resulting in 20 x 8 x 8 x 8 outputs

(3) a second convolutional layer with 20 feature maps of size 5 x 5 x 5
    resulting in 20 x 4 x 4 x 4 outputs

(4) a second max pooling layer with 2 x 2 x 2-sized non-overlapping divisions
    resulting in 20 x 2 x 2 x 2 outputs

(5) a fully connected layer with 300 hidden nodes

(6) final output is based on a softmax over 10 labels, including 9 categories 
    and an empty label
'''
##########################################  parameters  ########################################
batch_size = 1
n_epochs = 100
learning_rate = 0.00005

fc_size = 300
num_classes = 10

depth = 20
height = 20
width = 20
channels = 1

loss = 0.0

def _weight_variable(name, shape):
    return tf.get_variable(name, shape, tf.float32, tf.truncated_normal_initializer(stddev=0.1))

def _bias_variable(name, shape):
    return tf.get_variable(name, shape, tf.float32, tf.constant_initializer(0.1, dtype=tf.float32))
    

# Placeholder
tf_prev_layer = tf.placeholder(tf.float32, [batch_size, depth, height, width, channels])
tf_labels = tf.placeholder(tf.int32, [batch_size, num_classes])

# (1)
in_filters = 1
with tf.variable_scope("conv1") as scope:
    out_filters = 20
    kernel = _weight_variable("weights", [5, 5, 5, in_filters, out_filters])
    conv = tf.nn.conv3d(tf_prev_layer, kernel, [1, 1, 1, 1, 1], padding="SAME")
    biases = _bias_variable("biases", [out_filters])
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)

    prev_layer = conv1
    in_filters = out_filters

# (2)
pool1 = tf.nn.max_pool3d(prev_layer, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding="SAME")
norm1 = pool1

# (3)
prev_layer = norm1
with tf.variable_scope("conv2") as scope:
    out_filters = 20
    kernel = _weight_variable("weights", [5, 5, 5, in_filters, out_filters])
    conv = tf.nn.conv3d(prev_layer, kernel, [1, 1, 1, 1, 1], padding="SAME")
    biases = _bias_variable("biases", [out_filters])
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)

    prev_layer = conv2
    in_filters = out_filters

# (4)
pool2 = tf.nn.max_pool3d(prev_layer, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding="SAME")
norm2 = pool2

# (5)
prev_layer = norm2
with tf.variable_scope("fully_connected_layer") as scope:
    dim = np.prod(prev_layer.get_shape().as_list()[1:])
    prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
    weights = _weight_variable("weights", [dim, fc_size])
    biases = _bias_variable("biases", [fc_size])
    fully_connected_layer = tf.nn.relu(tf.matmul(prev_layer_flat, weights) + biases, name=scope.name)

# (6)
prev_layer = fully_connected_layer
with tf.variable_scope("softmax_linear") as scope:
    dim = np.prod(prev_layer.get_shape().as_list()[1:])
    weights = _weight_variable("weights", [dim, num_classes])
    biases = _bias_variable("biases", [num_classes])
    logits = tf.add(tf.matmul(prev_layer, weights), biases, name=scope.name)

# calculate loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, tf_labels, name="cross_entropy") 
loss = tf.reduce_mean(cross_entropy)/batch_size

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

##################################################################################################
points = open("test.txt", "r").read().splitlines()

line_points = []
for idx, line in enumerate(points):
    for point in line.split(" "):
        line_points.append(int(point))

voxel_points = np.asarray(line_points).astype(np.float32)
voxel_points = np.reshape(voxel_points, [depth, height, width])

[depth, height, width] = voxel_points.shape
current_voxel = np.reshape(voxel_points, [1, depth, height, width, 1])
labels = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for epoch in range(0, 100):
    feed_dict = {tf_prev_layer: current_voxel,
                 tf_labels: labels}
    print sess.run(loss, feed_dict)
    sess.run(train_op, feed_dict)
    print sess.run(logits, feed_dict)

