''' Demonstrates image classification using CNNs '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import pickle
import random
import numpy as np ; np.random.seed(seed = 1)
import tensorflow as tf

# rktools
from rktools.monitors import ProgressBar


# Set parameters
image_size = 32
num_channels = 3
num_categories = 10
num_filters = 32
filter_size = 5
num_epochs = 20 # 200
batch_size = 10
num_batches = int(50000/batch_size)
keep_prob = 0.6

# Read CIFAR training data

print(">>> Loading the CIFAR10 data...")

start = time.time()
train_data = None
train_labels = None
for file_index in range(5):
    train_file = open('/home/developer/workspace/rklearn-lib/rklearn/tests/data/CIFAR/cifar-10-batches-py/data_batch_' + str(file_index+1), 'rb')
    train_dict = pickle.load(train_file, encoding='latin1')
    train_file.close()

    if train_data is None:
        train_data = np.array(train_dict['data'], float)/255.0
        train_labels = train_dict['labels']
    else:
        train_data = np.concatenate((train_data, train_dict['data']), 0)
        train_labels = np.concatenate((train_labels, train_dict['labels']), 0)

# Preprocess training data and labels
train_data = train_data.reshape([-1, num_channels, image_size, image_size])
train_data = train_data.transpose([0, 2, 3, 1])
train_labels = np.eye(num_categories)[train_labels]
end = time.time()
print("\ttraining data loaded in {} secs".format((end - start)))


# Read CIFAR test data
start = time.time()
test_file = open('/home/developer/workspace/rklearn-lib/rklearn/tests/data/CIFAR/cifar-10-batches-py/test_batch', 'rb')
test_dict = pickle.load(test_file, encoding='latin1')
test_file.close()
test_data = test_dict['data']
test_labels = test_dict['labels']
end = time.time()
print("\ttesting data loaded in {} secs".format((end - start)))

# Preprocess test data and labels
test_data = test_data.reshape([-1, num_channels, image_size, image_size])
test_data = test_data.transpose([0, 2, 3, 1])
test_labels = np.eye(num_categories)[test_labels]

# for dev (comment for prod)
print(">>> Reducing the size of the CIFAR10 data for DEV...")
train_sample_size = int(len(train_data) * 0.1)
test_sample_size = int(len(test_data) * 0.1)
train_data = train_data[:train_sample_size]
train_labels = train_labels[:train_sample_size]
test_data = test_data[:test_sample_size]
test_labels = test_labels[:test_sample_size]

print("\ttrain_data.shape = ", train_data.shape, ", train_labels.shape = ", train_labels.shape)
print("\ttest_data.shape = ", test_data.shape, ", test_labels.shape = ", test_labels.shape)

# Placeholders for CIFAR images
img_holder = tf.placeholder(tf.float32, [None, image_size, image_size, num_channels])
lbl_holder = tf.placeholder(tf.float32, [None, num_categories])
train = tf.placeholder(tf.bool)

# Create convolution/pooling layers
conv1 = tf.layers.conv2d(img_holder, num_filters, filter_size, padding='same', activation=tf.nn.relu)
drop1 = tf.layers.dropout(conv1, keep_prob, training=train)
pool1 = tf.layers.max_pooling2d(drop1, 2, 2)
conv2 = tf.layers.conv2d(pool1, num_filters, filter_size, padding='same', activation=tf.nn.relu)
drop2 = tf.layers.dropout(conv2, keep_prob, training=train)
pool2 = tf.layers.max_pooling2d(drop2, 2, 2)
conv3 = tf.layers.conv2d(pool2, num_filters, filter_size, padding='same', activation=tf.nn.relu)
pool3 = tf.layers.max_pooling2d(conv3, 2, 2)
conv4 = tf.layers.conv2d(pool3, num_filters, filter_size, padding='same', activation=tf.nn.relu)
drop3 = tf.layers.dropout(conv4, keep_prob, training=train)

# Flatten input data
flatten = tf.reshape(drop3, [-1, 512])

# Create connected layers
with tf.contrib.framework.arg_scope(
    [tf.contrib.layers.fully_connected],
    normalizer_fn=tf.contrib.layers.batch_norm,
    normalizer_params={'is_training': train}):
        fc1 = tf.contrib.layers.fully_connected(flatten, 512)
        fc2 = tf.contrib.layers.fully_connected(fc1, num_categories, activation_fn=None)

# Compute loss
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        logits=fc2, labels=lbl_holder))

# Create optimizer
learning_rate = 0.0005
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()

# Launch session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True 
with tf.Session(config = config) as sess:
# with tf.Session() as sess:
    sess.run(init)

    p_epoch = ProgressBar(max_value = num_epochs, desc="Epoch: ", ascii = True)
    
    # Loop over epochs
    for epoch in range(num_epochs):

        # Loop over batches
        for batch in range(num_batches):
            batch_start = random.randint(0, batch_size*(num_batches-1)-1)
            batch_end = batch_start + batch_size
            img_batch = train_data[batch_start:batch_end, :]
            lbl_batch = train_labels[batch_start:batch_end, :]
            sess.run(optimizer, feed_dict={img_holder: img_batch,
                                           lbl_holder: lbl_batch, train: True})

        p_epoch.update(1)
    p_epoch.close()

    # Determine success rate
    print(">>>> Determine the success rate on test data:")
    prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(lbl_holder, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    print('\tAccuracy: ', sess.run(accuracy, feed_dict={img_holder: test_data,
                                                      lbl_holder: test_labels, train: False}))

