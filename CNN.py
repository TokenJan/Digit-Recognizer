import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf # tensor flow
from keras.utils.np_utils import to_categorical # one hot encoding

# configuration parameter
TRAINING_ITERATIONS = 1000
BATCH_SIZE = 128
LEARNING_RATE = 1e-4

def get_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def get_weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def get_bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# read training data from CSV file
dataset = pd.read_csv("./data/train.csv")
x_train = dataset.iloc[:,1:].values.astype('float32')
y_train = dataset.iloc[:,0].values.astype('float32')
y_train = to_categorical(y_train)
y_train = y_train.astype(np.uint8)

# data normalization
x_train = np.multiply(x_train, 1.0/255.0)

# fetch next batch func
epochs_completed = 0
index_in_epoch = 0
num_examples = y_train.shape[0]

def next_batch(batch_size):
    global x_train
    global y_train
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        x_train = x_train[perm]
        y_train = y_train[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
    end = index_in_epoch
    return x_train[start:end], y_train[start:end]

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])  # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])

## conv1 layer ##
W_conv1 = get_weight([5,5, 1,32]) # patch 5x5, in size 1, out size 32
b_conv1 = get_bias([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1) # output size 14x14x32

## conv2 layer ##
W_conv2 = get_weight([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = get_bias([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2) # output size 7x7x64

## fc1 layer ##
W_fc1 = get_weight([7*7*64, 1024])
b_fc1 = get_bias([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = get_weight([1024, 10])
b_fc2 = get_bias([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(TRAINING_ITERATIONS):
    batch_xs, batch_ys = next_batch(BATCH_SIZE)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(get_accuracy(batch_xs, batch_ys))

# read testing data from CSV file
x_test = pd.read_csv("./data/test.csv").values.astype('float32')

y_test = []
test_batch_size = 1000
num_test = x_test.shape[0]
predict = tf.argmax(prediction, 1)

for i in range(28):
    # convert from [0:255] => [0.0:1.0]
    x_test_batch = np.multiply(x_test, 1.0 / 255.0)
    x_test_batch = x_test[test_batch_size*i:test_batch_size*(i+1), :]

    # predict test set
    y_test_batch = sess.run(predict, feed_dict={xs: x_test_batch, keep_prob: 1.0})
    y_test = np.concatenate([y_test, y_test_batch])

# save results
np.savetxt('submission.csv',
           np.c_[range(1, len(x_test) + 1), y_test],
           delimiter=',',
           header='ImageId,Label',
           comments='',
           fmt='%d')

sess.close()
