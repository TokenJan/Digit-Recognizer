import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf # tensor flow
from keras.utils.np_utils import to_categorical # one hot encoding

# configuration parameter
TRAINING_ITERATIONS = 100000
BATCH_SIZE = 128
LEARNING_RATE = 0.001

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

# define the hidden layer
def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs

# define the placeholder
with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, [None, 784], name="x_input")
    ys = tf.placeholder(tf.float32, [None, 10], name="y_input")

# add hidden layer
y_predict = add_layer(xs, 784, 10,"l1", activation_function=tf.nn.softmax)

# the error between prediciton and real data
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(y_predict)))

# train the network
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/", sess.graph)

# initization
init = tf.global_variables_initializer()
sess.run(init)

display_step = 1
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

for i in range(TRAINING_ITERATIONS):
    batch_xs, batch_ys = next_batch(BATCH_SIZE, i)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        result = sess.run(merged, feed_dict={xs: batch_xs, ys: batch_ys})
        writer.add_summary(result, i)
    if i % display_step == 0 or (i + 1) == TRAINING_ITERATIONS:
        train_accuracy = sess.run(accuracy, feed_dict={xs: batch_xs, ys: batch_ys})
        print('training_accuracy => %.4f for step %d' % (train_accuracy, i))

        # increase display_step
        if i % (display_step * 10) == 0 and i:
            display_step *= 10

# read testing data from CSV file
x_test = pd.read_csv("./data/test.csv").values.astype('float32')

# convert from [0:255] => [0.0:1.0]
x_test = np.multiply(x_test, 1.0 / 255.0)

# predict test set
predict = tf.argmax(y_predict, 1)
y_test = sess.run(predict, feed_dict={xs: x_test})


# save results
np.savetxt('submission.csv',
           np.c_[range(1, len(x_test) + 1), y_test],
           delimiter=',',
           header='ImageId,Label',
           comments='',
           fmt='%d')

sess.close()
