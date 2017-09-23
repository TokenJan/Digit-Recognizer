import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

np.random.seed(0)
tf.set_random_seed(0)

# configuration parameter
TRAINING_ITERATIONS = 100000
BATCH_SIZE = 128

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

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
        assert batch_size <= num_examples
    end = index_in_epoch
    return x_train[start:end], y_train[start:end]

# read training data from CSV file
dataset = pd.read_csv("./data/train.csv")
x_train = dataset.iloc[:,1:].values.astype('float32')
y_train = dataset.iloc[:,0].values.astype('float32')
y_train = dense_to_one_hot(y_train, y_train.size)
y_train = y_train.astype(np.uint8)

# read testing data from CSV file
x_test = pd.read_csv("./data/test.csv").values.astype('float32')

# define the model
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
x_ = tf.placeholder(tf.float32, [None, 784])
y = tf.nn.softmax(tf.matmul(x_, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

# loss func
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

epochs_completed = 0
index_in_epoch = 0
num_examples = y_train.size

# init
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# train
for i in range(1000):
    batch_xs, batch_ys = next_batch(128)
    sess.run(train_step, feed_dict={x_: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print("Accuarcy on Test-dataset: ", sess.run(accuracy, feed_dict={x_: x_train, y_: sess.run(y_train)}))
