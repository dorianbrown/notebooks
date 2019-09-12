import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

# We'll put this into the graph later
rand_matrix = np.random.randint(0, 10, (10, 10))

# We start making our graph here
x = tf.Variable(np.zeros((10, 10)), dtype=tf.float32, name="A")
A = tf.constant(rand_matrix, dtype=tf.float32)
output = tf.matmul(x, A)
id_mat = tf.constant(np.identity(10), dtype=tf.float32)
loss = tf.reduce_sum(tf.abs(output - id_mat))

# Here we define some "meta-graph" stuff
optimiser = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(loss)
init_op = tf.global_variables_initializer()

# Number of passes over the data
epochs = 50_000

learning_graph = list()

# Context managers work nicely with tensorflow sessions
with tf.Session() as sess:
    sess.run(init_op)
    for e in range(epochs):
        _, epoch_loss = sess.run([optimiser, loss])
        if e % 100 == 0:
            learning_graph.append([e, epoch_loss])
            print(epoch_loss)

    # Lets evaluate the learned matrix A
    x_ = sess.run(x)
    print(x_.dot(rand_matrix))

import matplotlib.pyplot as plt

plt.plot([x[0] for x in learning_graph], [x[1] for x in learning_graph])
plt.ylabel('epoch')
plt.xlabel('loss')
plt.title('Model loss at each epoch')

plt.savefig("test.png")
