import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np

## Fancier graph with variables
rand_matrix = np.random.randint(0,10,(10,10))

x = tf.Variable(np.zeros((10,10)), dtype=tf.float32, name="A")
A = tf.constant(rand_matrix, dtype=tf.float32)
output = tf.matmul(x, A)
id_mat = tf.constant(np.identity(10), dtype=tf.float32)
loss = tf.reduce_sum(tf.abs(output - id_mat))

# Here we define some "meta-graph" stuff
optimiser = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(loss)
init_op = tf.global_variables_initializer()

# Now we do the actual computation part
epochs = 100_000

with tf.Session() as sess:
    sess.run(init_op)
    for e in range(epochs):
        _, acc = sess.run([optimiser, loss])
        if e % 100 == 0:
            print(acc)

    x_ = sess.run(x)
    A_ = sess.run(A)
    print(x_.dot(A_))

