# Tensor Flow cheat sheet

These are notes that were written while following the coursera course, Intro to Tensor Flow

#### Evaluating a tensor
```python
import tensorflow as tf

x = tf.constant([3, 5, 7])
y = tf.constant([1, 2, 3])

z = tf.add([x, y])

with tf.Session() as sess:
    print z.eval()

```

#### Evaluating a tensor using the run command
```python
with tf.Session() as sess:
    print sess.run(z)
```

***

#### Evaluating a list of tensors
``` python
a1, a3 = sess.run([z1, z3])

 ```

#### Enabling Tensor Flow in eager mode

This is beneficial for debugging.
```python
from tensorflow.contrib.eager.python import tfe

tfe.enable_eager_execution()
# The above code must be called only once
# One can now continue with their code
```

#### Visualizing the graph

```python
x = tf.constant([3, 5, 7], name="x")
y = tf.constant([1, 2, 3], name="y")
# name="..." here names the tensors

z1 = tf.add([x, y], name="z1")
z2 = x * y
z3 = z2 - z1

with tf.Session() as sess:
    with tf.summary.FileWriter('summaries', sess.graph) as writer:
        # sess.graph here writes out the graph
        a1, a3 = sess.run([z1, z3])
```

Running the above code will create a new directory called 'summaries'
To view the graph(which is not in human readable form) we use a Tensor board.
```python
from google.datalab.ml import TensorBoard
TensorBoard().start('./summaries')
```

Run the following command directly from CloudShell to start TensorBoard

`tensorboard --port 8080 --logdir gs://${BUCKET}/${SUMMARY_DIR}`


#### Slicing tensors

```python
import tensorflow as tf
x = tf.constant([[3, 5, 7], [4, 6, 8]])

y = x[:, 1]
# Take all the rows(:) of the first column(1)
z = x[1, :]
# Take the 1st row(1) and all the columns(:)
a = x[1, 0:2]
# Take the 1st row(1) and the columns 0 through 1(0:2)

with tf.Session() as sess:
    print y.eval()
    print z.eval()
    print a.eval()

# output
# [5 6]
# [4, 6, 8]
# [4, 6]
# Remember that this is 0 indexed.
```

#### Reshaping tensors

```python
x = tf.constant([[3, 5, 7], [4, 6, 8]])
y = tf.reshape(x, [3, 2])

# print y.eval()
# [[3 5]
#  [7 4]
#  [6 8]]
```

#### Variables

```python
with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
    w = tf.get_variable("weights", shape = (1, 2), initializer = tf.truncate_normal_initializer(), trainable = True)
# variable name = weights
# shape = 1 row and 2 columns
# When w is initialized, it will be initialized as a truncated normal initializer(Gaussian normal distribution)
# Trainable, can be changed to non trainable (helps in freezing the graph)
# tf.AUTO_REUSE tell tensor flow to reuse the variable each time rather than creating a new one

# the following line is run inside the session part
tf.global_varialbe_initializer().run()
# Used to initialize all the variables
```

Placeholders allow one to feed in values to the graph, such as by reading from a text

```python
a = tf.placeholder("float", None)
b = a * 4

with tf.Session() as sess:
    print sess.run(b, feed_dict={a: [1, 2, 3]})

# feed_dict = here you must place a list or a numpy array of numbers for the placeholder a
```

