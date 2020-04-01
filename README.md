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


