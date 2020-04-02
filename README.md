# TensorFlow cheat sheet

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

Running the above code will create a new directory called 'summaries'.
To view the graph(which is not in human readable form) we use a Tensor board.
```python
from google.datalab.ml import TensorBoard
TensorBoard().start('./summaries')
```

Run the following command directly from CloudShell(online tool) to start TensorBoard

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

#### Batch size(Variable length tensors)
```python
n_input = tf.constant([3])
X = tf.placeholder(tf.float32, [None, n_input])
# Here, "None" tells us that there may be variable number of rows but a fixed size of 3 columns
```

#### Shape coercing(changing) can be done using these methods
1. `tf.reshape()` Takes the numbers we have and puts them into a different shape
2. `tf.expand_dims()` Way of changing a shape by inserting a dimension of 1 into a tensor
    * If x is a (3, 2) matrix. Calling `tf.expand_dims(x, 1)` will make the dimension of x as (3, 1, 2).
3. `tf.slice()` This is the actual method of `x[1, :]`
    * tf.slice(x, [0, 1], [2, 1]): slice x from (0, 1) element and take out two rows and one column.
4. `tf.squeeze()` Removes dimensions of size 1 from the shape of a tensor


#### Converting data types
`tf.cast()` is used to convert from one data type to the other.

Ex. `tf.cast(b, tf.float32)` converts b to data type float32.


#### Some methods to debug full blown programs
* `tf.Print()` is a way to print out the value of the tensors when specific conditions are met.
* **tfdbg** is an interactice debugger that you can run from the terminal that you can attach to a remote tensorflow session.
* **TensorBoard** is a visual monitoring tool.
* `tf.logging.set_verosity(tf.logging.INFO)` is used to change the logging output of tensorflow. Levels are `DEBUG`, `INFO`,`WARN`,`ERROR`,`FATAL`
    * 'DEBUG' is most quiet
    * 'FATAL' is most verbose

Ex. of `tf.Print()` statement. Let us say dividing a/b is causing nan to show up in the matrix. Hence we want to see the values of a 
and b that is causing this nan. One way we can do that is.
```python
s = a / b
print_ab = tf.Print(s, [a, b])
s = tf.where(tf.is_nan(s), print_ab, s)
```

This has to be done in a standalone program. Create another file and then execute this to see the error.


## Estimator API

#### Example of using a type of Estimator API 
```python
import tensorflow as tf

# We define the feature columns.
featcols = [
    tf.feature_column.numeric_column('sq_footage'),
    tf.feature_column.categorical_column_with_vocabulary_list('type', ["house", "apt"])
]
# Other columns include
# bucketized_column
# embedding_column
# crossed_column
# categorical_column_with_hash_bucket and so on

# We now chose a model to train the data from the premade Estimator API's
model = tf.estimator.LinearRegressor(featcols, '${dir_name_to_output_checkpoints}')
# Other estimators are
# DNNRegressor(Dense Neural Network)
# DNNLinearCombinedRegressor
# LinearClassifier
# DNNClassifier
# DNNLinearCombinedClassifier and so on

# We will now call train to train the model iteratively a 100 times.
model.train(train_input_fn, steps=100)

# We will predict the values for a given input
model.predict(predict_input_fn)

def train_input_fn():
    features = {
        "sq_footage": [1000, 2000, 3000, 1000, 2000, 3000],
        "type": ["house", "house", "house", "apt", "apt", "apt"]
    }

    labels = [500, 1000, 1500, 700, 1300, 1900]
    return features, labels

def predict_input_fn():
    features = {
        "sq_footage": [1500, 1800],
        "type": ["house", "apt"]
    }

    return features
```
