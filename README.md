#Tensor Flow cheat sheet
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

#### Evaluating a list of tensors
``` python
a1, a3 = sess.run([z1, z3])

 ```


