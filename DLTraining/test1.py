from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


hello = tf.constant(tf.__version__)
sess = tf.compat.v1.Session()
#sess = tf.Session()
print(sess.run(hello))