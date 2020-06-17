import tensorflow.compat.v1 as tf
from tf.compat.v1.nn.rnn_cell import LSTMCell
import numpy as np

tf.disable_eager_execution()

flags = tf.app.flags
FLAGS = flags.FLAGS

class Model(object):
    def __init__(self):
        # placeholder for storing rotated input images
        self.input_mesra = tf.placeholder(dtype=tf.float32,
                                                   shape=(None, FLAGS.len_mesra, FLAGS.len_dic))
        # placeholder for storing original images without rotation
        self.output_mesra = tf.placeholder(dtype=tf.float32,
                                                    shape=(None, FLAGS.len_mesra, FLAGS.len_dic))
        
        
        
    def encoder(self, inputs):
        x = tf.unstack(self.input_mesra, FLAGS.len_mesra, 1)
        lstm = LSTMCell(num_units = FLAGS.len_dic)
        outputs, state = tf.compat.v1.nn.dynamic_rnn(cell=lstm, 
                                                     inputs=x, 
                                                     dtype=tf.float32)        
        
    def decoder:
        
    