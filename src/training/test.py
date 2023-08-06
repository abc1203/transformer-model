import tensorflow as tf
from tensorflow import TensorArray, int64


decoder_output = tf.TensorArray(dtype=int64, size=0, dynamic_size=True)
decoder_output = decoder_output.write(0, 1)
print((decoder_output.stack())[0])