from tensorflow.keras.layers import Layer
from tensorflow import matmul, sqrt, softmax, cast, float32
import numpy as np


class ScaledDotProductAttention(Layer):
    """
    Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
    Q, K, V shape = (batch_size = 64, input_seq_length = 5, d_k = d_v = 64)
    shape(output) = shape(V)
    """

    def __init__(self, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def call(self, queries, keys, values, d_k, masking=False):
        # 1. Matmul Q & K^T
        # res has shape(64, 5, 5)
        res = matmul(queries, keys, transpose_b=True)

        # 2. scaling factor of sqrt(d_k)
        res /= sqrt(cast(d_k, float32))

        # 2a. optional: set all values inside softmax to -inf when deemed fit
        if masking:
            res = -np.inf

        # 3. softmax
        # shape(resMat) doesn't change
        res = softmax(res)

        # 4. multiply by V
        # shape(resMat) = (64, 5, 64) = (batch_size, input_seq_length, d_v)
        res = matmul(res, values)

        return res



