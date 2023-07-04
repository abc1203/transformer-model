from tensorflow.keras.layers import Layer
from tensorflow import matmul, sqrt, softmax
import numpy as np


class ScaledDotProductAttention(Layer):
    # Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V

    def __init__(self, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def call(queries, keys, values, d_k, masking=False):
        # 1. Matmul Q & K
        resMat = matmul(queries, keys, transpose_b=True)

        # 2. scaling factor of sqrt(d_k)
        resMat /= sqrt(d_k)

        # 2a. optional: set all values inside softmax to -inf when deemed fit
        if masking:
            resMat = -np.inf

        # 3. softmax
        resMat = softmax(resMat)

        # 4. multiply by V
        resMat = matmul(resMat, values)

        return resMat



