from tensorflow.keras.layers import Layer, Softmax
from tensorflow import matmul, sqrt, cast, linalg, float32, ones, math, maximum
import numpy as np


class ScaledDotProductAttention(Layer):
    """
    Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
    Q, K, V shape = (batch_size, input_seq_length, d_k = d_v = 64)
    shape(output) = shape(V)
    """

    def __init__(self, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.softmax = Softmax()
    

    def get_mask(self, input, is_masking):
        # apply padding mask
        mask_mat = math.equal(input, 0)
        mask_mat = cast(mask_mat, float32)
        
        if is_masking:
            # apply lookahead mask to the existing padding mask
            max_seq_len = input.shape[2]
            mask_mat_lookahead = 1 - linalg.band_part(ones((max_seq_len, max_seq_len)), -1, 0)
            mask_mat = maximum(mask_mat, mask_mat_lookahead)

        return mask_mat


    def call(self, queries, keys, values, d_k, is_masking=False):
        # 1. Matmul Q & K^T
        res = matmul(queries, keys, transpose_b=True)

        # 2. scaling factor of sqrt(d_k)
        res /= sqrt(cast(d_k, float32))

        # 3. create a mask to be put into the softmax
        mask_mat = self.get_mask(res, is_masking)

        # 4. softmax
        # shape(res) doesn't change
        res = self.softmax(res, mask=mask_mat)

        # 5. multiply by V
        # shape(res) = (batch_size, input_seq_length, d_v)
        res = matmul(res, values)

        return res

