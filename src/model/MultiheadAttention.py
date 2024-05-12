from tensorflow.keras.layers import Layer, Dense
from tensorflow import shape, reshape, transpose
from ScaledDotProductAttention import ScaledDotProductAttention



class MultiheadAttention(Layer):
    """
    MultiheadAttention(Q, K, V) = Concat(head_1, head_2, ..., head_h)W^O, where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V),
    W_i^Q, W_i^K, W_i^V, W^O are the parameter matrices

    To compute the result in parallel, we reshape the dimensionality of Q, K, V before feeding it to ScaledDotProductAttention

    Below are the calculation of tensor shapes (taking the default values):
        for input shape = (batch size, ..., input dim), output shape for dense layer = (batch size, ..., units)
        shape(queries) = (batch size, input seq length, d_k); this is the input_shape
        W_q = Dense(d_k) => units = d_k

        Calculating shapes of q, k, v:
        shape(W_q(queries)) = (batch size, input seq length, d_k)
        => after reshape(q, shape=[batch_size, q_len, h, -1]), shape = (batch_size, seq_len, h, d_k/h)
        => after transposing, shape = (batch_size, h, seq_len, d_k/h)
        => shape(q) = shape(k) = shape(v) = (batch_size, h, seq_len, d_k/h) before feeding into ScaledDotProductAttention

        Calculating shape of output:
        shape(o) = (64, 8, 5, 5) after ScaledDotProductAttention
        => shape(o) = (batch_size, h, seq_len, d_k/h) after transpose
        => shape(output) = (batch_size, seq_len, d_k)
    """
    
    def __init__(self, h = 8, d_k = 64, d_v = 64, d_model = 512, **kwargs):
        super(MultiheadAttention, self).__init__(**kwargs)
        self.heads = h
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        self.W_queries = Dense(d_k)
        self.W_keys = Dense(d_k)
        self.W_values = Dense(d_v)
        self.W_output = Dense(d_model)
        self.attention = ScaledDotProductAttention()
    

    def call(self, queries, keys, values, mask=None):
        # linear project the queries, keys, and values
        q = self.W_queries(queries)
        q = reshape(q, shape=[shape(q)[0], shape(q)[1], self.heads, -1])
        q = transpose(q, perm=[0, 2, 1, 3])

        k = self.W_keys(keys)
        k = reshape(k, shape=[shape(k)[0], shape(k)[1], self.heads, -1])
        k = transpose(k, perm=[0, 2, 1, 3])

        v = self.W_values(values)
        v = reshape(v, shape=[shape(v)[0], shape(v)[1], self.heads, -1])
        v = transpose(v, perm=[0, 2, 1, 3])

        # apply dot product attention
        o = self.attention(q, k, v, self.d_k, mask)

        # reshape the output tensor back to original
        o = transpose(o, perm=[0, 2, 1, 3])
        output = reshape(o, shape=[shape(o)[0], shape(o)[1], self.d_v])

        # linear project the output to have dimension d_model
        output = self.W_output(output)

        return output

    # def reshape_tensor(self, x, heads, flag):
    #     if flag:
    #         # Tensor shape after reshaping and transposing: (batch_size, heads, seq_length, -1)
    #         x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
    #         x = transpose(x, perm=(0, 2, 1, 3))
    #     else:
    #         # Reverting the reshaping and transposing operations: (batch_size, seq_length, d_k)
    #         x = transpose(x, perm=(0, 2, 1, 3))
    #         x = reshape(x, shape=(shape(x)[0], shape(x)[1], self.d_k))
    #     return x
 
    def call(self, queries, keys, values, mask=None):
        # Rearrange the queries to be able to compute all heads in parallel
        q_reshaped = self.reshape_tensor(self.W_queries(queries), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Rearrange the keys to be able to compute all heads in parallel
        k_reshaped = self.reshape_tensor(self.W_keys(keys), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Rearrange the values to be able to compute all heads in parallel
        v_reshaped = self.reshape_tensor(self.W_values(values), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Compute the multi-head attention output using the reshaped queries, keys and values
        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Rearrange back the output into concatenated form
        o = transpose(o_reshaped, perm=[0, 2, 1, 3])
        output = reshape(o, shape=[shape(queries)[0], shape(o)[1], self.d_v])
        # Resulting tensor shape: (batch_size, input_seq_length, d_v)
 
        # Apply one final linear projection to the output to generate the multi-head attention
        # Resulting tensor shape: (batch_size, input_seq_length, d_model)
        return self.W_output(output)

        
