from tensorflow.keras.layers import Layer, Dropout
from MultiheadAttention import MultiheadAttention
from PositionwiseFeedForwardNetwork import PositionwiseFeedForwardNetwork
from LayerNorm import LayerNorm


class TransformerEncoderLayer(Layer):
    """
    the TransformerEncoderLayer represents a single layer in the transformer encoder
    the layer takes in embedded inputs after TransformerEmbedding has been performed
    the layer contains the following components:
        1a. multihead attention, where the input is the queries, keys, and values
        1b. a residual dropout (with rate = 0.1) is applied to the output of the attention mechanism
        1c. apply LayerNorm(x + Sublayer(x)), where x = the input and Sublayer = the output from 1b
        
        2a. feed forward network as implemented in PositionwiseFeedForwardNetwork, which takes the output of 1c as input
        2b. a residual dropout (with rate = 0.1) is applied to the output of the FFN
        2c. apply LayerNorm(x + Sublayer(x)), where x = the output from 1c and Sublayer = the output from 2b
    """
    
    
    def __init__(self, h = 8, d_k = 64, d_v = 64, d_model = 512, d_ff = 2048, dropout_rate = 0.1, **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)
        self.multihead_attention = MultiheadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(dropout_rate)
        self.add_norm1 = LayerNorm()
        self.feed_forward = PositionwiseFeedForwardNetwork(d_ff, d_model)
        self.dropout2 = Dropout(dropout_rate)
        self.add_norm2 = LayerNorm()
    

    def call(self, embedded_input, is_masking = False, is_training = False):
        # 1a. go through multihead attention
        res1 = self.multihead_attention(embedded_input, embedded_input, embedded_input, is_masking)

        # 1b. apply dropout to the output of the attention
        res1 = self.dropout1(res1, is_training)

        # 1c. apply add & normalization
        res1 = self.add_norm1(embedded_input, res1)

        # 2a. feed forward the result from the 1st sublayer
        res2 = self.feed_forward(res1)

        # 2b. apply dropout to the output of the FFN
        res2 = self.dropout2(res2, is_training)

        # 2c. apply add & normalization
        res2 = self.add_norm2(res1, res2)

        return res2


