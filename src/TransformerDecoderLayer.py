from tensorflow.keras.layers import Layer, Dropout
from MultiheadAttention import MultiheadAttention
from PositionwiseFeedForwardNetwork import PositionwiseFeedForwardNetwork
from LayerNorm import LayerNorm


class TransformerDecoderLayer(Layer):
    """
    the TransformerDecoderLayer represents a single layer in the transformer decoder
    the layer takes in embedded outputs after TransformerEmbedding has been performed
    the layer contains the following components:
        1a. masked multihead attention, where the input is the queries, keys, and values
             - the mask prevents positions from attending to subsequent positions
        1b. a residual dropout (with rate = 0.1) is applied to the output of the attention mechanism
        1c. apply LayerNorm(x + Sublayer(x)), where x = embedded output and Sublayer(x) = output from 1b

        2a. non-masked multihead attention, where queries = output from 1c, keys = values = encoder output
        2b. a residual dropout (with rate = 0.1) identical to 1b
        2c. apply LayerNorm(x + Sublayer(x)), where x = output from 1c and Sublayer(x) = output from 2b
        
        3a. feed forward network as implemented in PositionwiseFeedForwardNetwork, which takes in the output from 2c
        3b. a residual dropout (with rate = 0.1) identical to 1b
        3c. apply LayerNorm(x + Sublayer(x)), where x = the output from 2c and Sublayer(x) = the output from 3b
    """

    def __init__(self, h = 8, d_k = 64, d_v = 64, d_model = 512, d_ff = 2048, dropout_rate = 0.1, **kwargs):
        super(TransformerDecoderLayer, self).__init__(**kwargs)
        self.multihead_attention1 = MultiheadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(dropout_rate)
        self.add_norm1 = LayerNorm()
        self.multihead_attention2 = MultiheadAttention(h, d_k, d_v, d_model)
        self.dropout2 = Dropout(dropout_rate)
        self.add_norm2 = LayerNorm()
        self.feed_forward = PositionwiseFeedForwardNetwork(d_ff, d_model)
        self.dropout3 = Dropout(dropout_rate)
        self.add_norm3 = LayerNorm()
    
    def call(self, embedded_output, encoder_output, is_masking = False, is_training = False):
        # 1a. go through masked multihead attention
        res1 = self.multihead_attention1(embedded_output, embedded_output, embedded_output, is_masking=is_masking)

        # 1b. apply dropout
        res1 = self.dropout1(res1, is_training)

        # 1c. apply add & normalization
        res1 = self.add_norm1(embedded_output, res1)

        # 2a. go though 2nd nonmasked multihead attention
        res2 = self.multihead_attention2(res1, encoder_output, encoder_output, is_masking)

        # 2b. apply dropout
        res2 = self.dropout2(res2, is_training)

        # 2c. apply add & normalization
        res2 = self.add_norm2(res1, res2)

        # 3a. feed forward the result from the 2nd sublayer
        res3 = self.feed_forward(res2)

        # 3b. apply dropout
        res3 = self.dropout3(res3, is_training)

        # 3c. apply add & normalization
        res3 = self.add_norm3(res2, res3)

        return res3

