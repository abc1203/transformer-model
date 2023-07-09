from tensorflow.keras.layers import Layer, Embedding
import numpy as np


class TransformerEmbedding(Layer):
    """
    the tranformer embedding is the sum of the following 2 components:
    1. the embedding layer, which takes in input & output tokens and output vectors of dimension d_model
    2. positional encoding, calculated by:
        PE_(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE_(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, vocab_size, max_seq_len, d_model=512, **kwargs):
        super(TransformerEmbedding, self).__init__(**kwargs)
        # embedding layer
        self.token_embedding = Embedding(vocab_size, d_model)
        # positional encoding
        self.positional_encoding = self.get_positional_encoding(max_seq_len, d_model)
    

    def get_positional_encoding(self, max_seq_len, d_model):
        pos_enc = np.zeros((max_seq_len, d_model))

        for pos in range(max_seq_len):
            for i in range(int(d_model/2)):
                pos_enc[pos, 2*i] = np.sin(pos / np.power(10000, 2*i/d_model))
                pos_enc[pos, 2*i+1] = np.cos(pos / np.power(10000, 2*i/d_model))
        
        return pos_enc
    

    def call(self, inputs):
        res = self.token_embedding(inputs) + self.positional_encoding

        return res


