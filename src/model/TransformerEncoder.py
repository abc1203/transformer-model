from tensorflow.keras.layers import Layer, Dropout
from TransformerEmbedding import TransformerEmbedding
from TransformerEncoderLayer import TransformerEncoderLayer
from PositionEmbeddingFixedWeights import PositionEmbeddingFixedWeights


class TransformerEncoder(Layer):
    """
    the Transformer Encoder implemented
    the encoder consists of 3 main components:
        1. the TransformerEmbedding, which performs input embedding & positional encoding on the input (which is an int vector)
        2. Dropout layer to the output of the TransformerEmbedding
        3. N = 6 (default) TransformerEncoderLayers, whose structure is specified in TransformerEncoderLayer.py
    the resulting output is a tensor with shape (batch_size, max_seq_len, d_model)
    """

    def __init__(self, vocab_size, max_seq_len, N = 6,
            h = 8, d_k = 64, d_v = 64, d_model = 512, d_ff = 2048, dropout_rate = 0.1, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.transformer_embedding = TransformerEmbedding(vocab_size, max_seq_len, d_model)
        self.pos_encoding = PositionEmbeddingFixedWeights(max_seq_len, vocab_size, d_model)
        self.transformer_encoder_layers = [TransformerEncoderLayer(h, d_k, d_v, d_model, d_ff, dropout_rate) for _ in range(N)]
        self.dropout = Dropout(dropout_rate)
    

    # def call(self, inputs, is_training = False):
    #     # perform input embedding & positional encoding onto the inputs
    #     res = self.transformer_embedding(inputs)

    #     # apply dropout to the sum of the embeddings and the positional encodings
    #     res = self.dropout(res, is_training)

    #     # put the resulting output into the encoder layers (N = 6 by default)
    #     for i, transformer_encoder_layer in enumerate(self.transformer_encoder_layers):
    #         res = transformer_encoder_layer(res, is_training)
        
    #     return res

    def call(self, input_sentence, padding_mask, training):
        # Generate the positional encoding
        pos_encoding_output = self.pos_encoding(input_sentence)
        # Expected output shape = (batch_size, sequence_length, d_model)
 
        # Add in a dropout layer
        x = self.dropout(pos_encoding_output, training=training)
 
        # Pass on the positional encoded values to each encoder layer
        for i, layer in enumerate(self.transformer_encoder_layers):
            x = layer(x, padding_mask, training)
 
        return x


