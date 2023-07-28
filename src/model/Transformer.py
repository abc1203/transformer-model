from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras import Model
from TransformerEncoder import TransformerEncoder
from TransformerDecoder import TransformerDecoder


class Transformer(Model):
    """
    the complete Transformer Model implemented
    the transformer model makes use of the encoder & decoder to compute tensors of shape = (batch_size, decoder_seq_len, d_model)
    a final linear transformation is applied on the outputs, before softmaxing to obtain output probabilities
    """

    def __init__(self, encoder_vocab_size, decoder_vocab_size, encoder_seq_len, decoder_seq_len,
            N = 6, h = 8, d_k = 64, d_v = 64, d_model = 512, d_ff = 2048, dropout_rate = 0.1, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.transformer_encoder = TransformerEncoder(encoder_vocab_size, encoder_seq_len, N, h, d_k, d_v, d_model, d_ff, dropout_rate)
        self.transformer_decoder = TransformerDecoder(decoder_vocab_size, decoder_seq_len, N, h, d_k, d_v, d_model, d_ff, dropout_rate)
        self.linear = Dense(decoder_vocab_size)
        self.softmax = Softmax()
    

    def call(self, encoder_inputs, decoder_inputs, is_training = False):
        # encoder
        encoder_outputs = self.transformer_encoder(encoder_inputs, is_training)

        # decoder
        decoder_outputs = self.transformer_decoder(decoder_inputs, encoder_outputs, is_training)

        # apply linear layer
        res = self.linear(decoder_outputs)

        # apply softmax
        res = self.softmax(res)

        return res


# from numpy import random

# enc_vocab_size = 20 # Vocabulary size for the encoder
# dec_vocab_size = 20 # Vocabulary size for the decoder
 
# enc_seq_length = 5  # Maximum length of the input sequence
# dec_seq_length = 5  # Maximum length of the target sequence

# batch_size = 64 
# h = 8  # Number of self-attention heads
# d_k = 64  # Dimensionality of the linearly projected queries and keys
# d_v = 64  # Dimensionality of the linearly projected values
# d_ff = 2048  # Dimensionality of the inner fully connected layer
# d_model = 512  # Dimensionality of the model sub-layers' outputs
# n = 6  # Number of layers in the encoder stack
 
# dropout_rate = 0.1  # Frequency of dropping the input units in the dropout layers
 
# # Create model
# training_model = Transformer(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length)

# input_seq = random.random((batch_size, enc_seq_length))
# output_seq = random.random((batch_size, dec_seq_length))

# print(training_model(input_seq, output_seq))
