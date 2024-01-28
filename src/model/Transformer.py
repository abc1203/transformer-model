from tensorflow.keras.layers import Dense, Softmax
from tensorflow.keras import Model
from TransformerEncoder import TransformerEncoder
from TransformerDecoder import TransformerDecoder


# class Transformer(Model):
#     """
#     the complete Transformer Model implemented
#     the transformer model makes use of the encoder & decoder to compute tensors of shape = (batch_size, decoder_seq_len, d_model)
#     a final linear transformation is applied on the outputs
#     a final softmax layer is in the original model architecture; ditched because of dimensionality & accuracy issues
#     shape(output) = (batch_size, decoder_seq_len, decoder_vocab_size)
#     """

#     def __init__(self, encoder_vocab_size, decoder_vocab_size, encoder_seq_len, decoder_seq_len,
#             N = 6, h = 8, d_k = 64, d_v = 64, d_model = 512, d_ff = 2048, dropout_rate = 0.1, **kwargs):
#         super(Transformer, self).__init__(**kwargs)
#         self.transformer_encoder = TransformerEncoder(encoder_vocab_size, encoder_seq_len, N, h, d_k, d_v, d_model, d_ff, dropout_rate)
#         self.transformer_decoder = TransformerDecoder(decoder_vocab_size, decoder_seq_len, N, h, d_k, d_v, d_model, d_ff, dropout_rate)
#         self.linear = Dense(decoder_vocab_size)
#         # self.softmax = Softmax()
    

#     def call(self, encoder_inputs, decoder_inputs, is_training = False):
#         # encoder
#         encoder_outputs = self.transformer_encoder(encoder_inputs, is_training)

#         # decoder
#         decoder_outputs = self.transformer_decoder(decoder_inputs, encoder_outputs, is_training)

#         # apply linear layer
#         res = self.linear(decoder_outputs)

#         # apply softmax
#         # res = self.softmax(res)

#         return res

from tensorflow import math, cast, float32, linalg, ones, maximum, newaxis
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
class Transformer(Model):
    def __init__(self, enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length,
             N = 6, h = 8, d_k = 64, d_v = 64, d_model = 512, d_ff = 2048, dropout_rate = 0.1, **kwargs):
        super(Transformer, self).__init__(**kwargs)
 
        # Set up the encoder
        self.encoder = TransformerEncoder(enc_vocab_size, enc_seq_length, h, d_k, d_v, d_model, d_ff, N, dropout_rate)
 
        # Set up the decoder
        self.decoder = TransformerDecoder(dec_vocab_size, dec_seq_length, h, d_k, d_v, d_model, d_ff, N, dropout_rate)
 
        # Define the final dense layer
        self.model_last_layer = Dense(dec_vocab_size)
 
    def padding_mask(self, input):
        # Create mask which marks the zero padding values in the input by a 1.0
        mask = math.equal(input, 0)
        mask = cast(mask, float32)
 
        # The shape of the mask should be broadcastable to the shape
        # of the attention weights that it will be masking later on
        return mask[:, newaxis, newaxis, :]
 
    def lookahead_mask(self, shape):
        # Mask out future entries by marking them with a 1.0
        mask = 1 - linalg.band_part(ones((shape, shape)), -1, 0)
 
        return mask
 
    def call(self, encoder_input, decoder_input, training):
 
        # Create padding mask to mask the encoder inputs and the encoder outputs in the decoder
        enc_padding_mask = self.padding_mask(encoder_input)
 
        # Create and combine padding and look-ahead masks to be fed into the decoder
        dec_in_padding_mask = self.padding_mask(decoder_input)
        dec_in_lookahead_mask = self.lookahead_mask(decoder_input.shape[1])
        dec_in_lookahead_mask = maximum(dec_in_padding_mask, dec_in_lookahead_mask)
 
        # Feed the input into the encoder
        encoder_output = self.encoder(encoder_input, enc_padding_mask, training)
 
        # Feed the encoder output into the decoder
        decoder_output = self.decoder(decoder_input, encoder_output, dec_in_lookahead_mask, enc_padding_mask, training)
 
        # Pass the decoder output through a final dense layer
        model_output = self.model_last_layer(decoder_output)
 
        return model_output


