from tensorflow.keras.layers import Layer, Dropout
from TransformerEmbedding import TransformerEmbedding
from TransformerDecoderLayer import TransformerDecoderLayer


class TransformerDecoder(Layer):
    """
    the Transformer Decoder implemented
    the decoder consists of 3 main components:
        1. the TransformerEmbedding, which performs output embedding & positional encoding on the output (which is an int vector)
        2. Dropout layer to the output of the TransformerEmbedding
        3. N = 6 (default) TransformerDecoderLayers, whose structure is specified in TransformerDecoderLayer.py
    the resulting output is a tensor with shape (batch_size, max_seq_len, d_model)
    """

    def __init__(self, vocab_size, max_seq_len, N = 6,
            h = 8, d_k = 64, d_v = 64, d_model = 512, d_ff = 2048, dropout_rate = 0.1, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.transformer_embedding = TransformerEmbedding(vocab_size, max_seq_len, d_model)
        self.transformer_decoder_layers = [TransformerDecoderLayer(h, d_k, d_v, d_model, d_ff, dropout_rate) for _ in range(N)]
        self.dropout = Dropout(dropout_rate)
    

    def call(self, outputs, encoder_output, is_training = False):
        # perform output embedding & positional encoding onto the outputs
        res = self.transformer_embedding(outputs)

        # apply dropout to the sum of the embeddings and the positional encodings
        res = self.dropout(res, is_training)

        # put the resulting output into the encoder layers (N = 6 by default)
        for i, transformer_decoder_layer in enumerate(self.transformer_decoder_layers):
            res = transformer_decoder_layer(res, encoder_output, is_training)
        
        return res

