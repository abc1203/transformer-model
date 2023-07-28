import tensorflow as tf
from tensorflow import math, convert_to_tensor
from keras.preprocessing.sequence import pad_sequences



class TestTransformer:
    def __init__(self, tokenizer, transformer_model, encoder_test, encoder_vocab_size, encoder_seq_len, decoder_test, decoder_vocab_size, decoder_seq_len, **kwargs):
        self.transformer_model = transformer_model
        self.tokenizer = tokenizer

        self.encoder_test = encoder_test
        self.encoder_vocab_size = encoder_vocab_size
        self.encoder_seq_len = encoder_seq_len
        self.decoder_test = decoder_test
        self.decoder_vocab_size = decoder_vocab_size
        self.decoder_seq_len = decoder_seq_len
    

    # convert output probabilites from model to sequences
    # (batch_size, seq_len, vocab_size) => (batch_size, seq_len)
    def convert_to_seq(self, transformer_output, target):
        output_seq = []

        for i in range(transformer_output.shape[0]):
            # replace vocab tensor with index of highest probability + padding
            sentence_seq = [tf.get_static_value(math.argmax(transformer_output[i][j])) if target[i][j]!=0 else 0 for j in range(transformer_output[1])]

            output_seq.append(sentence_seq)

        return convert_to_tensor(output_seq)
    
    
    # convert sequences into text
    # sequences have shape (batch_size, seq_len)
    def convert_to_text(self, tokenizer, sequence):
        text = tokenizer.sequence_to_texts(sequence)

        return text
    

    # calculate BLEU score for transformer model's prediction
    # both pred & target are in the form of text with shape (batch_size, seq_len)
    def calculate_BLEU(self, pred, target):
        return 0


