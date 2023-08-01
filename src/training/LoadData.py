from pickle import load
from posixpath import split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow import convert_to_tensor, math
from numpy import random
from utils import *


class LoadData:
    """
    loading data from finalized pkl files (from data_processing_utils.py) into configs that can be fed into the transformer model
    calculates:
    - encoder_vocab_size
    - decoder_vocab_size
    - encoder_seq_len
    - decoder_seq_len
    """

    def __init__(self, dataset_size = 10000, **kwargs):
        self.dataset_size = dataset_size
        self.tokenizer = None
    

    def get_tokenizer(self):
        return self.tokenizer
    

    def get_configs(self, data, train_test_split = 0.8, oov_token = 'unk'):
        tokenizer = Tokenizer(oov_token=oov_token)
        
        # encode data in the form of numeric sequences
        tokenizer.fit_on_texts(data)
        sequences = tokenizer.texts_to_sequences(data)

        # get seq len + vocab size
        max_seq_len, min_seq_len = get_sentences_length(sequences)
        vocab_size = len(tokenizer.word_index) + 2

        # padding + convert to tensor
        padded_sequences = pad_sequences(sequences, maxlen=max_seq_len, padding='post')
        tensor_data = convert_to_tensor(padded_sequences)

        # train test split
        split_idx = self.dataset_size * train_test_split
        train_data = tensor_data[:int(split_idx)]
        test_data = tensor_data[int(split_idx):]

        # save tokenizer for potential later use in testing
        self.tokenizer = tokenizer

        return train_data, test_data, vocab_size, max_seq_len


    def __call__(self, encoder_filename, decoder_filename, is_training = True):
        encoder_data = load(open(get_dir(encoder_filename), 'rb'))
        decoder_data = load(open(get_dir(decoder_filename), 'rb'))

        # take a subset of the dataset
        encoder_data = encoder_data[:self.dataset_size]
        decoder_data = decoder_data[:self.dataset_size]

        # add <eol> symbol to every line
        for i in range(self.dataset_size):
            encoder_data[i] = encoder_data[i] + ' <eol>'
            decoder_data[i] = decoder_data[i] + ' <eol>'

        # shuffle the dataset
        idx_arr = np.arange(len(encoder_data))
        np.random.shuffle(idx_arr)
        encoder_data = np.array(encoder_data)[idx_arr.astype(int)]
        decoder_data = np.array(decoder_data)[idx_arr.astype(int)]

        # obtain configs from data
        # after these are called, self.tokenizer is the tokenizer for decoder data
        encoder_inputs, encoder_test, encoder_vocab_size, encoder_seq_len = self.get_configs(encoder_data)
        decoder_inputs, decoder_test, decoder_vocab_size, decoder_seq_len = self.get_configs(decoder_data)

        print("Data Loading Complete")
        print("=======================================================================================================")

        if is_training:
            return encoder_inputs, encoder_vocab_size, encoder_seq_len, decoder_inputs, decoder_vocab_size, decoder_seq_len
        return encoder_test, encoder_vocab_size, encoder_seq_len, decoder_test, decoder_vocab_size, decoder_seq_len


