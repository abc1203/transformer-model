from cgi import test
from copyreg import pickle
from json import decoder
from pickle import load, dump
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
        self.tmp_tokenizer = None
        self.encoder_tokenizer = None
        self.decoder_tokenizer = None
    

    def get_configs(self, data, train_test_split = 0.8, tokenizer = None, test_size = 10):
        training = False
        if tokenizer == None:
            training = True
            tokenizer = Tokenizer(oov_token='unk', filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
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
        test_data = tensor_data[int(split_idx):int(split_idx)+test_size]

        # save tokenizer for potential later use in testing
        self.tmp_tokenizer = tokenizer

        if training:
            return train_data, vocab_size, max_seq_len
        else:
            return test_data, vocab_size, max_seq_len


    def __call__(self, encoder_filename, decoder_filename, is_training = True):
        encoder_data = load(open(get_dir(encoder_filename), 'rb'))
        decoder_data = load(open(get_dir(decoder_filename), 'rb'))

        # take a subset of the dataset
        encoder_data = encoder_data[:self.dataset_size]
        decoder_data = decoder_data[:self.dataset_size]

        # add <eol> symbol to every line
        for i in range(self.dataset_size):
            encoder_data[i] = '<start> ' + encoder_data[i] + ' <eol>'
            decoder_data[i] = '<start> ' + decoder_data[i] + ' <eol>'

        # shuffle the dataset
        idx_arr = np.arange(len(encoder_data))
        np.random.shuffle(idx_arr)
        encoder_data = np.array(encoder_data)[idx_arr.astype(int)]
        decoder_data = np.array(decoder_data)[idx_arr.astype(int)]

        if is_training:
            # obtain train configs from data
            encoder_inputs, encoder_vocab_size, encoder_seq_len = self.get_configs(encoder_data)
            self.encoder_tokenizer = self.tmp_tokenizer
            decoder_inputs, decoder_vocab_size, decoder_seq_len = self.get_configs(decoder_data)
            self.decoder_tokenizer = self.tmp_tokenizer

            print("Training data loaded")
            print("=======================================================================================================")

            # save tokenizers if training
            print("Saving tokenizers...")

            dump(self.encoder_tokenizer, open(get_dir('encoder_tokenizer.pkl'), 'wb'))
            dump(self.decoder_tokenizer, open(get_dir('decoder_tokenizer.pkl'), 'wb'))

            print("Tokenizers saved")
            print("=======================================================================================================")

            return encoder_inputs, encoder_vocab_size, encoder_seq_len, decoder_inputs, decoder_vocab_size, decoder_seq_len
        else:
            # load tokenizers
            self.encoder_tokenizer = load_tokenizer(get_dir('encoder_tokenizer.pkl'))
            self.decoder_tokenizer = load_tokenizer(get_dir('decoder_tokenizer.pkl'))

            # obtain test configs from data
            encoder_test, encoder_vocab_size, encoder_seq_len = self.get_configs(encoder_data, tokenizer=self.encoder_tokenizer)
            decoder_test, decoder_vocab_size, decoder_seq_len = self.get_configs(decoder_data, tokenizer=self.decoder_tokenizer)

            print("Test data loaded")
            print("=======================================================================================================")

            return encoder_test, encoder_vocab_size, encoder_seq_len, decoder_test, decoder_vocab_size, decoder_seq_len


