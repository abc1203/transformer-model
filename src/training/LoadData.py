from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow import convert_to_tensor, math
from numpy import random
from data_preprocessing_utils import *


class LoadData:
    """
    loading data from finalized pkl files (from data_processing_utils.py) into configs that can be fed into the transformer model
    calculates:
    - encoder_vocab_size
    - decoder_vocab_size
    - encoder_seq_len
    - decoder_seq_len
    """

    def __init__(self, dataset_size = 1000, **kwargs):
        self.dataset_size = dataset_size
    
    def get_configs(self, data, oov_token = 'unk'):
        tokenizer = Tokenizer(oov_token=oov_token)
        
        # encode data in the form of numeric sequences
        tokenizer.fit_on_texts(data)
        sequences = tokenizer.texts_to_sequences(data)

        # get seq len + vocab size
        max_seq_len, min_seq_len = get_sentences_length(sequences)
        vocab_size = len(tokenizer.word_index) + 1

        # padding + convert to tensor
        padded_sequences = pad_sequences(sequences, maxlen=max_seq_len, padding='post')
        tensor_data = convert_to_tensor(padded_sequences)

        return tensor_data, vocab_size, max_seq_len


    def __call__(self, encoder_filename, decoder_filename):
        encoder_data = load(open(get_dir(encoder_filename), 'rb'))
        decoder_data = load(open(get_dir(decoder_filename), 'rb'))

        # randomly select a starting index and take a subset of the dataset
        start_idx = random.randint(len(encoder_data) - self.dataset_size)
        print("start index: ", start_idx)

        encoder_data = encoder_data[start_idx:start_idx+self.dataset_size]
        decoder_data = decoder_data[start_idx:start_idx+self.dataset_size]
        # encoder_data = encoder_data[:self.dataset_size]
        # decoder_data = decoder_data[:self.dataset_size]

        # shuffle the dataset
        idx_arr = np.arange(len(encoder_data))
        np.random.shuffle(idx_arr)
        encoder_data = np.array(encoder_data)[idx_arr.astype(int)]
        decoder_data = np.array(decoder_data)[idx_arr.astype(int)]

        # obtain configs from data
        encoder_inputs, encoder_vocab_size, encoder_seq_len = self.get_configs(encoder_data, start_idx)
        decoder_inputs, decoder_vocab_size, decoder_seq_len = self.get_configs(decoder_data, start_idx)

        print("Data Loading Complete")
        print("=======================================================================================================")

        return encoder_inputs, encoder_vocab_size, encoder_seq_len, decoder_inputs, decoder_vocab_size, decoder_seq_len



# data = LoadData()
# encoder_inputs, encoder_vocab_size, encoder_seq_len, decoder_inputs, decoder_vocab_size, decoder_seq_len = data('english_updated.pkl', 'german_updated.pkl')

# print(encoder_inputs[0:5])
# print("Encoder vocab size: ", encoder_vocab_size)
# print("Encoder seq len: ", encoder_seq_len)
# print(decoder_inputs[0:5])
# print("Decoder vocab size: ", decoder_vocab_size)
# print("Decoder seq len: ", decoder_seq_len)
