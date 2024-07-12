import sys
import os

import tensorflow as tf
sys.path.insert(0, os.getcwd() + '\src\model')
from Transformer import Transformer
sys.path.insert(0, os.getcwd() + '\src\\training')
from LoadData import LoadData
from TestTransformer import TestTransformer


"""
load the deep learning transformer model for english-german translation task
model is pretrained from a subset of the 2014 WMT English-German dataset
"""

if __name__ == "__main__":
    model_dir = os.getcwd() + '\saved_model\\transformer_model'

    # get test data from 2014 wmt english-german dataset
    data = LoadData(dataset_size=2000)
    encoder_test, encoder_vocab_size, encoder_seq_len, decoder_test, decoder_vocab_size, decoder_seq_len = data('english_updated.pkl', 'german_updated.pkl', is_training=False)
    tokenizer = data.get_tokenizer()

    # load pre-trained transformer model
    transformer_model = tf.keras.models.load_model(model_dir)
    transformer_model.summary()

    # evaluate model (TestTransformer)
    test = TestTransformer(tokenizer, transformer_model, encoder_test, encoder_vocab_size, encoder_seq_len, decoder_test, decoder_vocab_size, decoder_seq_len)
    test()







