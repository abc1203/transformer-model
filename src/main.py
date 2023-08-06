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
    model_dir = os.getcwd() + '\saved_model_ckpts\\ckpt-5.index'

    # get test data from 2014 wmt english-german dataset
    data = LoadData(dataset_size=25000)
    encoder_test, encoder_vocab_size, encoder_seq_len, decoder_test, decoder_vocab_size, decoder_seq_len = data('english_updated.pkl', 'german_updated.pkl', is_training=False)

    # load pre-trained transformer model
    transformer_model = Transformer(encoder_vocab_size, decoder_vocab_size, encoder_seq_len, decoder_seq_len, dropout_rate=0)
    transformer_model.load_weights(model_dir)

    # evaluate model (TestTransformer)
    test = TestTransformer(transformer_model, encoder_test, encoder_vocab_size, encoder_seq_len, decoder_test, decoder_vocab_size, decoder_seq_len)
    test()







