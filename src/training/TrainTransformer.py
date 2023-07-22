import sys
import os
sys.path.insert(0, os.getcwd() +'\src\model')

import Transformer
from LoadData import LoadData
from AdamOptimizer import AdamOptimizer



class TrainTransformer:
    def __init__(self, encoder_filename, decoder_filename, batch_size=2500, **kwargs):
        # load dataset and related configs
        self.Dataset = LoadData()
        self.encoder_inputs, self.encoder_vocab_size, self.encoder_seq_len, 
        self.decoder_inputs, self.decoder_vocab_size, self.decoder_seq_len = self.Dataset(encoder_filename, decoder_filename)

        # batching the dataset


        # load Transformer model
        self.Transformer = Transformer(self.encoder_vocab_size, self.decoder_vocab_size, self.encoder_seq_len, self.decoder_seq_len)

        # load optimizer
        self.AdamOptimizer = AdamOptimizer()


