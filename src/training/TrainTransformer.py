import sys
import os
sys.path.insert(0, os.getcwd() +'\src\model')

import tensorflow as tf
from tensorflow import data
from Transformer import Transformer
from LoadData import LoadData
from AdamOptimizer import AdamOptimizer



class TrainTransformer:
    def __init__(self, encoder_filename, decoder_filename, step_num, batch_size=64, **kwargs):
        # load dataset and related configs
        self.Dataset = LoadData()
        self.encoder_inputs, self.encoder_vocab_size, self.encoder_seq_len, self.decoder_inputs, self.decoder_vocab_size, self.decoder_seq_len = self.Dataset(encoder_filename, decoder_filename)

        print("Encoder vocab size: ", self.encoder_vocab_size)
        print("Encoder seq len: ", self.encoder_seq_len)
        print("Decoder vocab size: ", self.decoder_vocab_size)
        print("Decoder seq len: ", self.decoder_seq_len)
        
        # batching the dataset
        self.data_train = data.Dataset.from_tensor_slices((self.encoder_inputs, self.decoder_inputs))
        self.data_train = self.data_train.batch(batch_size)

        # load Transformer model
        self.Transformer = Transformer(self.encoder_vocab_size, self.decoder_vocab_size, self.encoder_seq_len, self.decoder_seq_len)

        # load optimizer
        self.AdamOptimizer = AdamOptimizer()
    

    def train_step(self, encoder_inputs, decoder_inputs, decoder_outputs):
        pred = self.Transformer(encoder_inputs, decoder_inputs, is_training = True)
        print(pred)
        return pred
    
    def __call__(self):
        for step, (data_trainX, data_trainY) in enumerate(self.data_train):
            encoder_inputs = data_trainX[1:]
            decoder_inputs = data_trainY[:-1]
            decoder_outputs = data_trainY[1:]

            print(tf.shape(encoder_inputs))

            self.train_step(encoder_inputs, decoder_inputs, decoder_outputs)

            break


train_transformer = TrainTransformer('english_updated.pkl', 'german_updated.pkl', 25)
train_transformer()
