import sys
import os
sys.path.insert(0, os.getcwd() +'\src\model')

import tensorflow as tf
from tensorflow import data, math, cast, float32, one_hot, shape
from tensorflow.keras.losses import CategoricalCrossentropy
from Transformer import Transformer
from LoadData import LoadData
from AdamOptimizer import AdamOptimizer



class TrainTransformer:
    def __init__(self, encoder_filename, decoder_filename, batch_size=64, **kwargs):
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

        # load loss function with label smoothing of epsilon = 0.1
        self.LossFunction = CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)

        print("Transformer Initialized")
        print("=======================================================================================================")

    
    def get_loss(self, pred, target, target_vocab_size):
        # create padding mask so zeros will not be included
        # shape(mask_mat) = (batch_size, seq_len)
        mask_mat = math.logical_not(math.equal(target, 0))
        mask_mat = cast(mask_mat, float32)
        print(shape(mask_mat))

        # apply one-hot encoding on target
        # shape(target) = (batch_size, seq_len, vocab_size)
        target = one_hot(target, depth=target_vocab_size)
        loss = self.LossFunction(target, pred)
    

    def train_step(self, encoder_inputs, decoder_inputs, decoder_outputs):
        pred = self.Transformer(encoder_inputs, decoder_inputs, is_training = True)
        print(pred[0][pred.shape[1]-1])
        print(math.reduce_sum(pred[0][pred.shape[1]-1]))
        return pred
    
    def __call__(self):
        print("Starting Training: ")
        
        # train by batch
        for step, (data_trainX, data_trainY) in enumerate(self.data_train):
            # decoder input is shifted right
            encoder_inputs = data_trainX[1:]
            decoder_inputs = data_trainY[:-1]
            decoder_outputs = data_trainY[1:]
            print(decoder_outputs[0])

            pred = self.train_step(encoder_inputs, decoder_inputs, decoder_outputs)

            self.get_loss(pred, decoder_outputs, self.decoder_vocab_size)

            break

        print("Training Complete")


train_transformer = TrainTransformer('english_updated.pkl', 'german_updated.pkl')
train_transformer()
