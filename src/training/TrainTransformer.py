import sys
import os
sys.path.insert(0, os.getcwd() + '\src\model')

import tensorflow as tf
from tensorflow import data, math, cast, float32, one_hot, shape, convert_to_tensor, train
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from Transformer import Transformer
from LoadData import LoadData
from AdamOptimizer import AdamOptimizer



class TrainTransformer:
    def __init__(self, encoder_filename, decoder_filename, dataset_size=10000, batch_size=64, **kwargs):
        # load dataset and related configs
        self.Dataset = LoadData(dataset_size)
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
        self.LossFunction = SparseCategoricalCrossentropy(from_logits=True)

        print("Transformer Initialized")
        print("=======================================================================================================")

        
    def get_loss(self, pred, target):
        # create padding mask so zeros will not be included
        # shape(mask_mat) = (batch_size, seq_len)
        mask_mat = math.logical_not(math.equal(target, 0))
        mask_mat = cast(mask_mat, float32)

        # loss function
        loss = self.LossFunction(target, pred)

        # apply masking 
        loss_masked = loss * mask_mat

        # calculate mean loss
        mean_loss = math.reduce_sum(loss_masked) / math.reduce_sum(mask_mat)
        print("Mean Loss in current batch (None => graph execution): ", tf.get_static_value(mean_loss))

        # mean_loss is in the form of a one-element tensor
        return mean_loss


    # train step; returns the loss of the step
    # graph execution
    @tf.function
    def train_step(self, encoder_inputs, decoder_inputs, decoder_outputs, graph_execution = True):
        print("Training step...")
        with tf.GradientTape() as tape:
            pred = self.Transformer(encoder_inputs, decoder_inputs, training=True)
            print(pred)
    
            # get loss value (in the form of a tensor)
            loss = self.get_loss(pred, decoder_outputs)
        
        print("Updating weights...")
    
        # get gradients from loss & update the trainable weights
        gradients = tape.gradient(loss, self.Transformer.trainable_weights)
        self.AdamOptimizer.apply_gradients(zip(gradients, self.Transformer.trainable_weights))

        print("Done")
        print("Step finished")
        print("=============================================================================================")

        # return loss as a numeric value
        if graph_execution == False:
            return tf.get_static_value(loss)
        return loss
    

    def __call__(self, epoch_num = 25):
        print("Starting Training: ")

        # initialize loss summaries
        losses = []

        ckpt = train.Checkpoint(model=self.Transformer, optimizer=self.AdamOptimizer)
        ckpt_manager = train.CheckpointManager(ckpt, os.getcwd() + '\saved_model_ckpts', max_to_keep=1)

        # train by epoch
        for epoch in range(epoch_num):
            print("Training epoch #", epoch+1, ": ")

            # train by batch
            for step, (data_trainX, data_trainY) in enumerate(self.data_train):
                # decoder input is shifted right
                encoder_inputs = data_trainX[:, 1:]
                decoder_inputs = data_trainY[:, :-1]
                decoder_outputs = data_trainY[:, 1:]

                loss = self.train_step(encoder_inputs, decoder_inputs, decoder_outputs).numpy()

            # record loss after each epoch
            losses.append(loss)
            print("Loss summary (one after each epoch): ")
            print(losses)
            
            print("Epoch ", epoch+1, " done")
            print("=======================================================================================================")

            # save model after every 5 epochs
            if (epoch+1) % 5 == 0:
                print("Saving model...")
                save_path = ckpt_manager.save()
                print("Model saved")

        
        print("Training Complete")
        print("=======================================================================================================")
        

if __name__ == '__main__':
    train_transformer = TrainTransformer('english_updated.pkl', 'german_updated.pkl', dataset_size=10000)
    train_transformer()
