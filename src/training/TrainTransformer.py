import sys
import os
sys.path.insert(0, os.getcwd() + '\src\model')

import tensorflow as tf
from tensorflow import data, math, cast, float32, one_hot, shape, convert_to_tensor
from tensorflow.keras.losses import CategoricalCrossentropy
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
        self.LossFunction = CategoricalCrossentropy(from_logits=False, label_smoothing=0.1, reduction="none")

        print("Transformer Initialized")
        print("=======================================================================================================")

    
    def get_loss(self, pred, target, target_vocab_size):
        # create padding mask so zeros will not be included
        # shape(mask_mat) = (batch_size, seq_len)
        mask_mat = math.logical_not(math.equal(target, 0))
        mask_mat = cast(mask_mat, float32)

        # apply one-hot encoding on target
        # shape(target_one_hot) = (batch_size, seq_len, vocab_size)
        # shape(loss) = (batch_size, seq_len)
        target_one_hot = one_hot(target, depth=target_vocab_size)
        loss = self.LossFunction(target_one_hot, pred)

        # apply masking 
        loss_masked = []
        for i in range(loss.shape[0]):
            loss_item = [loss[i][j] * mask_mat[i][j] for j in range(loss.shape[1])]
            loss_masked.append(loss_item)
        loss_masked = convert_to_tensor(loss_masked)

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
            pred = self.Transformer(encoder_inputs, decoder_inputs, is_training=True)
            print(pred)
    
            # get loss value (in the form of a tensor)
            loss = self.get_loss(pred, decoder_outputs, self.decoder_vocab_size)
        
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
    

    def __call__(self, epoch_num = 50):
        print("Starting Training: ")

        # initialize loss summaries
        losses = []

        # train by epoch
        for epoch in range(epoch_num):
            print("Training epoch #", epoch+1, ": ")

            # train by batch
            for step, (data_trainX, data_trainY) in enumerate(self.data_train):
                # decoder input is shifted right
                encoder_inputs = data_trainX[1:]
                decoder_inputs = data_trainY[:-1]
                decoder_outputs = data_trainY[1:]

                loss = self.train_step(encoder_inputs, decoder_inputs, decoder_outputs).numpy()

            # record loss after each epoch
            losses.append(loss)
            print("Loss summary (one after each epoch): ")
            print(losses)
            
            print("Epoch ", epoch+1, " done")
            print("=======================================================================================================")

        
        print("Training Complete")
        print("=======================================================================================================")
        
        # save the trained model
        print("Saving model...")
        self.Transformer.save(os.getcwd() + '\saved_model\\transformer_model')
        print("Model saved")


train_transformer = TrainTransformer('english_updated.pkl', 'german_updated.pkl', dataset_size=10000)
train_transformer()
