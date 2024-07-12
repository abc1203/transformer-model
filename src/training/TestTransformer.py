import tensorflow as tf
from tensorflow import math, convert_to_tensor, data, transpose, TensorArray, int64, newaxis
from keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu
import numpy as np



class TestTransformer:
    def __init__(self, tokenizer, transformer_model, encoder_test, encoder_vocab_size, encoder_seq_len, decoder_test, decoder_vocab_size, decoder_seq_len, batch_size = 64, **kwargs):
        self.transformer_model = transformer_model
        self.tokenizer = tokenizer

        self.encoder_test = encoder_test
        self.encoder_vocab_size = encoder_vocab_size
        self.encoder_seq_len = encoder_seq_len
        self.decoder_test = decoder_test
        self.decoder_vocab_size = decoder_vocab_size
        self.decoder_seq_len = decoder_seq_len

        print("Encoder vocab size: ", self.encoder_vocab_size)
        print("Encoder seq len: ", self.encoder_seq_len)
        print("Decoder vocab size: ", self.decoder_vocab_size)
        print("Decoder seq len: ", self.decoder_seq_len)

        # batching the dataset
        self.data_test = data.Dataset.from_tensor_slices((self.encoder_test, self.decoder_test))
        self.data_test = self.data_test.batch(batch_size)

        print("Test Initialized")
        print("=======================================================================================================")
    

    # convert output probabilites from model to sequences
    # (batch_size, seq_len, vocab_size) => (batch_size, seq_len)
    def convert_to_seq(self, transformer_output, target):
        output_seq = []

        for i in range(transformer_output.shape[0]):
            # replace vocab tensor with index of highest probability + padding
            sentence_seq = [tf.get_static_value(math.argmax(transformer_output[i][j])) if target[i][j]!=0 else 0 for j in range(transformer_output.shape[1])]

            output_seq.append(sentence_seq)

        return convert_to_tensor(output_seq)
    
    
    # convert sequences into text
    # sequences have shape (batch_size, seq_len)
    def convert_to_text(self, tokenizer, sequence):
        # get rid of padding at the end of seq
        sequence_no_padding = []

        for i in range(sequence.shape[0]):
            sentence_length = sequence.shape[1]

            while sequence[i][sentence_length-1] == 0: sentence_length = sentence_length - 1

            sequence_no_padding.append(sequence[i][:sentence_length])

        text = tokenizer.sequences_to_texts(sequence_no_padding)

        return text
    

    # calculate BLEU score for transformer model's prediction
    def calculate_BLEU(self, pred, target):
        # split the sentences so that each word is an element
        pred_processed = [pred[i].split() for i in range(len(pred))]
        target_processed = [target[i].split() for i in range(len(target))]

        # calculate BLEU score for each sentence
        
        print([pred_processed[i] for i in range(len(pred_processed))])
        bleu_scores_batch = [sentence_bleu([target_processed[i]], pred_processed[i]) for i in range(len(target_processed))]

        return np.mean(bleu_scores_batch)

    

    def __call__(self):
        print("Start testing: ")
        bleu_scores = []

        for (data_testX, data_testY) in enumerate(self.data_test):
            # decoder input is shifted right
            encoder_inputs = data_testX[:]
            ref_outputs = data_testY[:]

            # convert <s> to tensor
            output_start = self.tokenizer.texts_to_sequences(["<s>"])
            output_start = convert_to_tensor(output_start[0], dtype=int64)
    
            # convert <eol> to tensor
            output_end = self.tokenizer.texts_to_sequences(["<eol>"])
            output_end = convert_to_tensor(output_end[0], dtype=int64)

            # prepare output
            decoder_output = TensorArray(dtype=int64, size=0, dynamic_size=True)
            decoder_output = decoder_output.write(0, output_start)

            # generate next word
            for i in range(self.decoder_seq_len):
                prediction = self.transformer(encoder_inputs, transpose(decoder_output.stack()), training=False)
    
                prediction = prediction[:, -1, :]
    
                # select prediction
                predicted_id = tf.argmax(prediction, axis=-1)
                predicted_id = predicted_id[0][newaxis]
    
                # write prediction to output
                decoder_output = decoder_output.write(i + 1, predicted_id)
    
                # break of <eol> outputted
                if predicted_id == output_end:
                    break
    
            pred = transpose(decoder_output.stack())[0].numpy()

            print("Evaluating...")

            # convert pred & decoder_outputs to text
            pred = self.convert_to_text(self.tokenizer, pred.numpy())
            target = self.convert_to_text(self.tokenizer, ref_outputs.numpy())

            # evalute BLEU score
            print("Calculating BLEU score...")

            bleu_score = self.calculate_BLEU(pred, target)
            bleu_scores.append(bleu_score)
            print(bleu_scores)

            print("=======================================================================================================")
        
        print("Avg BLEU score: ", math.reduce_mean(bleu_scores))
        print("=======================================================================================================")
        






