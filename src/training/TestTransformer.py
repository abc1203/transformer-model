import tensorflow as tf
from tensorflow import math, convert_to_tensor, data, int64
from keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from utils import load_tokenizer, get_dir



class TestTransformer:
    def __init__(self, transformer_model, encoder_test, encoder_vocab_size, encoder_seq_len, decoder_test, decoder_vocab_size, decoder_seq_len, batch_size = 64, **kwargs):
        self.transformer_model = transformer_model
        self.encoder_tokenizer = load_tokenizer(get_dir('encoder_tokenizer.pkl'))
        self.decoder_tokenizer = load_tokenizer(get_dir('decoder_tokenizer.pkl'))

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

        # Prepare the output <START> token by tokenizing, and converting to tensor
        self.output_start = self.decoder_tokenizer.texts_to_sequences(['<start>'])
        self.output_start = convert_to_tensor(self.output_start[0], dtype=int64)
 
        # Prepare the output <EOS> token by tokenizing, and converting to tensor
        self.output_end = self.decoder_tokenizer.texts_to_sequences(['<eol>'])
        self.output_end = convert_to_tensor(self.output_end[0], dtype=int64)

        print("index for <start>: ", tf.get_static_value(self.output_start))
        print("index for <eol>: ", tf.get_static_value(self.output_end))

        # batching the dataset
        self.data_test = data.Dataset.from_tensor_slices((self.encoder_test, self.decoder_test))
        self.data_test = self.data_test.batch(batch_size)

        print("Test Initialized")
        print("=======================================================================================================")
    

    def predict_sentence(self, sentenceX, max_repeat=5):
        # Prepare the output array of dynamic size
        decoder_output = tf.TensorArray(dtype=int64, size=0, dynamic_size=True)
        decoder_output = decoder_output.write(0, self.output_start)
        
        for i in range(self.decoder_seq_len):
            # Predict an output token
            prediction = self.transformer_model(convert_to_tensor([sentenceX]), tf.transpose(decoder_output.stack()), is_training = False)
 
            prediction = prediction[:, -1, :]
 
            # Select the prediction with the highest score
            predicted_id = math.argmax(prediction, axis=-1)
            predicted_id = predicted_id[0][tf.newaxis]
 
            # Write the selected prediction to the output array at the next available index
            decoder_output = decoder_output.write(i + 1, predicted_id)

            # Break if an <eol> token is predicted
            if predicted_id == self.output_end: break

            # break if same word is repeated > threshold
            same_word = True
            for j in range(max_repeat):
                if i+1-j < 0 or (i+1-j >= 0 and (decoder_output.stack())[i+1] != (decoder_output.stack())[i+1-j]):
                    same_word = False
            if same_word: 
                decoder_output = decoder_output.write(i + 2, tf.get_static_value(self.output_end))
                break
 
        output = tf.transpose(decoder_output.stack())[0]
        output = output.numpy()
        print("output seq: ", output)
        
        return output
    
    
    # convert sequence into text
    def convert_to_text(self, tokenizer, sequence):
        sequence = np.asarray(sequence)
        
        # get rid of padding at the end of seq
        sequence_no_padding = []

        sentence_length = sequence.shape[0]
        if sentence_length <= 0:
            print("ERROR: sentence length <= 0")
            return

        while sentence_length >= 1 and sequence[sentence_length-1] == 0: sentence_length = sentence_length - 1
        sequence_no_padding = [sequence[:sentence_length]]

        text = tokenizer.sequences_to_texts(sequence_no_padding)

        return text


    def __call__(self):
        print("Starting test: ")
        bleu_scores = []

        for step, (data_testX, data_testY) in enumerate(self.data_test):
            bleu_scores_batch = []

            print("Evaluating batch...")
            print("=======================================================================================================")

            for idx in range(data_testX.shape[0]):
                # predict sentence; returns a sequence
                print("To translate: ", self.convert_to_text(self.encoder_tokenizer, data_testX[idx]))
                pred_seq = self.predict_sentence(data_testX[idx])

                # calculate BLEU score for sentence
                pred = self.convert_to_text(self.decoder_tokenizer, pred_seq)
                target = self.convert_to_text(self.decoder_tokenizer, data_testY[idx])

                print("target: ", target)
                print("pred: ", pred)

                bleu_score = sentence_bleu([target], pred)
                print("BLEU score: ", bleu_score)
                bleu_scores_batch.append(bleu_score)
                print("=======================================================================================================")

            bleu_scores.append(np.mean(bleu_scores_batch))
            print("BLEU score for batch: ", np.mean(bleu_scores_batch))

            print("Batch done")
            print("=======================================================================================================")
        
        print("Avg BLEU score: ", math.reduce_mean(bleu_scores))
        print("=======================================================================================================")

