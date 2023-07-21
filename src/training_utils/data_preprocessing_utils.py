from pickle import dump, load
from collections import Counter
from turtle import update
import numpy as np
import re
import string
import unicodedata
import os


"""
a number of utility functions for data preprocessing

the wmt14 de-en dataset is used to train the transformer model
to load the dataset from huggingface locally:
import tensorflow_datasets as tfds
ds = tfds.load('huggingface:wmt14/de-en', split='train', download=False)
"""


DATA_DIR = os.getcwd() + '\data'

# return directory of target file
def get_dir(filename):
    return DATA_DIR + "\\" + filename


# read from data file
def get_file_content(filename):
    file = open(filename, mode='rt', encoding='utf-8')
    file_content = file.read()
    file.close()

    return file_content


# convert file content to list with element for each line
def get_sentences(file_content):
    return file_content.strip().split('\n')


# get the longest and shortest sentence length
def get_sentences_length(sentences):
    max_len, min_len = -np.inf, np.inf

    for sentence in sentences:
        if len(sentence) > max_len: max_len = len(sentence)
        elif len(sentence) < min_len: min_len = len(sentence)

    return max_len, min_len


# remove non-printable chars, punctations, words that contain numbers
def get_clean_sentences(sentences):
        clean_sentences = []
        # get regex pattern for printable chars
        re_print = re.compile('[^%s]' % re.escape(string.printable))
        # table for removing punctuation
        punctation_table = str.maketrans('', '', string.punctuation)
        
        for sentence in sentences:
                # normalize unicode characters
                sentence = unicodedata.normalize('NFD', sentence).encode('ascii', 'ignore')
                sentence = sentence.decode('UTF-8')
                
                # split sentence into words separated by whitespace
                sentence = sentence.split()

                for word in sentence:
                    # convert to lower case + remove punctation
                    word = word.lower().translate(punctation_table)

                    # remove non-printable chars
                    word = re_print.sub('', word)

                # remove all words that contains numbers
                sentence = [word for word in sentence if word.isalpha()]
                # add space to the cleaned words
                clean_sentences.append(' '.join(sentence))
        
        return clean_sentences


# load raw data files into .pkl files
def get_pkl_file(data_filename, output_filename):
    file_content = get_file_content(get_dir(data_filename))
    sentences = get_sentences(file_content)
    
    max_len, min_len = get_sentences_length(sentences)
    print(data_filename + ' Data INFO: sentences=%d, min=%d, max=%d' % (len(sentences), min_len, max_len))
    
    clean_sentences = get_clean_sentences(sentences)
    
    # get pkl file from the cleaned sentences
    output_file = open(get_dir(output_filename), 'wb')
    dump(clean_sentences, output_file)

    output_file.close()
    print(output_filename," saved")


# ==============================================================================================================
# further process generated pkl files with updated vocab & OOV tokens


# load dataset from generated pkl file
def load_clean_sentences(filename):
    return load(open(get_dir(filename), 'rb'))


# get all vocab above a certain occurrence number into a list
def get_vocab(sentences, min_occurrence = 5):
    vocab = Counter()

    # get all vocab
    for sentence in sentences:
        word = sentence.split()
        vocab.update(word)
    
    # remove vocab with low occurrence
    updated_tokens = [k for k,c in vocab.items() if c >= min_occurrence]
    return vocab, set(updated_tokens)


# mark all out-of-vocab words with "unk" token
def mark_OOV(sentences, vocab):
    new_sentences = []
    for sentence in sentences:
        new_sentence = []
        for token in sentence.split():
                if token in vocab:
                    new_sentence.append(token)
                else:
                    new_sentence.append('unk')
        new_sentence = ' '.join(new_sentence)
        new_sentences.append(new_sentence)

    return new_sentences

 
# save a list of clean sentences to pkl file
def save_clean_sentences(sentences, filename):
    dump(sentences, open(get_dir(filename), 'wb'))
    print('Saved: %s' % filename)


# further process the data and get finalized .pkl files
def get_finalized_pkl_file(pkl_filename):
    sentences = load_clean_sentences(pkl_filename)
    
    # process the vocab size
    all_vocab, updated_vocab = get_vocab(sentences, min_occurrence=5)
    print('All Vocabulary Size: %d' % len(all_vocab))
    print('Updated Vocabulary Size: %d' % len(updated_vocab))

    # mark OOV words & save the updated sentences into .pkl files
    updated_sentences = mark_OOV(sentences, updated_vocab)
    output_pkl_filename = pkl_filename.split('.')[0] + '_updated.pkl'
    save_clean_sentences(updated_sentences, output_pkl_filename)

    # print head of updated sentences
    for i in range(20):
        print("line",i,":",updated_sentences[i])
    

