
import tensorflow_datasets as tfds
import os

data_dir = os.getcwd() + '\datasets'
print(data_dir)

ds = tfds.load('huggingface:wmt14/de-en', split='train', data_dir=data_dir, download=False)