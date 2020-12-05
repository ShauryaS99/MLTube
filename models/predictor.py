import tensorflow as tf
from models.convnets import ConvolutionalNet
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import sequence
from models.preprocess_text import clean
import sys
import string 
import re

class Predictor (object):
    def __init__(self, model_path, vocab_path):
        MATCH_MULTIPLE_SPACES = re.compile("\ {2,}")
        self.SEQUENCE_LENGTH = 20
        EMBEDDING_DIMENSION = 30

        self.UNK = "<UNK>"
        PAD = "<PAD>"

        vocabulary = open(vocab_path).read().split("\n")
        self.inverse_vocabulary = dict((word, i) for i, word in enumerate(vocabulary))

        model = ConvolutionalNet(vocabulary_size=len(vocabulary), embedding_dimension=EMBEDDING_DIMENSION, input_length=self.SEQUENCE_LENGTH)
        model.load_weights(model_path)
        self.model = model
    
    def predict (self, headline):
        #headline = headline.encode("ascii", "ignore")
        inputs = sequence.pad_sequences([self.words_to_indices(self.inverse_vocabulary, clean(headline).lower().split())], maxlen=self.SEQUENCE_LENGTH)
        clickbaitiness = self.model.predict(inputs)[0, 0]
        return clickbaitiness

    def words_to_indices(self, inverse_vocabulary, words):
        return [inverse_vocabulary.get(word, inverse_vocabulary[self.UNK]) for word in words]