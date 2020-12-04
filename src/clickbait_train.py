import os
import numpy as np
from tensorflow.python.keras.layers import Embedding
from models.convnets import ConvolutionalNet
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from models.preprocess_text import clean

SEQUENCE_LENGTH = 20
EMBEDDING_DIMENSION = 30
MODEL_FILE = "models/youtube_detector.h5"



def words_to_indices(inverse_vocabulary, words):
    return [inverse_vocabulary[word] for word in words]

if __name__ == "__main__":

    vocabulary = open("data/vocabulary_youtube_porn.txt").read().split("\n")
    inverse_vocabulary = dict((word, i) for i, word in enumerate(vocabulary))


    clickbait = open("data/clickbait_youtube_porn.txt").read().split("\n")
    clickbait = sequence.pad_sequences([words_to_indices(inverse_vocabulary, clean(sentence).lower().split()) for sentence in clickbait], maxlen=SEQUENCE_LENGTH)

    genuine = open("data/genuine_youtube_porn.txt").read().split("\n")
    genuine = sequence.pad_sequences([words_to_indices(inverse_vocabulary, clean(sentence).lower().split()) for sentence in genuine], maxlen=SEQUENCE_LENGTH)

    X = np.concatenate([clickbait, genuine], axis=0)
    y = np.array([[1] * clickbait.shape[0] + [0] * genuine.shape[0]], dtype=np.int32).T
    p = np.random.permutation(y.shape[0])
    X = X[p]
    y = y[p]

    X_train, X_test, y_train, y_test =  train_test_split(X, y, stratify=y)


    embedding_weights = np.load("models/youtube_embeddings.npy")
    params = dict(vocabulary_size=len(vocabulary), embedding_dimension=EMBEDDING_DIMENSION, input_length=SEQUENCE_LENGTH, embedding_weights=embedding_weights)
    model = ConvolutionalNet(**params)


    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=20, shuffle=True, callbacks=[EarlyStopping(monitor="val_loss", patience=2)])
    model.save_weights(MODEL_FILE)