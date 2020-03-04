from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Dense, TimeDistributed, Embedding,
                                     Bidirectional, Masking)


def get_rnn_model(vocab_size: int,
                  embedding_dim: int,
                  units: int,
                  neurons: int,
                  dropout: float) -> Sequential:
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=embedding_dim,
                        mask_zero=True))
    model.add(Masking())
    model.add(Bidirectional(LSTM(units=units,
                                 return_sequences=True,
                                 dropout=dropout,
                                 recurrent_dropout=dropout)))
    model.add(Bidirectional(LSTM(units=units,
                                 return_sequences=True,
                                 dropout=dropout,
                                 recurrent_dropout=dropout)))
    model.add(LSTM(units=units,
                   return_sequences=True,
                   dropout=dropout,
                   recurrent_dropout=dropout))
    model.add(TimeDistributed(Dense(neurons, activation="relu")))
    model.add(TimeDistributed(Dense(vocab_size, activation="softmax")))
    return model


def compile_model(model: Sequential,
                  optimizer: Optimizer) -> Sequential:
    model.compile(optimizer=optimizer,
                  metrics=["acc"],
                  loss="categorical_crossentropy")
    return model
