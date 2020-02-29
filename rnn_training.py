import argparse
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from vectorization import (get_data, get_label_vectorizer,
                           get_ohe, ohe_vectorizes, vectorizes_sequences)
from rnn_models import get_rnn_model, compile_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # training parameters
    parser.add_argument("-f", "--file")
    parser.add_argument("-l", "--max-len", type=int, default=14)
    parser.add_argument("-s", "--save")
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument("-e", "--epochs", default=100, type=int)
    parser.add_argument("-t", "--test", action="store_true", default=False)

    # model parameters
    parser.add_argument("-u", "--units", type=int, default=150)
    parser.add_argument("-n", "--neurons", type=int, default=150)
    parser.add_argument("--embedding-size", type=int, default=300)
    parser.add_argument("--vocabulary-size", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.2)

    # optimizer parameters
    parser.add_argument("--learning-rate", type=float, default=0.001)

    args = parser.parse_args()

    data = get_data(args.file)
    if args.test:
        data = data[:1000]
    l2i, i2l = get_label_vectorizer(data,
                                    start_end=True,
                                    vocabulary_size=args.vocabulary_size)
    vec_features, vec_labels = vectorizes_sequences(data, l2i, args.max_len)
    ohe = get_ohe(vec_labels)
    ohe_vec_labels = ohe_vectorizes(ohe, vec_labels)

    x_train, x_test, y_train, y_test = train_test_split(vec_features,
                                                        ohe_vec_labels,
                                                        test_size=0.1)

    model = get_rnn_model(len(l2i) - 1,
                          args.embedding_size,
                          args.units,
                          args.neurons,
                          args.dropout)
    optimizer = Adam(learning_rate=args.learning_rate)
    model = compile_model(model, optimizer)
    save_callback = ModelCheckpoint(args.save, save_best_only=True, verbose=1)
    his = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    callbacks=[save_callback])
