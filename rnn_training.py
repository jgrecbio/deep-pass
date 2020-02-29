import numpy as np
from toolz import concat
import logging
import argparse
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from vectorization import (get_data, get_label_vectorizer,
                           get_ohe_rnn, ohe_vectorizes_rnn,
                           vectorizes_sequences)
from rnn_models import get_rnn_model, compile_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # training parameters
    parser.add_argument("-f", "--file")
    parser.add_argument("-l", "--max-len", type=int, default=14)
    parser.add_argument("--tensorboard-logs")
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

    logging.getLogger().setLevel(logging.INFO)

    data = get_data(args.file)
    if args.test:
        data = data[:1000000]
    logging.info("data loaded")
    l2i, i2l = get_label_vectorizer(data,
                                    start_end=True,
                                    vocabulary_size=args.vocabulary_size)
    logging.info("vectorizer dicts created")
    print(len(l2i))
    vec_features, vec_labels, = vectorizes_sequences(data, l2i)
    padded_vec_features = pad_sequences(vec_features,
                                        maxlen=args.max_len,
                                        truncating="post",
                                        padding="post",
                                        value=0.)
    padded_vec_labels = pad_sequences(ohe_vectorizes_rnn(l2i, vec_labels),
                                      maxlen=args.max_len,
                                      truncating="post",
                                      padding="post",
                                      value=0.)

    print(padded_vec_labels.shape)
    print(padded_vec_features.shape)
    print(np.max(padded_vec_features))
    logging.info("ohe vectorization performed")

    x_train, x_test, y_train, y_test = train_test_split(padded_vec_features,
                                                        padded_vec_labels,
                                                        test_size=0.1)
    logging.info("train test split done")

    model = get_rnn_model(len(l2i),
                          args.embedding_size,
                          args.units,
                          args.neurons,
                          args.dropout)
    optimizer = Adam(learning_rate=args.learning_rate)
    model = compile_model(model, optimizer)
    logging.info("model compiled")
    save_cb = ModelCheckpoint(args.save, save_best_only=True, verbose=1)
    tensorboard_cb = TensorBoard(log_dir=args.tensorboard_logs,
                                 embeddings_freq=1,
                                 write_graph=True,
                                 histogram_freq=1)
    his = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    callbacks=[save_cb, tensorboard_cb])
