import os
import json
from math import ceil
import logging
import argparse
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from vectorization import (get_data, get_label_vectorizer, get_words,
                           vectorizes_sequences, get_generator_rnn)
from rnn_models import get_rnn_model, compile_model
from utils import get_client, upload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # training parameters
    parser.add_argument("-f", "--file", help="File with passwords")
    parser.add_argument("--english-words",
                        help="File containing english words")
    parser.add_argument("-l", "--max-len", type=int, default=14,
                        help="Length of passwords")
    parser.add_argument("-b", "--batch-size", default=128, type=int)
    parser.add_argument("-e", "--epochs", default=10, type=int)
    parser.add_argument("-t", "--test", action="store_true", default=False)
    parser.add_argument("--tensorboard-logs", default="log/")

    # model parameters
    parser.add_argument("-u", "--units", type=int, default=500,
                        help="number of rnn neurons")
    parser.add_argument("-n", "--neurons", type=int, default=500,
                        help="number of neurons in dense layers")
    parser.add_argument("--embedding-size", type=int, default=300)
    parser.add_argument("--vocabulary-size", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.2)

    # optimizer parameters
    parser.add_argument("--learning-rate", type=float, default=0.001)

    # save parameters
    parser.add_argument("-s", "--save")
    parser.add_argument("--model-fname", default="rnn.h5")
    parser.add_argument("--encoder-fname", default="encoder.json")
    parser.add_argument("--bucket")

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    data = get_data(args.file)
    if args.english_words:
        words = get_words(args.english_words)
        data = words + data
    data = data[:3000000]
    if args.test:
        data = data[:1000]
    print(len(data))
    logging.info("data loaded")
    l2i, i2l = get_label_vectorizer(data,
                                    start_end=True,
                                    vocabulary_size=args.vocabulary_size)
    with open(os.path.join(args.save, args.encoder_fname), 'w') as f:
        json.dump((l2i, i2l), f)
    logging.info("vectorizer dicts created")
    vec_features, vec_labels = vectorizes_sequences(data, l2i)
    logging.info("vectorization performed")

    x_train, x_test, y_train, y_test = train_test_split(vec_features,
                                                        vec_labels,
                                                        test_size=0.1)
    logging.info("train test split done")

    train_nb_steps = ceil(len(x_train) / args.batch_size)
    train_gen = get_generator_rnn(l2i, x_train, y_train,
                                  args.max_len, args.batch_size)
    test_nb_steps = ceil(len(x_test) / args.batch_size)
    test_gen = get_generator_rnn(l2i, x_test, y_test,
                                 args.max_len, args.batch_size,
                                 training=False)

    model = get_rnn_model(len(l2i),
                          args.embedding_size,
                          args.units,
                          args.neurons,
                          args.dropout)
    optimizer = Adam(learning_rate=args.learning_rate)
    model = compile_model(model, optimizer)
    logging.info("model compiled")
    save_cb = ModelCheckpoint(os.path.join(args.save, args.model_fname),
                              monitor="val_loss",
                              save_best_only=True, verbose=0)
    tensorboard_cb = TensorBoard(log_dir=args.tensorboard_logs,
                                 embeddings_freq=1,
                                 write_graph=True,
                                 histogram_freq=1)
    his = model.fit(train_gen,
                    steps_per_epoch=train_nb_steps,
                    validation_data=test_gen,
                    validation_steps=test_nb_steps,
                    epochs=args.epochs,
                    callbacks=[save_cb, tensorboard_cb],
                    verbose=1)

    if args.bucket:
        client = get_client()
        upload(client,
               os.path.join(args.save,
                            args.model_fname),
               args.bucket,
               args.save)
        upload(client,
               os.path.join(args.save, args.encoder_fname),
               args.bucket,
               args.save)
