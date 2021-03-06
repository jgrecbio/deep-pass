import logging
import argparse
import tensorflow as tf
from sklearn.model_selection import train_test_split

from utils import get_client, upload
from vectorization import (get_data, get_label_vectorizer,
                           get_ohe, vectorizes_psswds)

from models import ResBlock, get_generator, get_discriminator
from training_uils import train, generate_psswds


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file")
parser.add_argument("-b", "--batch-size", default=128, type=int)
parser.add_argument("-e", "--epochs", default=300000, type=int)
parser.add_argument("-t", "--test", action="store_true", default=False)
parser.add_argument("-l", "--log-epoch", type=int, default=10000)

parser.add_argument("-g", "--generator", default="generator.h5")
parser.add_argument("-d", "--discriminator", default="discriminator.h5")
parser.add_argument("--bucket")
parser.add_argument("--tensorboard", action="store_true", default=False)

parser.add_argument("--discriminator-iter", type=int, default=10)
parser.add_argument("--noise-size", type=int, default=300)
parser.add_argument("--filters", type=int, default=128)
parser.add_argument("--seq-len", type=int, default=14)
parser.add_argument("--kernel-size", type=int, default=5)

args = parser.parse_args()

logging.getLogger().setLevel(logging.DEBUG)

if args.tensorboard:
    train_summay_dir = "./logs/train"
    train_summary_writer = tf.summary.create_file_writer(train_summay_dir)

data = get_data(args.file)
logging.info("dataset loaded")
if args.test:
    data = data[:100000]
l2i, i2l = get_label_vectorizer(data)
logging.info("vectorizer dicts generated")
vec_data = vectorizes_psswds(data, l2i)

ohe = get_ohe(vec_data)
logging.info("one-hot-encoder generated")

train_data, test_data = train_test_split(vec_data)
logging.info("train-test splits generated")

gen = get_generator(args.seq_len,
                    args.noise_size,
                    args.filters,
                    args.kernel_size,
                    len(ohe.categories_[0]),
                    name="generator")
dis = get_discriminator(args.seq_len,
                        len(ohe.categories_[0]),
                        args.filters,
                        args.kernel_size,
                        name="discriminator")
logging.info("generator and discriminator generated")

trained_gen, trained_dis = train(gen, dis,
                                 args.discriminator_iter,
                                 args.noise_size, train_data,
                                 ohe, i2l,
                                 train_writer=train_summary_writer,
                                 epochs=args.epochs,
                                 log_epoch=args.log_epoch)
generated_psswds = generate_psswds(trained_gen, 1000, args.batch_size,
                                   args.noise_size, args.seq_len, i2l)
logging.info(generated_psswds[:10])

trained_gen.save(args.generator, save_format="h5")
trained_dis.save(args.discriminator, save_format="h5")

if args.bucket:
    client = get_client()
    upload(client, args.generator, args.bucket, args.generator)
    upload(client, args.discriminator, args.bucket, args.discriminator)

tf.keras.models.load_model(args.generator,
                           custom_objects={"ResBlock": ResBlock},
                           compile=False)
