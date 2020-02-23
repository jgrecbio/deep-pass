import logging
import argparse
from typing import Tuple, List, Iterable, Dict
from toolz import concat
import tensorflow as tf
import numpy as np
from math import ceil
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv1D, Layer, concatenate,
                                     Dense, Flatten, TimeDistributed)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from vectorization import (get_data, get_label_vectorizer,
                           get_ohe, vectorizes_psswds,
                           ohe_vectorizes, inv_transform)
from utils import get_client, upload


class ResBlock(Layer):
    def __init__(self,
                 filters: int, kernel_size: int, activation: str = "relu",
                 end_block: bool = False,
                 name: str = "resblock",
                 trainable: bool = True, dtype=None, dynamic=False,
                 kernel_initializer: str = "he_normal",
                 **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.end_block = end_block
        self.conv1 = Conv1D(filters, kernel_size,
                            kernel_initializer=kernel_initializer,
                            activation=activation,
                            name="conv1", padding="same")
        self.conv2 = Conv1D(filters, kernel_size,
                            kernel_initializer=kernel_initializer,
                            padding="same", name="conv2")

    def call(self, x):
        x_c1 = self.conv1(tf.nn.relu(x))
        x_c2 = self.conv2(x_c1)
        x_rescaled = 0.3 * x_c2
        if self.end_block:
            return x
        else:
            return concatenate([x, x_rescaled], -1)


class Generator(Model):
    def __init__(self, filters: int, kernel_size: int, output_dim: int,
                 initializer: str = "he_normal"):
        super().__init__()
        self.res1 = ResBlock(filters, kernel_size, name="gen_resblock1",
                             kernel_initializer=initializer)
        self.res2 = ResBlock(filters, kernel_size, name="gen_resblock2",
                             kernel_initializer=initializer)
        self.res3 = ResBlock(filters, kernel_size, name="gen_resblock3",
                             kernel_initializer=initializer)
        self.res4 = ResBlock(filters, kernel_size, name="gen_resblock4",
                             kernel_initializer=initializer)
        self.res5 = ResBlock(filters, kernel_size, name="gen_resblock5",
                             kernel_initializer=initializer, end_block=True)
        self.final_conv = Conv1D(filters, kernel_size, padding="same",
                                 activation="softmax",
                                 name="gen_final_conv")
        self.final_dense = TimeDistributed(
            Dense(output_dim, activation="softmax",
                  kernel_initializer=initializer,
                  bias_initializer=initializer)
        )

    def call(self, x):
        x1 = self.res1(x)
        x2 = self.res2(x1)
        x3 = self.res3(x2)
        x4 = self.res4(x3)
        x5 = self.res5(x4)
        return self.final_dense(self.final_conv(x5))


class Discriminator(Model):
    def __init__(self, filters: int, kernel_size: int,
                 initializer: str = "he_normal"):
        super().__init__()
        self.res1 = ResBlock(filters, kernel_size, name="dis_resblock1",
                             kernel_initializer=initializer)
        self.res2 = ResBlock(filters, kernel_size, name="dis_resblock2",
                             kernel_initializer=initializer)
        self.res3 = ResBlock(filters, kernel_size, name="dis_resblock3",
                             kernel_initializer=initializer)
        self.res4 = ResBlock(filters, kernel_size, name="dis_resblock4",
                             kernel_initializer=initializer)
        self.res5 = ResBlock(filters, kernel_size, name="dis_resblock5",
                             kernel_initializer=initializer, end_block=True)
        self.flat = Flatten()
        self.final_dense = Dense(2, activation="softmax", name="dis_final",
                                 kernel_initializer=initializer,
                                 bias_initializer=initializer)

    def call(self, x):
        x1 = self.res1(x)
        x2 = self.res2(x1)
        x3 = self.res3(x2)
        x4 = self.res4(x3)
        x5 = self.res5(x4)
        x_flat = self.flat(x5)
        return self.final_dense(x_flat)


def generator_loss(fake_x_pred: tf.Tensor) -> tf.Tensor:
    return -tf.reduce_mean(fake_x_pred)


def standard_wassertein(fake_pred, true_pred):
    return tf.reduce_mean(fake_pred) - tf.reduce_mean(true_pred)


def improved_wasserstein(discriminator: Discriminator,
                         fake_inputs: tf.Tensor,
                         real_inputs: tf.Tensor,
                         batch_size: int = 128,
                         lamb: int = 10) -> tf.Tensor:
    alpha = tf.random.uniform(shape=[batch_size, 1, 1],
                              minval=0., maxval=1.)

    differences = fake_inputs - real_inputs
    interpolates = real_inputs + tf.multiply(alpha, differences)
    gradients = tf.gradients(discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
                                   axis=[1, 2]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    return lamb * gradient_penalty


@tf.function
def train_step(generator: Generator,
               discriminator: Discriminator,
               x: tf.Tensor,
               noise: tf.Tensor):
    batch_size = tf.shape(x)[0]
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        fake_x = generator(noise)
        fake_x_pred = discriminator(fake_x)
        x_pred = discriminator(x)

        gen_loss = generator_loss(fake_x)
        standard_dis_loss = standard_wassertein(fake_x_pred, x_pred)
        im_dis_loss = improved_wasserstein(discriminator,
                                           fake_x,
                                           x,
                                           batch_size)
        imw_loss = standard_dis_loss + im_dis_loss

        gen_tape.gradient(gen_loss, generator.trainable_variables)
        dis_tape.gradient(imw_loss, discriminator.trainable_variables)


def construct_batches(x: np.ndarray,
                      batch_size: int = 128) -> Iterable[np.ndarray]:
    for i in range(0, len(x), batch_size):
        yield x[i: i + batch_size]


def train(generator: Generator,
          discriminator: Discriminator,
          xs: List[np.ndarray],
          ohe: OneHotEncoder,
          i2l: Dict[int, str],
          epochs: int = 10000,
          log_epoch: int = 1000) -> Tuple[Generator, Discriminator]:
    for e in range(epochs):
        for x in xs:
            x = tf.convert_to_tensor(ohe_vectorizes(ohe, x), dtype=tf.float32)
            r, c, _ = tf.shape(x)
            noise = tf.random.uniform([r, c, 300], minval=0., maxval=1,
                                      name="noise")
            train_step(generator, discriminator, x, noise)
            if e % log_epoch == 0:
                examples = generate_psswds(generator, 10, 10, 300, c, i2l)
                logging.info(examples)

    return generator, discriminator


def generate_psswds(generator: Generator,
                    num_psswds: int,
                    batch_size: int,
                    depth: int,
                    maxlen: int,
                    i2l: Dict[int, str]) -> List[str]:
    nb_batch = ceil(num_psswds / batch_size)
    psswds = []
    for b in range(nb_batch):
        noise = tf.random.uniform([batch_size, maxlen, depth],
                                  minval=0., maxval=1.)
        fake_inputs = np.argmax(generator(noise), -1)
        psswds.append(inv_transform(fake_inputs, i2l))
    return list(concat(psswds))[:num_psswds]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file")
    parser.add_argument("-b", "--batch-size", default=128, type=int)
    parser.add_argument("-e", "--epochs", default=300000, type=int)
    parser.add_argument("-t", "--test", action="store_true", default=False)
    parser.add_argument("-l", "--log-epoch", type=int, default=10000)

    parser.add_argument("-g", "--generator", default="generator.h5")
    parser.add_argument("-d", "--discriminator", default="discriminator.h5")
    parser.add_argument("--bucket")

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    data = get_data(args.file)
    logging.info("dataset loaded")
    if args.test:
        data = data[:1000]
    l2i, i2l = get_label_vectorizer(data)
    logging.info("vectorizer dicts generated")
    vec_data = vectorizes_psswds(data, l2i)

    ohe = get_ohe(vec_data)
    logging.info("one-hot-encoder generated")

    train_data, test_data = train_test_split(vec_data)
    train_batch = construct_batches(train_data, args.batch_size)
    logging.info("train-test splits generated")

    gen = Generator(128, 20, len(ohe.categories_[0]))
    dis = Discriminator(128, 20)
    logging.info("generator and discriminator generated")

    trained_gen, trained_dis = train(gen, dis,
                                     train_batch,
                                     ohe,
                                     i2l,
                                     args.epochs,
                                     log_epoch=1)
    generated_psswds = generate_psswds(trained_gen, 1000, 128,
                                       300, 14, i2l)
    logging.info(generated_psswds[:10])

    trained_gen.save(args.generator, save_format="h5")
    trained_dis.save(args.discriminator, save_format="h5")

    if args.bucket:
        client = get_client()
        upload(client, args.generator, args.bucket, args.generator)
        upload(client, args.discriminator, args.bucket, args.discriminator)
