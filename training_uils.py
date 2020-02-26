import logging
from typing import Dict, Tuple, Optional, List
from toolz import concat
from math import ceil
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import OneHotEncoder

from vectorization import construct_batches, ohe_vectorizes, inv_transform


def generator_loss(fake_x_pred: tf.Tensor) -> tf.Tensor:
    return -tf.reduce_mean(fake_x_pred)


def standard_wassertein(fake_pred: tf.Tensor,
                        true_pred: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(fake_pred) - tf.reduce_mean(true_pred)


def improved_wasserstein(discriminator: Model,
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
    cost = lamb * gradient_penalty
    return cost


@tf.function
def train_step(generator: Model,
               discriminator: Model,
               generator_optimizer: RMSprop,
               discriminator_optimizer: RMSprop,
               x: tf.Tensor,
               noise: tf.Tensor,
               gen_loss_storage: tf.keras.metrics.Mean,
               dis_sw_loss_storage: tf.keras.metrics.Mean):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        fake_x = generator(noise)
        fake_x_pred = discriminator(fake_x)
        x_pred = discriminator(x)

        gen_loss = generator_loss(fake_x_pred)
        gen_loss_storage(gen_loss)
        standard_dis_loss = standard_wassertein(fake_x_pred, x_pred)
        dis_sw_loss_storage(standard_dis_loss)

        generator_gradients = gen_tape.gradient(
            gen_loss, generator.trainable_variables
        )
        discriminator_gradients = dis_tape.gradient(
            standard_dis_loss, discriminator.trainable_variables
        )
        discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, discriminator.trainable_variables)
        )
        generator_optimizer.apply_gradients(
            zip(generator_gradients, generator.trainable_variables)
        )


def train(generator: Model,
          discriminator: Model,
          noise_size: int,
          features: np.ndarray,
          ohe: OneHotEncoder,
          i2l: Dict[int, str],
          optimizers: Optional[Tuple[RMSprop, RMSprop]] = None,
          train_writer: Optional[tf.summary.SummaryWriter] = None,
          epochs: int = 10000,
          log_epoch: int = 1000,
          batch_size: int = 128) -> Tuple[Model, Model]:
    if optimizers:
        generator_optimizer, discriminator_optimizer = optimizers
    else:
        generator_optimizer = RMSprop()
        discriminator_optimizer = RMSprop()

    gen_loss_storage = tf.keras.metrics.Mean(name="generator_loss")
    dis_sw_loss_storage = tf.keras.metrics.Mean(name="sw")

    for e in range(epochs):
        gen_loss_storage.reset_states()
        dis_sw_loss_storage.reset_states()

        xs = construct_batches(features, batch_size)
        logging.info("epoch: {}".format(e))
        for i, x in enumerate(xs):
            logging.debug("batch: {}".format(i))
            x = tf.convert_to_tensor(ohe_vectorizes(ohe, x), dtype=tf.float32)
            r, c, _ = tf.shape(x)
            noise = tf.random.uniform([r, c, noise_size],
                                      minval=0., maxval=1,
                                      name="noise")
            train_step(generator, discriminator,
                       generator_optimizer, discriminator_optimizer,
                       x, noise,
                       gen_loss_storage,
                       dis_sw_loss_storage)

        if train_writer:
            with train_writer.as_default():
                tf.summary.scalar("generator_loss",
                                  gen_loss_storage.result(),
                                  step=e)
                tf.summary.scalar("discriminator_loss",
                                  dis_sw_loss_storage.result(),
                                  step=e)

        if e % log_epoch == 0:
            examples = generate_psswds(generator, 10, 10, 300, c, i2l)
            logging.info(examples)
            logging.info(gen_loss_storage.result())
            logging.info(dis_sw_loss_storage.result())

    return generator, discriminator


def generate_psswds(generator: Model,
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
