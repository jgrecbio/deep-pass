from typing import Tuple
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv1D, Layer, concatenate,
                                     Dense, Flatten, TimeDistributed)


class ResBlock(Layer):
    def __init__(self,
                 filters: int, kernel_size: int, activation: str = "relu",
                 end_block: bool = False,
                 name: str = "resblock",
                 trainable: bool = True, dtype=None, dynamic=False,
                 **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.end_block = end_block
        self.conv1 = Conv1D(filters, kernel_size,
                            activation=activation,
                            name="conv1", padding="same")
        self.conv2 = Conv1D(filters, kernel_size,
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
                         fake_inputs, real_inputs,
                         batch_size: int = 32, lamb: int = 10):
    alpha = tf.random_uniform(shape=[batch_size, 1, 1],
                              minval=0., maxval=1.)

    differences = fake_inputs - real_inputs
    interpolates = real_inputs + (alpha*differences)
    gradients = tf.gradients(discriminator(interpolates))
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),
                                   reduction_indices=[1, 2]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    return lamb * gradient_penalty


@tf.function
def train_step(generator: Generator,
               discriminator: Discriminator,
               x: tf.Tensor,
               noise: tf.Tensor,
               batch_size: int = 32):
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


def train(generator: Generator,
          discriminator: Discriminator,
          xs: tf.data.Dataset,
          batch_size: int,
          epochs: int = 1000) -> Tuple[Generator, Discriminator]:
    for e in epochs:
        for x in xs:
            noise = tf.random.uniform(tf.shape(x), minval=0., maxval=1,
                                      name="noise")
            train_step(generator, discriminator, x, noise, batch_size)
    return generator, discriminator


if __name__ == "__main__":
    gen = Generator(128, 20, 26)
    dis = Discriminator(128, 20)

    x = tf.ones([32, 10, 26])
    y = gen(x)
    print(x.shape)
    y_ = dis(y)
    print(y_.shape)
