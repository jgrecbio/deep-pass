from typing import Optional
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv1D, Layer, Input,
                                     Dense, Flatten, TimeDistributed)


class ResBlock(Layer):
    def __init__(self,
                 filters: int, kernel_size: int, activation: str = "relu",
                 name: str = "resblock",
                 trainable: bool = True, dtype=None, dynamic=False,
                 kernel_initializer: str = "he_normal",
                 **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.conv1 = Conv1D(filters, kernel_size,
                            kernel_initializer=kernel_initializer,
                            activation=activation,
                            name="conv1", padding="same")
        self.conv2 = Conv1D(filters, kernel_size,
                            kernel_initializer=kernel_initializer,
                            padding="same", name="conv2")
        self.config = {"filters": filters,
                       "kernel_size": kernel_size,
                       "activation": activation,
                       "kernel_initializer": kernel_initializer}

    def get_config(self):
        config = super(ResBlock, self).get_config()
        config.update(self.config)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x):
        x_c1 = self.conv1(tf.nn.relu(x))
        x_c2 = self.conv2(x_c1)
        x_rescaled = 0.3 * x_c2
        return x + x_rescaled


def get_generator(seq_len: int,
                  depth: int,
                  filters: int,
                  kernel_size: int,
                  output_dim: int,
                  initializer: str = "he_normal",
                  name: Optional[str] = None) -> Model:
    x = Input(shape=[seq_len, depth])
    conv1 = Conv1D(filters, kernel_size, name="first_conv",
                   kernel_initializer=initializer, padding="same",
                   activation="relu")(x)
    res1 = ResBlock(filters, kernel_size, name="gen_resblock1",
                    kernel_initializer=initializer)(conv1)
    res2 = ResBlock(filters, kernel_size, name="gen_resblock2",
                    kernel_initializer=initializer)(res1)
    res3 = ResBlock(filters, kernel_size, name="gen_resblock3",
                    kernel_initializer=initializer)(res2)
    res4 = ResBlock(filters, kernel_size, name="gen_resblock4",
                    kernel_initializer=initializer)(res3)
    res5 = ResBlock(filters, kernel_size, name="gen_resblock5",
                    kernel_initializer=initializer)(res4)
    final_conv = Conv1D(filters, kernel_size, padding="same",
                        activation="softmax",
                        name="gen_final_conv")(res5)
    final_dense = TimeDistributed(
        Dense(output_dim, activation="softmax",
              kernel_initializer=initializer,
              bias_initializer=initializer))(final_conv)
    return Model(inputs=x, outputs=final_dense)


def get_discriminator(seq_len: int, depth: int,
                      filters: int, kernel_size: int,
                      initializer: str = "he_normal",
                      name: Optional[str] = None):
    x = Input(shape=[seq_len, depth])
    conv1 = Conv1D(filters, kernel_size, name="first_conv",
                   kernel_initializer=initializer, padding="same",
                   activation="relu")(x)
    res1 = ResBlock(filters, kernel_size, name="dis_resblock1",
                    kernel_initializer=initializer)(conv1)
    res2 = ResBlock(filters, kernel_size, name="dis_resblock2",
                    kernel_initializer=initializer)(res1)
    res3 = ResBlock(filters, kernel_size, name="dis_resblock3",
                    kernel_initializer=initializer)(res2)
    res4 = ResBlock(filters, kernel_size, name="dis_resblock4",
                    kernel_initializer=initializer)(res3)
    res5 = ResBlock(filters, kernel_size, name="dis_resblock5",
                    kernel_initializer=initializer)(res4)
    flat = Flatten()(res5)
    final_dense = Dense(2, activation="softmax", name="dis_final",
                        kernel_initializer=initializer,
                        bias_initializer=initializer)(flat)
    return Model(inputs=x, outputs=final_dense, name=name)
