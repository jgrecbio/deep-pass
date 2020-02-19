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
    def __init__(self, filters: int, kernel_size: int, output_dim: int):
        super().__init__()
        self.res1 = ResBlock(filters, kernel_size, name="gen_resblock1")
        self.res2 = ResBlock(filters, kernel_size, name="gen_resblock2")
        self.res3 = ResBlock(filters, kernel_size, name="gen_resblock3")
        self.res4 = ResBlock(filters, kernel_size, name="gen_resblock4")
        self.res5 = ResBlock(filters, kernel_size, name="gen_resblock5",
                             end_block=True)
        self.final_conv = Conv1D(filters, kernel_size, padding="same",
                                 activation="softmax",
                                 name="gen_final_conv")
        self.final_dense = TimeDistributed(
            Dense(output_dim, activation="softmax")
        )

    def call(self, x):
        x1 = self.res1(x)
        x2 = self.res2(x1)
        x3 = self.res3(x2)
        x4 = self.res4(x3)
        x5 = self.res5(x4)
        return self.final_dense(self.final_conv(x5))


class Discriminator(Model):
    def __init__(self, filters, kernel_size):
        super().__init__()
        self.res1 = ResBlock(filters, kernel_size, name="dis_resblock1")
        self.res2 = ResBlock(filters, kernel_size, name="dis_resblock2")
        self.res3 = ResBlock(filters, kernel_size, name="dis_resblock3")
        self.res4 = ResBlock(filters, kernel_size, name="dis_resblock4")
        self.res5 = ResBlock(filters, kernel_size, name="dis_resblock5",
                             end_block=True)
        self.flat = Flatten()
        self.final_dense = Dense(2, activation="softmax", name="dis_final")

    def call(self, x):
        x1 = self.res1(x)
        x2 = self.res2(x1)
        x3 = self.res3(x2)
        x4 = self.res4(x3)
        x5 = self.res5(x4)
        x_flat = self.flat(x5)
        return self.final_dense(x_flat)


if __name__ == "__main__":
    gen = Generator(128, 20, 26)
    dis = Discriminator(128, 20)

    x = tf.ones([32, 10, 26])
    y = gen(x)
    print(x.shape)
    y_ = dis(y)
    print(y_.shape)
