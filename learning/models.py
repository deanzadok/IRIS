import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.activations import softplus, relu
from tensorflow.keras.backend import random_normal
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Lambda, Conv2DTranspose, Reshape, Concatenate

# model definition class
class VAEModel(Model):
    def __init__(self, n_z, dim=5, stddev_epsilon=1e-6, freeze=False):
        super(VAEModel, self).__init__()
    
        self.freeze = freeze
        self.dim = dim
        self.n_z = n_z
        self.stddev_epsilon = stddev_epsilon

        if not self.freeze:
            self.merge = Concatenate(axis=-1)

            # Encoder architecture
            self.fce1 = Dense(units=2048, activation='relu')
            self.fce2 = Dense(units=1024, activation='relu')
            self.fce3 = Dense(units=2*self.n_z)

        # Decoder architecture
        self.fcd1 = Dense(name='dense_3', units=2048, activation='relu')
        self.fcd2 = Dense(name='dense_4', units=1024, activation='relu')
        self.fcd3 = Dense(name='dense_5', units=5)

    def call(self, inputs, inter=None):
        
        if not self.freeze:
            # encoding
            x = self.fce1(inputs)
            x = self.fce2(x)
            x = self.fce3(x)

            # latent space
            means, stddev = tf.split(x, [self.n_z, self.n_z], axis=-1)
            stddev = tf.math.exp(0.5*stddev)
            eps = tf.random.normal(tf.shape(stddev))
            z = means + eps * stddev

            # decoding
            _, y = tf.split(inputs, [self.dim, 10201], axis=-1)
            x = self.merge([z,y])

            x = self.fcd1(x)
            x = self.fcd2(x)
            x = self.fcd3(x)
            return x, means, stddev, z

        else:
            x = self.fcd1(inputs)
            x = self.fcd2(x)
            x = self.fcd3(x)
            return x

