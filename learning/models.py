import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.activations import softplus, relu
from tensorflow.keras.backend import random_normal
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Lambda, Conv2DTranspose, Reshape, Concatenate

# model definition class
class VAEModel(Model):
    def __init__(self, n_z, stddev_epsilon=1e-6):
        super(VAEModel, self).__init__()
    
        self.n_z = n_z
        self.stddev_epsilon = stddev_epsilon

        self.merge = Concatenate(axis=-1)

        # Encoder architecture
        self.fce1 = Dense(units=4096, activation='relu')
        self.fce2 = Dense(units=1024, activation='relu')
        self.fce3 = Dense(units=2*self.n_z)

        # Latent space
        self.mean_params = Lambda(lambda x: x[:, :self.n_z])
        self.stddev_params = Lambda(lambda x: x[:, self.n_z:])

        # Decoder architecture
        self.fcd1 = Dense(units=4096, activation='relu')
        self.fcd2 = Dense(units=1024, activation='relu')
        self.fcd3 = Dense(units=5)

    def call(self, x, y, inter=None):

        # Encoding
        x = self.merge([x,y])
        x = self.fce1(x)
        x = self.fce2(x)
        x = self.fce3(x)
        means = self.mean_params(x)
        stddev = tf.math.exp(0.5*self.stddev_params(x))
        eps = random_normal(tf.shape(stddev))

        # Decoding
        z = means + eps * stddev
        #if inter is not None:
        #    z = tf.keras.layers.add([z,inter])
        x = self.merge([z,y])
        x = self.fcd1(z)
        x = self.fcd2(z)
        x = self.fcd3(x)

        return x, means, stddev, z