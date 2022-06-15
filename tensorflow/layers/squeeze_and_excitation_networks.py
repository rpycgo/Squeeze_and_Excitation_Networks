import tensorflow as tf
from tensorflow.keras.layers import Layer, GlobalAveragePooling1D, Dense


class SqueezeAndExcitationNetworks(Layer):
    def __init__(self, reduction_ratio, **kwargs):
        super(SqueezeAndExcitationNetworks, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def call(self, x):
        average_pooling = GlobalAveragePooling1D()(x)
        fc1 = Dense(units=(x.shape[-1] // self.reduction_ratio), activation='relu')(average_pooling)
        fc2 = Dense(units=(x.shape[-1]), activation='sigmoid')(fc1)

        fc2_repeated = tf.repeat(fc2, repeats=x.shape[1], axis=0)
        fc2_for_multiply = tf.reshape(fc2_repeated, shape=(-1, *x.shape[1:]))

        return x * fc2_for_multiply
