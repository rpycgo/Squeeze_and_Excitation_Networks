import tensorflow as tf
from tensorflow.keras.layers import Layer, GlobalAveragePooling1D, Dense, Conv1D


class SqueezeAndExcitationNetworks(Layer):
    def __init__(self, reduction_ratio, **kwargs):
        super(SqueezeAndExcitationNetworks, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def call(self, x):
        filters=x.shape[-1]

        u = Conv1D(filters=filters, kernel_size=x.shape[1])(x)
        average_pooling = GlobalAveragePooling1D()(u)
        fc1 = Dense(units=(filters // self.reduction_ratio), activation='relu')(average_pooling)
        fc2 = Dense(units=(filters), activation='sigmoid')(fc1)

        fc2_repeated = tf.repeat(fc2, repeats=x.shape[1], axis=0)
        fc2_for_multiply = tf.reshape(fc2_repeated, shape=(-1, *x.shape[1:]))

        return u * fc2_for_multiply
