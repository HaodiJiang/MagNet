from keras.layers import Activation, Conv2D

import tensorflow as tf
from keras.layers import Layer
from keras.backend import batch_dot


class PAM(Layer):
    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(PAM, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1, ),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input):
        input_shape = input.get_shape().as_list()
        _, h, w, filters = input_shape

        b = Conv2D(filters // 8, 1, use_bias=False, kernel_initializer='he_normal')(input)
        c = Conv2D(filters // 8, 1, use_bias=False, kernel_initializer='he_normal')(input)
        d = Conv2D(filters, 1, use_bias=False, kernel_initializer='he_normal')(input)

        vec_b = tf.reshape(b, (-1, h * w, filters // 8))
        vec_cT = tf.transpose(tf.reshape(c, (-1, h * w, filters // 8)), (0, 2, 1))
        bcT = batch_dot(vec_b, vec_cT)
        softmax_bcT = Activation('softmax')(bcT)
        vec_d = tf.reshape(d, (-1, h * w, filters))
        bcTd = batch_dot(softmax_bcT, vec_d)
        bcTd = tf.reshape(bcTd, (-1, h, w, filters))

        out = self.gamma*bcTd + input
        return out


class CAM(Layer):
    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(CAM, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1, ),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input):
        input_shape = input.get_shape().as_list()
        _, h, w, filters = input_shape

        vec_a = tf.reshape(input, (-1, h * w, filters))
        vec_aT = tf.transpose(vec_a, (0, 2, 1))
        aTa = batch_dot(vec_aT, vec_a)
        softmax_aTa = Activation('softmax')(aTa)
        aaTa = batch_dot(vec_a, softmax_aTa)
        aaTa = tf.reshape(aaTa, (-1, h, w, filters))

        out = self.gamma*aaTa + input
        return out
