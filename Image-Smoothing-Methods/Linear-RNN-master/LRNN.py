import keras.keras.backend as K
from keras.keras.engine.topology import Layer
from keras.keras.models import Model
from keras.keras.layers import Input, Lambda
from keras.keras.layers.convolutional import Conv2D, UpSampling2D
from keras.keras.layers.pooling import MaxPooling2D
from keras.keras.layers.merge import maximum, add, concatenate

class LRNN(Layer):

    def __init__(self, horizontal, reverse, **kwargs):
        self.horizontal = horizontal
        self.reverse = reverse
        super(LRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        super(LRNN, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[0])

    def reorder_input(self, X):
        # X.shape = (batch_size, row, column, channel)
        if self.horizontal:
            X = K.permute_dimensions(X, (2, 0, 1, 3))
        else:
            X = K.permute_dimensions(X, (1, 0, 2, 3))
        if self.reverse:
            X = K.reverse(X, 0)
        return X

    def reorder_output(self, X):
        if self.reverse:
            X = K.reverse(X, 0)
        if self.horizontal:
            X = K.permute_dimensions(X, (1, 2, 0, 3))
        else:
            X = K.permute_dimensions(X, (1, 0, 2, 3))
        return X

    def call(self, x):
        [X, G] = x
        X = self.reorder_input(X)
        G = self.reorder_input(G)
        def compute(a, x):
            H = a
            X, G = x
            L = H - X
            H = G * L + X
            return H
        initializer = K.zeros_like(X[0])
        S = K.scan(compute, (X, G), initializer)
        H = self.reorder_output(S)
        return H

def gen_model(input_shape, filter_decomposition='cascade'):
    inputs = Input(shape=input_shape)

    channels = input_shape[2]
    #multi-scale
    scale0 = Conv2D(channels, (3, 3), padding='same')(inputs)
    scale1 = MaxPooling2D((2, 2))(scale0)
    scale2 = MaxPooling2D((2, 2))(scale1)
    scale3 = MaxPooling2D((2, 2))(scale2)
    resize1 = UpSampling2D((2, 2))(scale1)
    resize2 = UpSampling2D((4, 4))(scale2)
    resize3 = UpSampling2D((8, 8))(scale3)
    concat0 = concatenate([scale0, resize1, resize2, resize3], axis=-1)
    #Conv0
    conv0 = Conv2D(16, (3, 3), padding='same')(concat0)
    #96

    #Conv1
    conv1 = Conv2D(16, (5, 5), padding='same', activation='relu')(inputs)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #48
    #Conv2
    conv2 = Conv2D(32, (3, 3), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #24
    #Conv3
    conv3 = Conv2D(32, (3, 3), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D((2, 2))(conv3)
    #12
    #Conv4
    conv4 = Conv2D(32, (3, 3), padding='same', activation='relu')(pool3)
    pool4 = MaxPooling2D((2, 2))(conv4)
    #6
    #Conv5
    conv5 = Conv2D(64, (3, 3), padding='same', activation='relu')(pool4)
    upsample5 = UpSampling2D((2, 2))(conv5)
    #12
    concat5 = concatenate([conv4, upsample5], axis=-1)
    #Conv6
    conv6 = Conv2D(32, (3, 3), padding='same', activation='relu')(concat5)
    upsample6 = UpSampling2D((2, 2))(conv6)
    #24
    concat6 = concatenate([conv3, upsample6], axis=-1)
    #Conv7
    conv7 = Conv2D(32, (3, 3), padding='same', activation='relu')(concat6)
    upsample7 = UpSampling2D((2, 2))(conv7)
    #48
    concat7 = concatenate([conv2, upsample7], axis=-1)
    #Conv8
    conv8 = Conv2D(16, (3, 3), padding='same', activation='relu')(concat7)
    upsample8 = UpSampling2D((2, 2))(conv8)
    #96
    concat8 = concatenate([conv1, upsample8], axis=-1)
    #Conv9
    x = Conv2D(64, (3, 3), padding='same', activation='tanh')(concat8)

    wx1 = Lambda(lambda x: x[:,:,:,0:16])(x)
    wx2 = Lambda(lambda x: x[:,:,:,16:32])(x)
    wy1 = Lambda(lambda x: x[:,:,:,32:48])(x)
    wy2 = Lambda(lambda x: x[:,:,:,48:64])(x)

    if filter_decomposition == 'cascade':
        y1 = LRNN(horizontal=True, reverse=False)([conv0, wx1])
        y2 = LRNN(horizontal=True, reverse=True)([conv0, wx1])
        y3 = LRNN(horizontal=False, reverse=False)([conv0, wy1])
        y4 = LRNN(horizontal=False, reverse=True)([conv0, wy1])
        y5 = LRNN(horizontal=True, reverse=False)([y1, wx2])
        y6 = LRNN(horizontal=True, reverse=True)([y2, wx2])
        y7 = LRNN(horizontal=False, reverse=False)([y3, wy2])
        y8 = LRNN(horizontal=False, reverse=True)([y4, wy2])
        y = maximum([y5, y6, y7, y8])
    elif filter_decomposition == 'parallel':
        y1 = LRNN(horizontal=True, reverse=False)([conv0, wx1])
        y2 = LRNN(horizontal=True, reverse=False)([conv0, wx2])
        y3 = LRNN(horizontal=True, reverse=True)([conv0, wx1])
        y4 = LRNN(horizontal=True, reverse=True)([conv0, wx2])
        y5 = LRNN(horizontal=False, reverse=False)([conv0, wy1])
        y6 = LRNN(horizontal=False, reverse=False)([conv0, wy2])
        y7 = LRNN(horizontal=False, reverse=True)([conv0, wy1])
        y8 = LRNN(horizontal=False, reverse=True)([conv0, wy2])
        y12 = add([y1, y2])
        y34 = add([y3, y4])
        y56 = add([y5, y6])
        y78 = add([y7, y8])
        y = maximum([y12, y34, y56, y78])
    else:
        raise ValueError("filter_decomposition should be 'cascade' or"
                         " 'parallel' but received '{}' instead"
                         .format(filter_decomposition))

    y = Conv2D(64, (3, 3), padding='same', activation='relu')(y)
    outputs = Conv2D(3, (3, 3), padding='same', activation='sigmoid')(y)
    model = Model(inputs=inputs, outputs=outputs)
    return model
