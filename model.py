# -*- coding: utf-8 -*-
"""
U-Net Implementation for the paper
Increasing Consistency and Completeness of Rock Glacier Inventories with Machine Learning – the example of Austria
Georg H. Erharter, Thomas Wagner, Gerfried Winkler, Thomas Marcher
submitted to the Journal:
Computers and Geosciences

Coding was done by Georg H. Erharter
This U-Net Implementation is a modified version of
https://github.com/malhotraa/carvana-image-masking-challenge
which was created for the kaggle challange: Carvana Image Masking Challenge
https://www.kaggle.com/c/carvana-image-masking-challenge
Modifications concern the size of layers so that it fits the purpose of the
publication. Further information is available in the paper.

U-Net original reference:

Ronneberger, O., Fischer, P., Brox, T., 2015. U-Net: Convolutional Networks for
Biomedical Image Segmentation, In: Navab, N., Hornegger, J., Wells, W.M.,
Frangi, A.F. (Eds.) Medical Image Computing and Computer-Assisted Intervention
– MICCAI 2015, Springer International Publishing, Cham, pp. 234–241.
"""


from tensorflow.keras.models import Model
from tensorflow.keras.layers import UpSampling2D, Activation, MaxPooling2D
from tensorflow.keras.layers import concatenate, Input, Conv2D
from tensorflow.keras.layers import BatchNormalization, Dropout


#### disable warnings from tensorflow if desired
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
####


##############################################################################
# U-Net modified after https://github.com/malhotraa/carvana-image-masking-challenge

def down(filters, input_):
    down_ = Conv2D(filters, (3, 3), padding='same')(input_)
    down_ = BatchNormalization(epsilon=1e-4)(down_)
    down_ = Activation('relu')(down_)

    down_ = Conv2D(filters, (3, 3), padding='same')(down_)
    down_ = BatchNormalization(epsilon=1e-4)(down_)
    down_res = Activation('relu')(down_)

    down_pool = MaxPooling2D((2, 2), strides=(2, 2))(down_)
    return down_pool, down_res


def up(filters, input_, down_):
    up_ = UpSampling2D((2, 2))(input_)
    up_ = concatenate([down_, up_], axis=3)
    up_ = Conv2D(filters, (3, 3), padding='same')(up_)
    up_ = BatchNormalization(epsilon=1e-4)(up_)
    up_ = Activation('relu')(up_)

    up_ = Conv2D(filters, (3, 3), padding='same')(up_)
    up_ = BatchNormalization(epsilon=1e-4)(up_)
    up_ = Activation('relu')(up_)

    up_ = Conv2D(filters, (3, 3), padding='same')(up_)
    up_ = BatchNormalization(epsilon=1e-4)(up_)
    up_ = Activation('relu')(up_)

    return up_


def get_unet(input_shape, num_classes=1):

    inputs = Input(shape=input_shape)

    down0b, down0b_res = down(32, inputs)
    down0, down0_res = down(32, down0b)
    down1, down1_res = down(64, down0)
    down2, down2_res = down(128, down1)
    down3, down3_res = down(256, down2)

    center = Conv2D(512, (3, 3), padding='same')(down3)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)

    center = Conv2D(512, (3, 3), padding='same')(center)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)

    up3 = up(256, center, down3_res)
    up2 = up(128, up3, down2_res)
    up1 = up(64, up2, down1_res)
    up0 = up(32, up1, down0_res)
    up0b = up(32, up0, down0b_res)

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid',
                      name='final_layer')(up0b)

    model = Model(inputs=inputs, outputs=classify)
    model.summary()
    return model
