from keras.layers import Input, Dense, Conv2D, GlobalAvgPool2D, Lambda
from keras.layers import MaxPool2D, AvgPool2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D,\
    add, concatenate, Reshape, Activation, BatchNormalization, PReLU, LeakyReLU, SeparableConv2D
from keras import initializers
from keras import backend as K

from tensorflow import device
from keras.models import Model
import numpy as np
import os

import utils
from spp.SpatialPyramidPooling import SpatialPyramidPooling


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

activation = 'relu'
# conv_init = initializers.he_uniform(seed=1)
conv_init = initializers.glorot_uniform(seed=1)
# dense_init = initializers.orthogonal(seed=1)
dense_init = initializers.glorot_uniform(seed=1)
# initializer = initializers.he_uniform(seed=1)
# constant = Constant(value=0.0)

img_width = utils.img_width  # img cols
img_height = utils.img_height  # img rows
channels = utils.img_channels
input_shape = (img_height, img_width, channels)


def create_model(weights_path=None, tune=False):
    # def create_posenet(weights_path=None, tune=False, num_classes=1):
    with device('/gpu:0'):

        input = Input(shape=input_shape)
        # stem
        conv1 = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='same', kernel_initializer=conv_init, activation=activation, name='conv1')(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(conv1)
        # maxpool1 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='maxpool1')(conv1)
        # avgpool1 = AveragePooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='avgpool1')(conv1)
        # pool1 = add([maxpool1, avgpool1])

        norm1 = BatchNormalization(axis=-1, name='norm1')(pool1)
        # norm1 = GroupNormalization(groups=16, axis=-1)(pool1)

        reduction2 = Conv2D(filters=64, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='reduction2')(norm1)
        conv2 = Conv2D(filters=192, kernel_size=(3,3), padding='same', kernel_initializer=conv_init, activation=activation, name='conv2')(reduction2)

        norm2 = BatchNormalization(axis=-1, name='norm2')(conv2)
        # norm2 = GroupNormalization(groups=16, axis=-1)(conv2)

        pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='pool2')(norm2)
        # maxpool2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='maxpool2')(norm2)
        # avgpool2 = AveragePooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='avgpool2')(norm2)
        # pool2 = add([maxpool2, avgpool2])

        icp1_reduction1 = Conv2D(filters=96, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp1_reduction1')(pool2)
        icp1_out1 = Conv2D(filters=128, kernel_size=(3,3), padding='same', kernel_initializer=conv_init, activation=activation, name='icp1_out1')(icp1_reduction1)
        icp1_reduction2 = Conv2D(filters=16, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp1_reduction2')(pool2)
        icp1_out2 = Conv2D(filters=32, kernel_size=(5,5), padding='same', kernel_initializer=conv_init, activation=activation, name='icp1_out2')(icp1_reduction2)
        icp1_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='icp1_pool')(pool2)
        # icp1_maxpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp1_maxpool')(pool2)
        # icp1_avgpool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp1_avgpool')(pool2)
        # icp1_pool = add([icp1_maxpool, icp1_avgpool])
        icp1_out3 = Conv2D(filters=32, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp1_out3')(icp1_pool)
        icp1_out0 = Conv2D(filters=64, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp1_out0')(pool2)

        icp2_in = concatenate([icp1_out0, icp1_out1, icp1_out2, icp1_out3],axis=3,name='icp2_in')
        icp2_reduction1 = Conv2D(filters=128, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp2_reduction1')(icp2_in)
        icp2_out1 = Conv2D(filters=192, kernel_size=(3,3), padding='same', kernel_initializer=conv_init, activation=activation, name='icp2_out1')(icp2_reduction1)
        icp2_reduction2 = Conv2D(filters=32, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp2_reduction2')(icp2_in)
        icp2_out2 = Conv2D(filters=96, kernel_size=(5,5), padding='same', kernel_initializer=conv_init, activation=activation, name='icp2_out2')(icp2_reduction2)
        icp2_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='icp2_pool')(icp2_in)
        # icp2_maxpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp2_maxpool')(icp2_in)
        # icp2_avgpool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp2_avgpool')(icp2_in)
        # icp2_pool = add([icp2_maxpool, icp2_avgpool])
        icp2_out3 = Conv2D(filters=64, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp2_out3')(icp2_pool)
        icp2_out0 = Conv2D(filters=128, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp2_out0')(icp2_in)
        icp2_out = concatenate([icp2_out0, icp2_out1, icp2_out2, icp2_out3], axis=3, name='icp2_out')

        icp3_in = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='icp3_in')(icp2_out)
        # icp3_maxin = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp3_maxin')(icp2_out)
        # icp3_avgin = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='icp3_avgin')(icp2_out)
        # icp3_in = add([icp3_maxin, icp3_avgin])
        icp3_reduction1 = Conv2D(filters=96, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp3_reduction1')(icp3_in)
        icp3_out1 = Conv2D(filters=208, kernel_size=(3,3), padding='same', kernel_initializer=conv_init, activation=activation, name='icp3_out1')(icp3_reduction1)
        icp3_reduction2 = Conv2D(filters=16, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp3_reduction2')(icp3_in)
        icp3_out2 = Conv2D(filters=48, kernel_size=(5,5), padding='same', kernel_initializer=conv_init, activation=activation, name='icp3_out2')(icp3_reduction2)
        icp3_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='icp3_pool')(icp3_in)
        # icp3_maxpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp3_maxpool')(icp3_in)
        # icp3_avgpool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp3_avgpool')(icp3_in)
        # icp3_pool = add([icp3_maxpool, icp3_avgpool])
        icp3_out3 = Conv2D(filters=64, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp3_out3')(icp3_pool)
        icp3_out0 = Conv2D(filters=192, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp3_out0')(icp3_in)
        icp3_out = concatenate([icp3_out0, icp3_out1, icp3_out2, icp3_out3], axis=3, name='icp3_out')


        '''
        cls1_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),padding='valid',name='cls1_pool')(icp3_out)
        cls1_reduction_pose = Conv2D(filters=128,kernel_size=(1,1),padding='same',activation=activation,name='cls1_reduction_pose')(cls1_pool)
        cls1_fc1_flat = Flatten(name='cls1_fc1_flat')(cls1_reduction_pose)
        cls1_fc1 = Dense(1024, kernel_initializer=initializer, activation=activation, name='cls1_fc1')(cls1_fc1_flat)
        #cls1_fc1_pose = Dense(1024, kernel_initializer='random_uniform', activation=activation, name='cls1_fc1_pose')(cls1_fc1_flat)
        cls1_drop_fc = Dropout(0.7)(cls1_fc1)
        #cls1_fc_pose = Dense(7, kernel_initializer=initializer, name='cls1_fc_pose')(cls1_drop_fc)
        cls1_fc_pose_xyz = Dense(3, kernel_initializer=initializer, name='cls1_fc_pose_xyz')(cls1_drop_fc)
        cls1_fc_pose_wpqr = Dense(4, kernel_initializer=initializer, name='cls1_fc_pose_wpqr')(cls1_drop_fc)
        '''

        # act1 = Activation(activation)(norm3)
        # cls1_spp1_flat = SpatialPyramidPooling([1, 2, 3])(act1)
        t1 = SeparableConv2D(filters=256, kernel_size=(1, 1), padding='same', activation=activation,
                             depth_multiplier=1, kernel_initializer=conv_init, strides=(1, 1), name='t1')(icp3_out)

        # t1 = Conv2D(filters=256, kernel_size=(1, 1), padding='same', activation=activation,
        #            kernel_initializer=conv_init, strides=(1, 1))(icp3_out)

        cls1_spp1_flat = SpatialPyramidPooling([1, 2])(t1)
        # cls1_fc1 = Dense(1024, activation=activation, name='cls1_fc1')(cls1_spp1_flat)
        # cls1_drop_fc = Dropout(0.0)(cls1_spp1_flat)
        # cls1_fc_pose = Dense(7, name='cls1_fc_pose')(cls1_drop_fc)

        # norm3 = BatchNormalization(name='name3')(cls1_spp1_flat)

        cls1_fc_pose_xyz = Dense(3, name='cls1_fc_pose_xyz')(cls1_spp1_flat)
        cls1_fc_pose_wpqr = Dense(4, name='cls1_fc_pose_wpqr')(cls1_spp1_flat)
        # cls1_fc_pose_wpqr_norm = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='cls1_fc_pose_wpqr_norm')(cls1_fc_pose_wpqr)

        icp4_reduction1 = Conv2D(filters=112, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp4_reduction1')(icp3_out)
        icp4_out1 = Conv2D(filters=224, kernel_size=(3,3), padding='same', kernel_initializer=conv_init, activation=activation, name='icp4_out1')(icp4_reduction1)
        icp4_reduction2 = Conv2D(filters=24, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp4_reduction2')(icp3_out)
        icp4_out2 = Conv2D(filters=64, kernel_size=(5,5), padding='same', kernel_initializer=conv_init, activation=activation, name='icp4_out2')(icp4_reduction2)
        icp4_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='icp4_pool')(icp3_out)
        # icp4_maxpool = MaxPooling2D(pool_size=(3,3), strides=(1, 1), padding='same', name='icp4_maxpool')(icp3_out)
        # icp4_avgpool = AveragePooling2D(pool_size=(3,3), strides=(1, 1), padding='same', name='icp4_avgpool')(icp3_out)
        # icp4_pool = add([icp4_maxpool, icp4_avgpool])
        icp4_out3 = Conv2D(filters=64, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp4_out3')(icp4_pool)
        icp4_out0 = Conv2D(filters=160, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp4_out0')(icp3_out)
        icp4_out = concatenate([icp4_out0, icp4_out1, icp4_out2, icp4_out3], axis=3, name='icp4_out')

        icp5_reduction1 = Conv2D(filters=128, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp5_reduction1')(icp4_out)
        icp5_out1 = Conv2D(filters=256, kernel_size=(3,3), padding='same', kernel_initializer=conv_init, activation=activation, name='icp5_out1')(icp5_reduction1)
        icp5_reduction2 = Conv2D(filters=24, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp5_reduction2')(icp4_out)
        icp5_out2 = Conv2D(filters=64, kernel_size=(5,5), padding='same', kernel_initializer=conv_init, activation=activation, name='icp5_out2')(icp5_reduction2)
        icp5_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='icp5_pool')(icp4_out)
        # icp5_maxpool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='icp5_maxpool')(icp4_out)
        # icp5_avgpool = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='icp5_avgpool')(icp4_out)
        # icp5_pool = add([icp5_maxpool, icp5_avgpool])
        icp5_out3 = Conv2D(filters=64, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp5_out3')(icp5_pool)
        icp5_out0 = Conv2D(filters=128, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp5_out0')(icp4_out)
        icp5_out = concatenate([icp5_out0, icp5_out1, icp5_out2, icp5_out3], axis=3, name='icp5_out')

        icp6_reduction1 = Conv2D(filters=144, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp6_reduction1')(icp5_out)
        icp6_out1 = Conv2D(filters=288, kernel_size=(3,3), padding='same', kernel_initializer=conv_init, activation=activation, name='icp6_out1')(icp6_reduction1)
        icp6_reduction2 = Conv2D(filters=32, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp6_reduction2')(icp5_out)
        icp6_out2 = Conv2D(filters=64, kernel_size=(5,5), padding='same', kernel_initializer=conv_init, activation=activation, name='icp6_out2')(icp6_reduction2)
        icp6_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1), padding='same', name='icp6_pool')(icp5_out)
        # icp6_maxpool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='icp6_maxpool')(icp5_out)
        # icp6_avgpool = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='icp6_avgpool')(icp5_out)
        # icp6_pool = add([icp6_maxpool, icp6_avgpool])
        icp6_out3 = Conv2D(filters=64, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp6_out3')(icp6_pool)
        icp6_out0 = Conv2D(filters=112, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp6_out0')(icp5_out)
        icp6_out = concatenate([icp6_out0, icp6_out1, icp6_out2, icp6_out3], axis=3, name='icp6_out')

        '''
        cls2_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),padding='valid',name='cls2_pool')(icp6_out)
        cls2_reduction_pose = Conv2D(filters=128,kernel_size=(1,1),padding='same',activation=activation,name='cls2_reduction_pose')(cls2_pool)
        cls2_fc1_flat = Flatten(name='cls2_fc1_flat')(cls2_reduction_pose)
        cls2_fc1 = Dense(1024, kernel_initializer=initializer, activation=activation, name='cls2_fc1')(cls2_fc1_flat)
        #cls2_fc1_pose = Dense(1024, kernel_initializer=initializer, activation=activation, name='cls2_fc1_pose')(cls2_fc1_flat)
        cls2_drop_fc = Dropout(0.7)(cls2_fc1)
        #cls2_fc_pose = Dense(7, kernel_initializer=initializer, name='cls2_fc_pose')(cls2_drop_fc)
        cls2_fc_pose_xyz = Dense(3, kernel_initializer=initializer, name='cls2_fc_pose_xyz')(cls2_drop_fc)
        cls2_fc_pose_wpqr = Dense(4, kernel_initializer=initializer, name='cls2_fc_pose_wpqr')(cls2_drop_fc)
        '''

        # norm4 = BatchNormalization(name='norm4')(cls2_reduction_pose)
        # act2 = Activation(activation)(norm4)
        # cls2_spp1_flat = SpatialPyramidPooling([1, 2, 3])(act2)
        t2 = SeparableConv2D(filters=256, kernel_size=(1, 1), padding='same', activation=activation,
                             depth_multiplier=1, kernel_initializer=conv_init, strides=(1, 1), name='t2')(icp6_out)
        # t2 = Conv2D(filters=256, kernel_size=(1, 1), padding='same', activation=activation,
        #            kernel_initializer=conv_init, strides=(1, 1))(icp6_out)
        cls2_spp1_flat = SpatialPyramidPooling([1, 2])(t2)
        cls2_fc_pose_xyz = Dense(3, name='cls2_fc_pose_xyz')(cls2_spp1_flat)
        cls2_fc_pose_wpqr = Dense(4, name='cls2_fc_pose_wpqr')(cls2_spp1_flat)
        # cls2_fc_pose_wpqr_norm = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='cls2_fc_pose_wpqr_norm')(cls2_fc_pose_wpqr)

        # norm4 = BatchNormalization(axis=-1, name='norm4')(cls2_spp1_flat)
        # cls2_fc1 = Dense(1024, kernel_initializer=initializer, activation=activation, name='cls2_fc1')(cls2_spp1_flat)
        # cls2_drop_fc = Dropout(0.0)(cls2_spp1_flat)
        # cls2_fc_pose = Dense(7, name='cls2_fc_pose')(cls2_drop_fc)

        icp7_reduction1 = Conv2D(filters=160, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp7_reduction1')(icp6_out)
        icp7_out1 = Conv2D(filters=320, kernel_size=(3,3), padding='same', kernel_initializer=conv_init, activation=activation, name='icp7_out1')(icp7_reduction1)
        icp7_reduction2 = Conv2D(filters=32, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp7_reduction2')(icp6_out)
        icp7_out2 = Conv2D(filters=128, kernel_size=(5,5), padding='same', kernel_initializer=conv_init, activation=activation, name='icp7_out2')(icp7_reduction2)
        icp7_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same',name='icp7_pool')(icp6_out)
        # icp7_maxpool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='icp7_maxpool')(icp6_out)
        # icp7_avgpool = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='icp7_avgpool')(icp6_out)
        # icp7_pool = add([icp7_maxpool, icp7_avgpool])
        icp7_out3 = Conv2D(filters=128, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp7_out3')(icp7_pool)
        icp7_out0 = Conv2D(filters=256, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp7_out0')(icp6_out)
        icp7_out = concatenate([icp7_out0, icp7_out1, icp7_out2, icp7_out3], axis=3, name='icp7_out')

        icp8_in = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='icp8_in')(icp7_out)
        # icp8_maxin = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='icp8_maxin')(icp7_out)
        # icp8_avgin = AveragePooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='icp8_avgin')(icp7_out)
        # icp8_in = add([icp8_maxin, icp8_avgin])
        icp8_reduction1 = Conv2D(filters=160, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp8_reduction1')(icp8_in)
        icp8_out1 = Conv2D(filters=320, kernel_size=(3,3), padding='same', kernel_initializer=conv_init, activation=activation, name='icp8_out1')(icp8_reduction1)
        icp8_reduction2 = Conv2D(filters=32, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp8_reduction2')(icp8_in)
        icp8_out2 = Conv2D(filters=128, kernel_size=(5,5), padding='same', kernel_initializer=conv_init, activation=activation, name='icp8_out2')(icp8_reduction2)
        icp8_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='icp8_pool')(icp8_in)
        # icp8_maxpool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='icp8_maxpool')(icp8_in)
        # icp8_avgpool = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='icp8_avgpool')(icp8_in)
        # icp8_pool = add([icp8_maxpool, icp8_avgpool])
        icp8_out3 = Conv2D(filters=128, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp8_out3')(icp8_pool)
        icp8_out0 = Conv2D(filters=256, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp8_out0')(icp8_in)
        icp8_out = concatenate([icp8_out0, icp8_out1, icp8_out2, icp8_out3], axis=3, name='icp8_out')

        icp9_reduction1 = Conv2D(filters=192, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp9_reduction1')(icp8_out)
        icp9_out1 = Conv2D(filters=384, kernel_size=(3,3), padding='same', activation=activation, kernel_initializer=conv_init, name='icp9_out1')(icp9_reduction1)
        icp9_reduction2 = Conv2D(filters=48, kernel_size=(1,1), padding='same', kernel_initializer=conv_init, activation=activation, name='icp9_reduction2')(icp8_out)
        icp9_out2 = Conv2D(filters=128, kernel_size=(5,5), padding='same', activation=activation, kernel_initializer=conv_init, name='icp9_out2')(icp9_reduction2)
        icp9_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='icp9_pool')(icp8_out)
        # icp9_maxpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp9_maxpool')(icp8_out)
        # icp9_avgpool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='icp9_avgpool')(icp8_out)
        # icp9_pool = add([icp9_maxpool, icp9_avgpool])
        icp9_out3 = Conv2D(filters=128, kernel_size=(1,1), padding='same', activation=activation, kernel_initializer=conv_init, name='icp9_out3')(icp9_pool)
        icp9_out0 = Conv2D(filters=384, kernel_size=(1,1), padding='same', activation=activation, kernel_initializer=conv_init, name='icp9_out0')(icp8_out)
        icp9_out = concatenate([icp9_out0, icp9_out1, icp9_out2, icp9_out3], axis=3, name='icp9_out')

        # act3 = Activation(activation)(norm5)
        t3 = SeparableConv2D(filters=512, kernel_size=(1, 1), padding='same', activation=activation,
                             depth_multiplier=1, kernel_initializer=conv_init, strides=(1, 1), name='t3')(icp9_out)
        # t3 = Conv2D(filters=512, kernel_size=(1, 1), padding='same', activation=activation,
        #            kernel_initializer=conv_init, strides=(1, 1))(icp9_out)
        cls3_spp1_flat = SpatialPyramidPooling([1, 2])(t3)
        # cls3_spp1_flat = SpatialPyramidPooling([1, 2])(act3)

        # norm5 = BatchNormalization(axis=-1, name='norm5')(cls3_spp1_flat)
        # cls3_fc1 = Dense(2048, activation=activation, name='cls3_fc1', kernel_initializer=initializer)(cls3_spp1_flat)
        # cls3_drop_fc = Dropout(0.0)(cls3_spp1_flat)
        # cls3_fc_pose = Dense(7, name='cls3_fc_pose')(cls3_drop_fc)

        # cls3_fc = Dense(num_classes, activation='softmax', kernel_initializer=dense_init, name='cls3_fc')(cls3_spp1_flat)
        # cls3_fc_pose_wpqr = Dense(4, kernel_initializer=dense_init, name='cls3_fc_pose_wpqr')(cls3_spp1_flat)
        # cls3_fc_pose = Dense(7, name='cls3_fc_pose', kernel_initializer=dense_init)(cls3_spp1_flat)

        cls3_fc_pose_xyz = Dense(3, name='cls3_fc_pose_xyz')(cls3_spp1_flat)
        cls3_fc_pose_wpqr = Dense(4, name='cls3_fc_pose_wpqr')(cls3_spp1_flat)
        # cls3_fc_pose_wpqr_norm = Lambda(lambda x: K.l2_normalize(x, axis=-1), name='cls3_fc_pose_wpqr_norm')(cls3_fc_pose_wpqr)

        '''
        cls3_pool = AveragePooling2D(pool_size=(7,7), strides=(1,1), padding='valid', name='cls3_pool')(icp9_out)
        cls3_fc1_flat = Flatten(name='cls3_fc1_flat')(cls3_pool)
        cls3_fc1 = Dense(2048, kernel_initializer=initializer, activation=activation, name='cls3_fc1')(cls3_fc1_flat)
        #cls3_fc1_pose = Dense(2048, activation=activation, name='cls3_fc1_pose', kernel_initializer='random_uniform')(cls3_fc1_flat)
        cls3_drop_fc = Dropout(0.4)(cls3_fc1)
        #cls3_fc_pose = Dense(7, kernel_initializer=initializer, name='cls3_fc_pose')(cls3_drop_fc)
        cls3_fc_pose_xyz = Dense(3, kernel_initializer=initializer, name='cls3_fc_pose_xyz')(cls3_drop_fc)
        cls3_fc_pose_wpqr = Dense(4, kernel_initializer=initializer, name='cls3_fc_pose_wpqr')(cls3_drop_fc)
        '''

        # posenet = Model(inputs=input, outputs=[cls1_fc, cls2_fc, cls3_fc])

        # posenet = Model(inputs=input, outputs=[cls1_fc_pose, cls2_fc_pose, cls3_fc_pose])

        model = Model(inputs=input, outputs=[cls1_fc_pose_xyz, cls1_fc_pose_wpqr,
                                             cls2_fc_pose_xyz, cls2_fc_pose_wpqr,
                                             cls3_fc_pose_xyz, cls3_fc_pose_wpqr])

    if tune:
        if weights_path:
            weights_data = np.load(weights_path, encoding='latin1').item()
            for layer in model.layers:
                if layer.name in weights_data.keys():
                    layer_weights = weights_data[layer.name]
                    layer.set_weights((layer_weights['weights'], layer_weights['biases']))
            print("FINISHED SETTING THE WEIGHTS!")

    return model


if __name__ == "__main__":
    # print("Please run either test.py or train.py to evaluate or fine-tune the network!")
    posenet_model = create_model()
    posenet_model.summary()
    exit()
