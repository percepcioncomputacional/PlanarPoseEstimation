import os
import math
from spp.SpatialPyramidPooling import SpatialPyramidPooling

import utils
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import initializers
from keras.models import Model

import posenet
import posenet_spp

from keras.layers import Dense, Dropout, Reshape, Lambda, Conv2D, GlobalAvgPool2D, concatenate, \
    BatchNormalization, GlobalMaxPool2D, Flatten, Input, MaxoutDense, MaxPool2D, Activation, SeparableConv2D

from keras import backend as K


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
    config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
    sess = tf.Session(config=config)
    K.set_session(sess)

    directory = '/home/vocegued/SeqData/'
    # directory = '/home/vocegued/KingsCollege/'
    # train_directory = '/home/vocegued/RandomData/'
    # val_directory = '/home/vocegued/RandomData/'

    # dataset_train = 'train.csv'
    # dataset_val = 'validation.csv'

    # dataset_test = 'dataset_test.txt'
    dataset_test = 'test_2K.csv'

    mean_file = utils.mean_file
    std_file = utils.std_file

    # trained_weights = 'E6_checkpoint_weights.h5'
    trained_weights = '400x225_dot_loss_checkpoint_weights.h5'

    # trained_weights = 'posenet.npy'

    batch_size = 1

    img_width = utils.img_width
    img_height = utils.img_height
    img_channels = utils.img_channels
    # input_shape_1 = (360, 640, 3)
    # input_shape_2 = (225, 400, 3)
    input_shape = (img_height, img_width, img_channels)
    csv_file = 'test_data.csv'
    utils.shuffle = False
    # act = 'relu'

    '''
    posenet = posenet.create_posenet()
    # t1 = posenet.get_layer(name='cls1_reduction_pose').output
    t1 = posenet.get_layer(name='icp3_out').output
    t1 = SeparableConv2D(filters=256, kernel_size=(1, 1), padding='same', activation=act,
                         strides=(1, 1))(t1)
    t1 = SpatialPyramidPooling([2])(t1)
    # t1 = GlobalAvgPool2D()(t1)

    # t1 = BatchNormalization(name='norm3')(t1)
    x1 = Dense(3, name='cls1_fc_pose_xyz')(t1)
    q1 = Dense(4, name='cls1_fc_pose_wpqr')(t1)

    # t2 = posenet.get_layer(name='cls2_reduction_pose').output
    t2 = posenet.get_layer(name='icp6_out').output
    t2 = SeparableConv2D(filters=256, kernel_size=(1, 1), padding='same', activation=act,
                         strides=(1, 1))(t2)
    t2 = SpatialPyramidPooling([2])(t2)
    # t2 = GlobalAvgPool2D()(t2)
    # t2 = BatchNormalization(name='norm4')(t2)
    x2 = Dense(3, name='cls2_fc_pose_xyz')(t2)
    q2 = Dense(4, name='cls2_fc_pose_wpqr')(t2)

    t3 = posenet.get_layer(name='icp9_out').output
    t3 = SeparableConv2D(filters=512, kernel_size=(1, 1), padding='same', activation=act,
                         strides=(1, 1))(t3)
    t3 = SpatialPyramidPooling([2])(t3)
    # t3 = GlobalAvgPool2D()(t3)
    # t3 = BatchNormalization(name='norm5')(t3)
    x3 = Dense(3, name='cls3_fc_pose_xyz')(t3)
    q3 = Dense(4, name='cls3_fc_pose_wpqr')(t3)

    model = Model(inputs=posenet.input, outputs=[x1, q1, x2, q2, x3, q3])
    '''

    # model = posenet.create_model()
    # model = posenet.create_model('posenet.npy', True)
    model = posenet_spp.create_model()
    model.load_weights(trained_weights)

    model.summary()
    print("Inputs: {}".format(model.input_shape))
    print("Outputs: {}".format(model.output_shape))

    data_test = utils.get_data(dataset_test, directory)

    if not os.path.exists(mean_file):
        print('Error: missing mean file.')
        exit()
    else:
        mean_img = np.load(mean_file)

    utils.mean_img = mean_img

    if not os.path.exists(std_file):
        print('Error: missing std file.')
        exit()
    else:
        std_img = np.load(std_file)

    utils.std_img = std_img

    data_size = len(data_test.images)

    training = np.zeros((data_size, 8))
    prediction_error = np.zeros((data_size, 2), dtype=np.float32)

    for i in range(0, data_size, 1):

        pose_x = np.asarray(data_test.poses[i][0:3])
        pose_q = np.asarray(data_test.poses[i][3:7])
        pose_x = np.squeeze(pose_x)
        pose_q = np.squeeze(pose_q)

        x1 = utils.preprocess(data_test.images[i])
        x1 = np.expand_dims(x1, axis=0)

        y_pred = model.predict(x1, verbose=1)
        predicted_x = np.squeeze(y_pred[4])
        predicted_q = np.squeeze(y_pred[5])
        
        q1 = pose_q / np.linalg.norm(pose_q)

        # Compute Individual Sample Error
        error_x = np.linalg.norm(pose_x - predicted_x)
        q2 = predicted_q / np.linalg.norm(predicted_q)

        d = abs(np.sum(np.multiply(q1, q2)))
        theta = 2 * np.arccos(d) * 180 / math.pi

        training[i][0:1] = i + 1
        training[i][1:4] = pose_x
        training[i][4:8] = pose_q

        prediction_error[i, :] = [error_x, theta]

        print ('Iteration:  ', i, '  Error XYZ (cm):  ', error_x, '  Error Q (degrees):  ', theta)

    # np.savetxt('E1_test_results.csv', training, delimiter=',')
    np.savetxt('E6_test_error.csv', prediction_error, delimiter=',')

    mean_result = np.mean(prediction_error, axis=0)
    median_result = np.median(prediction_error, axis=0)
    std_result = np.std(prediction_error, axis=0)
    print('Number of samples: ', data_size)
    print('Mean position error: ', mean_result[0], 'm')
    print('Median position value: ', median_result[0], 'm')
    print()
    # print('Standard deviation of position: ', std_result[0], 'm')
    print('Mean rotation error: ', mean_result[1], 'degrees')
    print('Median rotation value:  ', median_result[1], 'degrees')
    # print('Standard deviation of rotation:  ', std_result[1], 'degrees')

    '''
    p = plt.boxplot([prediction_error[:, 0]])
    r = plt.boxplot([prediction_error[:, 1]])

    p_whiskers = p['whiskers'][1].get_data()[1]
    r_whiskers = r['whiskers'][1].get_data()[1]
    print('Top whiskers range (position): ', p_whiskers)
    print('Top whiskers range (rotation): ', r_whiskers)

    p_outliers = p['fliers'][0].get_data()[0]
    r_outliers = r['fliers'][0].get_data()[0]
    print('Number of position outliers: ', p_outliers.size)
    print('Number of rotation outliers: ', r_outliers.size)
    '''

    # boxprops = dict(linestyle='-', linewidth=1, color='black')
    # bplot = ax.boxplot(data, showmeans=False, showfliers=False, labels=['PoseNet', 'Ours'])
    # colors = ['blue', 'green']

    font = {'family': 'sans',
            'color': 'black',
            'weight': 'normal',
            'size': 14,
            }

    # Multiple box plots on one Axes
    fig, ax = plt.subplots()
    plt.ylabel('Centimeters', fontdict=font)
    ax.set_xticklabels(['Position', 'Orientation'], fontsize=14)
    bp = ax.boxplot(prediction_error, labels=['Position', 'Orientation'], showfliers=True, patch_artist=True)
    plt.show()


    '''
    # ax = fig.add_subplot(111, frameon=False)
    ax.set_ylabel('Centimeters', fontdict=font)

    ax1 = fig.add_subplot(121)
    ax1.set_xlabel('PoseNet', fontdict=font)
    ax1.boxplot([prediction_error[:, 0]], labels=[''], showfliers=False, patch_artist=True)

    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('Ours', fontdict=font)
    ax2.boxplot([prediction_error[:, 1]], labels=[''], showfliers=False, patch_artist=True)
    '''


    '''
    plt.figure()
    plt.boxplot([prediction_error[:, 0]], labels=[""], showfliers=False, patch_artist=True)
    plt.xlabel('Position', fontdict=font)
    plt.ylabel('Centimeters', fontdict=font)
    plt.figure()
    plt.boxplot([prediction_error[:, 1]], labels=[""], showfliers=False, patch_artist=True)
    plt.xlabel('Orientation', fontdict=font)
    plt.ylabel('Degrees', fontdict=font)
    '''

    # Visualize the data
    plt.show()
    K.clear_session()

