import utils
import posenet
# import posenet_spp_classifier
import tfquaternion as tfq
import posenet_spp

import numpy as np
from spp.SpatialPyramidPooling import SpatialPyramidPooling
from keras import backend as K
from keras.applications import Xception
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Input, BatchNormalization, Activation, Flatten, GlobalMaxPool2D, \
    GlobalAvgPool2D, SeparableConv2D
from keras.optimizers import Adam, SGD, Adadelta, Adagrad, TFOptimizer, Nadam, RMSprop
from keras.callbacks import ModelCheckpoint, TensorBoard, LambdaCallback
# from keras.regularizers import l1, l2
from keras import initializers
import matplotlib.pyplot as plt
from keras.callbacks import Callback, EarlyStopping, LearningRateScheduler
# from sklearn.utils import shuffle as skl_shuffle
import tensorflow as tf
import math
import os
# import gc


# warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class LossHistory(Callback):

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_train_end(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

    def on_epoch_end(self, epoch, logs={}):
        # global ALPHA
        global BETA
        global curr_k
        global prev_k
        global val_loss
        global best
        # global thr
        # global sess
        global n_lr

        # loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        best = chkp.__dict__.get('best')

        x_loss = logs.get('cls3_fc_pose_xyz_loss')
        # q_loss = logs.get('cls3_fc_pose_wpqr_norm_loss')
        q_loss = logs.get('cls3_fc_pose_wpqr_loss')
        # k = x_loss / q_loss
        curr_k = x_loss / q_loss

        # val_x_loss = logs.get('val_cls3_fc_pose_xyz_loss')
        # val_q_loss = logs.get('val_cls3_fc_pose_wpqr_loss')

        optimizer = self.model.optimizer
        n_lr = K.eval(self.model.optimizer.lr)

        margin = abs(curr_k - prev_k)
        # if nb_epoch > 1:
        # if (val_loss > best) and (margin <= thr):
        if epoch >= int(0.5 * epochs) and (val_loss > best) and (margin <= thr):
            BETA = curr_k * BETA
            # gc.collect()
            # del model

            # self.model.compile(optimizer=optimizer, loss={'cls1_fc_pose_xyz': euc_loss1x,
            #                                              'cls1_fc_pose_wpqr': riemann_loss1r,
            #                                              'cls2_fc_pose_xyz': euc_loss2x,
            #                                              'cls2_fc_pose_wpqr': riemann_loss2r,
            #                                              'cls3_fc_pose_xyz': euc_loss3x,
            #                                              'cls3_fc_pose_wpqr': riemann_loss3r})

            self.model.compile(optimizer=optimizer, loss={'cls1_fc_pose_xyz': euc_loss1x,
                                                          'cls1_fc_pose_wpqr': frob_loss1r,
                                                          'cls2_fc_pose_xyz': euc_loss2x,
                                                          'cls2_fc_pose_wpqr': frob_loss2r,
                                                          'cls3_fc_pose_xyz': euc_loss3x,
                                                          'cls3_fc_pose_wpqr': frob_loss3r})

            # self.model.compile(optimizer=optimizer, loss={'cls1_fc_pose_xyz': euc_loss1x,
            #                                              'cls1_fc_pose_wpqr': euc_loss1q,
            #                                              'cls2_fc_pose_xyz': euc_loss2x,
            #                                              'cls2_fc_pose_wpqr': euc_loss2q,
            #                                              'cls3_fc_pose_xyz': euc_loss3x,
            #                                              'cls3_fc_pose_wpqr': euc_loss3q})

            self.model.load_weights(checkpoint_weights)

        prev_k = curr_k
        BETAS.append(BETA)
        print('thr = ', thr)
        print('BETA = ', BETA)
        print('lr = ', n_lr)


'''
def euc_loss1x(y_true, y_pred):
    # lx = K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1))
    # y_true = K.reshape(y_true, [-1, y_pred.shape[1]])
    lx = tf.norm((y_true - y_pred), ord='euclidean', axis=-1)
    return 0.3 * ALPHA * lx


def riemann_loss1r(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1, y_pred.shape[1]])

    y_true = tfq.Quaternion(y_true)
    y_true = y_true.as_rotation_matrix()

    y_pred = tfq.Quaternion(y_pred)
    y_pred = y_pred.as_rotation_matrix()

    r = tf.matmul(y_true, y_pred, transpose_b=True)
    r = (tf.trace(r) - 1) / 2.
    # r = matmul(y_true, y_pred, transpose_a=True)
    # r = abs(log(r))
    r = tf.clip_by_value(r, -1 + K.epsilon(), 1 - K.epsilon())
    # r = K.clip(r, -1, 1)

    return 0.3 * BETA * tf.acos(r)
    # return 0.3 * BETA * r


def euc_loss2x(y_true, y_pred):
    # lx = K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1))
    # y_true = K.reshape(y_true, [-1, y_pred.shape[1]])
    lx = tf.norm((y_true - y_pred), ord='euclidean', axis=-1)
    return 0.3 * ALPHA * lx


def riemann_loss2r(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1, y_pred.shape[1]])

    y_true = tfq.Quaternion(y_true)
    y_true = y_true.as_rotation_matrix()

    y_pred = tfq.Quaternion(y_pred)
    y_pred = y_pred.as_rotation_matrix()

    r = tf.matmul(y_true, y_pred, transpose_b=True)
    r = (tf.trace(r) - 1) / 2.
    # r = matmul(y_true, y_pred, transpose_a=True)
    # r = abs(log(r))
    r = tf.clip_by_value(r, -1 + K.epsilon(), 1 - K.epsilon())
    # r = K.clip(r, -1, 1)

    return 0.3 * BETA * tf.acos(r)
    # return 0.3 * BETA * r


def euc_loss3x(y_true, y_pred):
    # lx = K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1))
    # y_true = K.reshape(y_true, [-1, y_pred.shape[1]])
    lx = tf.norm((y_true - y_pred), ord='euclidean', axis=-1)
    return ALPHA * lx


def riemann_loss3r(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1, y_pred.shape[1]])

    y_true = tfq.Quaternion(y_true)
    y_true = y_true.as_rotation_matrix()

    y_pred = tfq.Quaternion(y_pred)
    y_pred = y_pred.as_rotation_matrix()

    r = tf.matmul(y_true, y_pred, transpose_b=True)
    r = (tf.trace(r) - 1) / 2.
    # r = matmul(y_true, y_pred, transpose_a=True)
    # r = abs(log(r))
    r = tf.clip_by_value(r, -1 + K.epsilon(), 1 - K.epsilon())
    # r = K.clip(r, -1, 1)

    return BETA * tf.acos(r)
    # return BETA * r
'''


def euc_loss1x(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:, :] - y_pred[:, :]), axis=-1, keepdims=True))
    return 0.3 * ALPHA * lx


def frob_loss1r(y_true, y_pred):
    # lq = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=-1, keepdims=True))
    y_true = K.reshape(y_true, [-1, y_pred.shape[-1]])
    y_true = tfq.Quaternion(y_true)
    m_true = y_true.as_rotation_matrix()

    y_pred = tfq.Quaternion(y_pred)
    m_pred = y_pred.as_rotation_matrix()

    return 0.3 * BETA * tf.norm((m_true - m_pred), ord='fro', axis=[-2, -1], keepdims=True)
    # return 0.3 * BETA * lq


def euc_loss2x(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:, :] - y_pred[:,:]), axis=-1, keepdims=True))
    return 0.3 * ALPHA * lx


def frob_loss2r(y_true, y_pred):
    # lq = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=-1, keepdims=True))
    y_true = K.reshape(y_true, [-1, y_pred.shape[-1]])
    y_true = tfq.Quaternion(y_true)
    m_true = y_true.as_rotation_matrix()

    y_pred = tfq.Quaternion(y_pred)
    m_pred = y_pred.as_rotation_matrix()

    return 0.3 * BETA * tf.norm((m_true - m_pred), ord='fro', axis=[-2, -1], keepdims=True)
    # return 0.3 * BETA * lq


def euc_loss3x(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:, :] - y_pred[:,:]), axis=-1, keepdims=True))
    return ALPHA * lx


def frob_loss3r(y_true, y_pred):
    # lq = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=-1, keepdims=True))
    y_true = K.reshape(y_true, [-1, y_pred.shape[-1]])
    y_true = tfq.Quaternion(y_true)
    m_true = y_true.as_rotation_matrix()

    y_pred = tfq.Quaternion(y_pred)
    m_pred = y_pred.as_rotation_matrix()

    return BETA * tf.norm((m_true - m_pred), ord='fro', axis=[-2, -1], keepdims=True)
    # return BETA * lq


'''
def euc_loss1x(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:, :] - y_pred[:, :]), axis=-1, keepdims=True))
    return 0.3 * ALPHA * lx


def euc_loss1q(y_true, y_pred):
    lq = K.sqrt(K.sum(K.square(y_true[:, :] - y_pred[:, :]), axis=-1, keepdims=True))
    # lq = 0.5 * (1 - K.sum(y_true * y_pred, axis=-1, keepdims=True))
    return 0.3 * BETA * lq


def euc_loss2x(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:, :] - y_pred[:, :]), axis=-1, keepdims=True))
    return 0.3 * ALPHA * lx


def euc_loss2q(y_true, y_pred):
    lq = K.sqrt(K.sum(K.square(y_true[:, :] - y_pred[:, :]), axis=-1, keepdims=True))
    # lq = 0.5 * (1 - K.sum(y_true * y_pred, axis=-1, keepdims=True))
    return 0.3 * BETA * lq


def euc_loss3x(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:, :] - y_pred[:, :]), axis=-1, keepdims=True))
    return ALPHA * lx


def euc_loss3q(y_true, y_pred):
    lq = K.sqrt(K.sum(K.square(y_true[:, :] - y_pred[:, :]), axis=-1, keepdims=True))
    # lq = 0.5 * (1 - K.sum(y_true * y_pred, axis=-1, keepdims=True))
    return BETA * lq
'''


def step_decay(epoch):
    initial_lr = 1e-4
    drop = 0.5
    epochs_drop = 0.1 * epochs
    lrate = initial_lr * math.pow(drop, math.floor((1 + epoch)/epochs_drop))
    return lrate


def linear_decay(epoch):
    initial_lr = 1e-4
    lrate = 0.5 * initial_lr * (1 - (epoch / epochs))
    # lrate = initial_lr * (1 - (epoch / epochs))
    return lrate


if __name__ == "__main__":

    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.8)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=False, log_device_placement=False)

    sess = tf.Session(config=config)
    K.set_session(sess)

    # Variables
    BATCH_SIZE = 8
    # window_size = 1
    epochs = 50
    BETAS = []
    curr_k = 1.
    prev_k = 1.

    # BETA = K.variable(1., dtype='float')
    ALPHA = 1.
    BETA = 1.
    sw = 0
    val_loss = 0.
    best = 0.
    # thr = 0.025
    thr = 0.02
    initial_thr = 1.

    # conv_init = initializers.he_uniform(seed=1)
    # conv_init = initializers.he_uniform(seed=1)
    # dense_init = initializers.orthogonal(seed=1)
    # act = 'relu'

    train_directory = '/home/michelle/ImageSet/'
    val_directory = '/home/michelle/ImageSet/'

    dataset_train = 'train.csv'
    dataset_val = 'validation.csv'

    # dataset_train = 'dataset_train.txt'
    # dataset_val = 'dataset_test.txt'

    mean_file = utils.mean_file
    std_file = utils.std_file

    checkpoint_weights = 'E8_pnspp_checkpoint_weights.h5'
    # checkpoint_weights = '400x225_dot_loss_checkpoint_weights.h5'
    # cnn_initial_weights = 'PoseNet-SPP_places365_standard_weights.h5'
    # cnn_trained_weights = '400x225_dot_loss_posenet-spp_weights.h5'
    trained_weights = 'E8_pnspp_trained_weights.h5'
    img_width = utils.img_width
    img_height = utils.img_height
    img_channels = utils.img_channels
    input_shape = (img_height, img_width, img_channels)
    # input_sequence = Input(batch_shape=(batch_size, window_size, img_height, img_width, img_channels))
    # input = Input(shape=(img_height, img_width, img_channels))
    utils.shuffle = True
    lr = 1e-4
    n_lr = 0.0
    decay = lr / epochs
    # optimizer = Nadam(lr=lr)
    # optimizer = SGD(lr=lr, momentum=0.85, nesterov=True)
    optimizer = Adam(lr=lr)
    conv_init = initializers.glorot_uniform(seed=1)
    # dense_init = initializers.orthogonal(seed=1)
    dense_init = initializers.glorot_uniform(seed=1)
    act = 'relu'

    '''
    constant = np.zeros((batch_size, 1, 4), dtype=int)
    for i in range(0, batch_size):
        constant[i, :] = [0, 0, 0, 1]
    constant = K.variable(constant)
    '''

    data_train = utils.get_data(dataset_train, train_directory)
    # data_train.images, data_train.poses = skl_shuffle(data_train.images, data_train.poses, random_state=0)

    data_val = utils.get_data(dataset_val, val_directory)
    # data_val.images, data_val.poses = skl_shuffle(data_val.images, data_val.poses, random_state=0)

    if not os.path.exists(mean_file):
        mean_img = utils.compute_mean_image(data_train.images)
        np.save(mean_file, mean_img)
    else:
        mean_img = np.load(mean_file)
    utils.mean_img = mean_img

    if not os.path.exists(std_file):
        std_img = utils.compute_std_image(data_train.images)
        np.save(std_file, std_img)
    else:
        std_img = np.load(std_file)
    utils.std_img = std_img

    # model = posenet_spp.create_posenet('posenet.npy', True)  # GoogLeNet (Trained on Places
    # posenet = posenet.create_model('posenet.npy', True)  # GoogLeNet (Trained on Places)

    # model = posenet.create_posenet('posenet.npy', True)  # GoogLeNet (Trained on Places)
    # posenet.summary()
    # exit()

    # model = posenet.create_model()
    model = posenet_spp.create_model()

    # posenet_spp = posenet_spp_classifier.create_posenet(None, False, 365)
    # posenet_spp.load_weights(cnn_initial_weights)
    # model = Xception(weights='imagenet', input_shape=(utils.img_height, utils.img_width, utils.img_channels))
    model.summary()
    exit()

    '''
    t1 = posenet.get_layer(name='icp3_out').output
    t1 = SeparableConv2D(filters=256, kernel_size=(1, 1), padding='same', activation=act,
                         depth_multiplier=1, kernel_initializer=conv_init, strides=(1, 1))(t1)
    t1 = SpatialPyramidPooling([1, 2])(t1)
    x1 = Dense(3, kernel_initializer=dense_init, name='cls1_fc_pose_xyz')(t1)
    q1 = Dense(4, kernel_initializer=dense_init, name='cls1_fc_pose_wpqr')(t1)

    t2 = posenet.get_layer(name='icp6_out').output
    t2 = SeparableConv2D(filters=256, kernel_size=(1, 1), padding='same', activation=act,
                         depth_multiplier=1, kernel_initializer=conv_init, strides=(1, 1))(t2)
    t2 = SpatialPyramidPooling([1, 2])(t2)
    x2 = Dense(3, kernel_initializer=dense_init, name='cls2_fc_pose_xyz')(t2)
    q2 = Dense(4, kernel_initializer=dense_init, name='cls2_fc_pose_wpqr')(t2)

    t3 = posenet.get_layer(name='icp9_out').output
    t3 = SeparableConv2D(filters=512, kernel_size=(1, 1), padding='same', activation=act,
                         kernel_initializer=conv_init, strides=(1, 1))(t3)
    t3 = SpatialPyramidPooling([1, 2])(t3)
    x3 = Dense(3, kernel_initializer=dense_init, name='cls3_fc_pose_xyz')(t3)
    q3 = Dense(4, kernel_initializer=dense_init, name='cls3_fc_pose_wpqr')(t3)

    model = Model(inputs=posenet.input, outputs=[x1, q1, x2, q2, x3, q3])
    '''

    # model.load_weights(checkpoint_weights)

    # for layer in model.layers[:10]:
    #    layer.trainable = False

    # model.compile(optimizer=nadam, loss={'x': position_loss, 'q': rotation_loss})
    # model.compile(optimizer=nadam, loss={'cls1_fc_pose': pose_loss_1(constant),
    #                                     'cls2_fc_pose': pose_loss_2(constant),
    #                                     'cls3_fc_pose': pose_loss_3(constant)})

    # model.compile(optimizer=optimizer, loss={'cls1_fc_pose_xyz': euc_loss1x, 'cls1_fc_pose_wpqr': riemann_loss1r,
    #                                         'cls2_fc_pose_xyz': euc_loss2x, 'cls2_fc_pose_wpqr': riemann_loss2r,
    #                                         'cls3_fc_pose_xyz': euc_loss3x, 'cls3_fc_pose_wpqr': riemann_loss3r})

    model.compile(optimizer=optimizer, loss={'cls1_fc_pose_xyz': euc_loss1x, 'cls1_fc_pose_wpqr': frob_loss1r,
                                             'cls2_fc_pose_xyz': euc_loss2x, 'cls2_fc_pose_wpqr': frob_loss2r,
                                             'cls3_fc_pose_xyz': euc_loss3x, 'cls3_fc_pose_wpqr': frob_loss3r})

    # model.compile(optimizer=optimizer, loss={'cls1_fc_pose_xyz': euc_loss1x, 'cls1_fc_pose_wpqr': euc_loss1q,
    #                                         'cls2_fc_pose_xyz': euc_loss2x, 'cls2_fc_pose_wpqr': euc_loss2q,
    #                                         'cls3_fc_pose_xyz': euc_loss3x, 'cls3_fc_pose_wpqr': euc_loss3q})

    # model.compile(optimizer=optimizer, loss=pose_loss)

    # x1 = Dense(3, kernel_initializer=Constant(value=0.0075), name='x')(x)
    # x2 = Dense(4, kernel_initializer=Constant(value=0.0075), name='q')(x)

    '''
    conv_branch = Model(inputs=net.input, outputs=x)
    branch_a = Input(shape=input_shape)
    branch_b = Input(shape=input_shape)

    processed_a = conv_branch(branch_a)
    processed_b = conv_branch(branch_b)

    regression = concatenate([processed_a, processed_b])
    # regression = Flatten()(regression) # may not be necessary
    output = Dense(7, kernel_initializer='normal', name='output')(regression)
    model = Model(inputs=[branch_a, branch_b], outputs=[output])
    
    model.compile(loss=custom_objective, optimizer=adam, metrics=['accuracy'])
    '''

    # for layer in model.layers:
    #    print(layer, layer.trainable)
    # exit()

    # plot_model(cnn_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    # plot_model(cnn_model, to_file='model_plot.png')
    # cnn_model.summary()
    # print("Inputs: {}".format(cnn_model.input_shape))
    # print("Outputs: {}".format(cnn_model.output_shape))
    # exit()

    # x = cnn_model.get_layer(index=38).output

    # for layer in cnn_model.layers[:]:
    #    layer.trainable = False

    '''
    x = TimeDistributed(cnn_model)(input_sequence)
    x = Bidirectional(GRU(128, return_sequences=False))(x)
    #x = GRU(512, activation='relu', return_sequences=False)(x)

    x3 = Dense(3, kernel_initializer='normal', name='x')(x)
    x4 = Dense(4, kernel_initializer='normal', name='q')(x)
    '''

    # model = Model(inputs=net.input, outputs=x)
    # model = Model(inputs=input_sequence, outputs=[x3, x4])
    # model.load_weights(cnn_gru_initial_weights)
    # model.compile(optimizer=adam, loss=custom_objective)

    model.summary()
    print("Inputs: {}".format(model.input_shape))
    print("Outputs: {}".format(model.output_shape))

    # Setup checkpointing
    chkp = ModelCheckpoint(filepath=checkpoint_weights, verbose=1, save_best_only=True)
    print()

    # earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
    # tb = TensorBoard(log_dir='./logs', histogram_freq=10, batch_size=batch_size, write_graph=False,
    #                 write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None,
    #                 embeddings_metadata=None)

    # Setup loss history
    lh = LossHistory()

    # Setup learning decay policy
    # lr_scheduler = LearningRateScheduler(linear_decay)
    ls = LearningRateScheduler(linear_decay)

    # plot_losses = PlotLosses()
    train_size = len(data_train.images)
    val_size = len(data_val.images)

    history = model.fit_generator(utils.generator(data_train, BATCH_SIZE),
                                  steps_per_epoch=train_size // BATCH_SIZE,
                                  validation_data=utils.generator(data_val, BATCH_SIZE),
                                  validation_steps=val_size // BATCH_SIZE,
                                  shuffle=True,
                                  epochs=epochs,
                                  verbose=1,
                                  callbacks=[chkp, lh, ls])

    print()

    model.save_weights(trained_weights)
    # model.save_weights(cnn_gru_trained_weights)

    # summarize history for loss
    # Get training and test loss histories
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']

    np.savetxt('E8_pnspp_training_loss.csv', history.history['loss'], delimiter=',')
    np.savetxt('E8_pnspp_validation_loss.csv', history.history['val_loss'], delimiter=',')
    np.savetxt('E8_pnspp_BETAS.csv', BETAS, delimiter=',')

    # Create count of the number of epochs
    epoch_range = range(1, len(history.epoch) + 1, 1)

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.plot(epoch_range, training_loss)
    plt.plot(epoch_range, test_loss)
    plt.plot(epoch_range, BETAS)
    # plt.plot(epoch_range, lrs)
    plt.title('Model loss')
    plt.ylabel('loss/BETA')
    # plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation', 'BETA'], loc='upper right')
    # plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
