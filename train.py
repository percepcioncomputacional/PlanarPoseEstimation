from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import os
import posenet
import pnspp
import custom_callbacks
from loss_functions import euc_loss1x, euc_loss2x, euc_loss3x, \
    euc_loss1q, euc_loss2q, euc_loss3q, frob_loss1r, frob_loss2r, frob_loss3r
import settings
import utils


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.8)
    # config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=False, log_device_placement=False)
    # sess = tf.Session(config=config)
    # K.set_session(sess)

    # Variables
    BATCH_SIZE = settings.BATCH_SIZE
    epochs = settings.epochs
    BETA = settings.BETA
    BETA_list = settings.BETA_list
    prev_k = settings.prev_k
    optimizer = settings.optimizer
    image_dir = settings.image_dir
    training_loss_file = settings.training_loss_file
    validation_loss_file = settings.validation_loss_file
    BETA_file = settings.BETA_file
    labels_file = settings.labels_file
    mean_file = settings.mean_file
    checkpoint_weights = settings.checkpoint_weights
    trained_weights = settings.trained_weights

    # Data
    data = utils.get_data(labels_file, image_dir)
    data_train = utils.DataSource(data.images[0:100000], data.poses[0:100000])
    data_val = utils.DataSource(data.images[100000:120000], data.poses[1000:120000])

    # data_test = utils.DataSource(data.images[120000:140000], data.poses[120000:140000])
    # data_train.images, data_train.poses = skl_shuffle(data_train.images, data_train.poses, random_state=0

    if not os.path.exists(mean_file):
        mean_img = utils.compute_mean_image(data_train.images)
        np.save(mean_file, mean_img)
    else:
        mean_img = np.load(mean_file)
    settings.mean_array = mean_img

    # PoseNet / PNSPP models (comment/uncomment accordingly)
    model = posenet.create_model()
    # model = pnspp.create_model()

    # L1 / L2 loss functions (comment/uncomment accordingly)
    # Loss function L1
    model.compile(optimizer=optimizer, loss={'cls1_fc_pose_xyz': euc_loss1x, 'cls1_fc_pose_wpqr': euc_loss1q,
                                             'cls2_fc_pose_xyz': euc_loss2x, 'cls2_fc_pose_wpqr': euc_loss2q,
                                             'cls3_fc_pose_xyz': euc_loss3x, 'cls3_fc_pose_wpqr': euc_loss3q})

    '''
    # Loss function L2
    model.compile(optimizer=optimizer, loss={'cls1_fc_pose_xyz': euc_loss1x, 'cls1_fc_pose_wpqr': frob_loss1r,
                                             'cls2_fc_pose_xyz': euc_loss2x, 'cls2_fc_pose_wpqr': frob_loss2r,
                                             'cls3_fc_pose_xyz': euc_loss3x, 'cls3_fc_pose_wpqr': frob_loss3r})
    '''
    model.summary()
    print("Inputs: {}".format(model.input_shape))
    print("Outputs: {}".format(model.output_shape))

    # Setup checkpointing
    chkp = ModelCheckpoint(filepath=checkpoint_weights, verbose=1, save_best_only=True)
    print()

    # Setup loss history
    lh = custom_callbacks.LossHistory(BETA, BETA_list, prev_k, chkp)

    # Setup learning decay policy
    ls = LearningRateScheduler(utils.linear_decay)

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

    model.save_weights(trained_weights)

    # Summarize history for loss
    # Get training and test loss histories
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']

    np.savetxt(training_loss_file, history.history['loss'], delimiter=',')
    np.savetxt(validation_loss_file, history.history['val_loss'], delimiter=',')
    np.savetxt(BETA_file, BETA_list, delimiter=',')

    # Epochs
    epoch_range = range(1, len(history.epoch) + 1, 1)
    plt.plot(epoch_range, training_loss)
    plt.plot(epoch_range, test_loss)
    plt.plot(epoch_range, BETA_list)
    plt.title('Model loss')
    plt.ylabel('loss/BETA')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation', 'BETA'], loc='upper right')
    plt.show()
