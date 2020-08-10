from keras import backend as K
import tensorflow as tf
import numpy as np
import os
import math
import posenet
import pnspp
import utils
import settings


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
    # config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
    # sess = tf.Session(config=config)
    # K.set_session(sess)

    image_dir = settings.image_dir
    mean_file = settings.mean_file
    best_weights = settings.best_weights
    batch_size = 1

    img_width = settings.img_width
    img_height = settings.img_height
    img_channels = settings.img_channels
    input_shape = (img_height, img_width, img_channels)
    labels_file = settings.labels_file

    # PoseNet/PNSPP models (comment/uncomment accordingly)
    model = posenet.create_model()
    # model = pnspp.create_model()
    model.load_weights(best_weights)

    model.summary()
    print()
    print("Inputs: {}".format(model.input_shape))
    print("Outputs: {}".format(model.output_shape))

    data = utils.get_data(labels_file, image_dir)
    data_test = utils.DataSource(data.images[120000:140000], data.poses[120000:140000])

    if not os.path.exists(mean_file):
        print('Error: missing mean file.')
        exit()
    else:
        mean_img = np.load(mean_file)
    settings.mean_array = mean_img

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

        # Compute individual sample error
        error_x = np.linalg.norm(pose_x - predicted_x)
        q2 = predicted_q / np.linalg.norm(predicted_q)

        d = abs(np.sum(np.multiply(q1, q2)))
        theta = 2 * np.arccos(d) * 180 / math.pi

        training[i][0:1] = i + 1
        training[i][1:4] = pose_x
        training[i][4:8] = pose_q
        prediction_error[i, :] = [error_x, theta]
        print('Iteration:  ', i, '  Error XYZ (cm):  ', error_x, '  Error Q (degrees):  ', theta)

    np.savetxt(settings.error_test_file, prediction_error, delimiter=',')

    print()
    mean_result = np.mean(prediction_error, axis=0)
    median_result = np.median(prediction_error, axis=0)
    std_result = np.std(prediction_error, axis=0)
    print('Number of samples: ', data_size)
    print('Mean position error: ', mean_result[0], 'm')
    print('Median position value: ', median_result[0], 'm')
    print()
    print('Mean rotation error: ', mean_result[1], 'degrees')
    print('Median rotation value:  ', median_result[1], 'degrees')

