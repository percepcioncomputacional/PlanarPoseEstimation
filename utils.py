import numpy as np
import cv2
import random
from tqdm import tqdm

img_width = 400
img_height = 225

# img_width = 640
# img_height = 360

# img_width = 224
# img_height = 224

img_channels = 3
shuffle = True
subseq_size = 100

mean_file = 'meanfile_RGB_400x225_100K_desktop_numbers.npy'
std_file = 'stdfile_RGB_400x225_100K_desktop_numbers.npy'

# mean_file = 'meanfile_RGB_400x225_100K_desktop_bicolor.npy'
# std_file = 'stdfile_RGB_400x225_100K_desktop_bicolor.npy'

# mean_file = 'meanfile_400x225_100K_RD.npy'
# std_file = 'stdfile_400x225_100K_RD.npy'
# mean_file = 'meanfile_RGB_640x360_100K_RD.npy'

# std_file = 'stdfile_RGB_640x360_100K_RD.npy'
# mean_file = 'meanfile_640x360_100K_RD.npy'

# std_file = 'stdfile_640x360_100K_RD.npy'
# mean_file = 'meanfile_kingscollege_RGB_224x224.npy'
# std_file = 'stdfile_kingscollege_RGB_224x224.npy'
epsilon = 1e-8

mean_img = np.zeros((img_height, img_width, img_channels), dtype=np.float32)
std_img = np.zeros((img_height, img_width, img_channels), dtype=np.float32)


class DataSource(object):
    # def __init__(self, images, poses, normals):
    def __init__(self, images, poses):
        self.images = images
        self.poses = poses
        # self.normals = normals


def compute_mean_image(images):
    data_size = len(images)
    X = np.zeros((img_height, img_width, img_channels), dtype=np.float32)

    n = 0

    # for i in tqdm(range(len(images))):
    for i in tqdm(range(data_size)):
        x = cv2.imread(images[i])
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (img_width, img_height), interpolation=cv2.INTER_AREA)
        # x = cv2.resize(x, (455, 256), interpolation=cv2.INTER_AREA)
        # x = centered_crop(x, 224)
        x = x.astype(np.float32)

        X[:, :, 0] += x[:, :, 0]
        X[:, :, 1] += x[:, :, 1]
        X[:, :, 2] += x[:, :, 2]

        n += 1

    mean_img = X / n
    return mean_img


def compute_std_image(images):

    data_size = len(images)
    X = np.zeros((img_height, img_width, img_channels), dtype=np.float32)

    n = 0

    for i in tqdm(range(data_size)):
        x = cv2.imread(images[i])
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = cv2.resize(x, (img_width, img_height), interpolation=cv2.INTER_AREA)
        # x = cv2.resize(x, (455, 256), interpolation=cv2.INTER_AREA)
        # x = centered_crop(x, 224)
        x = x.astype(np.float32)

        X += np.square(x - mean_img)

        n += 1

    std_img = np.sqrt(X / n)
    return std_img


def add_random_noise(img):
    m = (127., 127., 127.)
    s = (25.5, 25.5, 25.5)
    noisy = img.copy()
    noisy = cv2.randn(noisy, m, s)
    return img + noisy


def centered_crop(img, output_side_length):
    height, width, depth = img.shape
    new_height = output_side_length
    new_width = output_side_length

    if height > width:
        new_height = output_side_length * (height / width)
    else:
        new_width = output_side_length * (width / height)

    # height_offset = int((new_height - output_side_length) / 2)
    # width_offset = int((new_width - output_side_length) / 2)
    height_offset = int((new_height - output_side_length) / 2)
    width_offset = int((new_width - output_side_length) / 2)

    cropped_img = img[height_offset:height_offset + output_side_length, width_offset:width_offset + output_side_length]
    return cropped_img


def preprocess(image):
    x = cv2.imread(image)
    x = cv2.resize(x, (img_width, img_height), interpolation=cv2.INTER_AREA)
    # x = cv2.resize(x, (455, 256), interpolation=cv2.INTER_AREA)
    # x = centered_crop(x, 224)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.astype(np.float32)

    x = (x - mean_img)/255.
    # x = (x - mean_img)
    # bgr_img = (bgr_img - mean_img) / 255.0
    # bgr_img = (bgr_img - mean_img) / (std_img + epsilon)
    # bgr_img = (bgr_img - mean_img) / std_img
    return x


def get_data(dataset, directory):
    poses = []
    normals = []
    images = []

    with open(directory + dataset) as f:
        # next(f)  # skip the 3 header lines
        # next(f)
        # next(f)
        for line in f:
            fname, p0, p1, p2, p3, p4, p5, p6 = line.split(',')
            # fname, p0, p1, p2, p3, p4, p5, p6, _, _, _, _ = line.split(',')
            # fname, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, _ = line.split(',')
            # fname, p0, p1, p2, p3, p4, p5, p6, _, _, _ = line.split(',')
            p0 = float(p0)
            p1 = float(p1)
            p2 = float(p2)
            p3 = float(p3)
            p4 = float(p4)
            p5 = float(p5)
            p6 = float(p6)
            # p7 = float(p7)
            # p8 = float(p8)
            # p9 = float(p9)
            poses.append((p0, p1, p2, p3, p4, p5, p6))
            # normals.append((p7, p8, p9))
            images.append(directory + fname)
    # images_out = preprocess(images, mean_img)
    # return datasource(images_out, poses)
    # return datasource(images, poses, normals)
    return DataSource(images, poses)


def generator(data, batch_size):
    # X = np.squeeze(np.array(data.images))
    # y = np.squeeze(np.array(data.poses))
    batch_ind = len(data.images) - batch_size

    # if batch_ind % batch_size > 0:
    #    batch_ind -= batch_size

    # indexes = np.random.permutation(range(batch_ind))

    # batch_ind = np.random.permutation(range(len(data.images)))
    X = np.array(data.images)
    y = np.array(data.poses).astype(np.float32)
    # y1 = np.array(data.poses)
    # y2 = np.array(data.normals)

    # prin(np.linalg.norm(y[:, :3].max() - y[:, :3].min()))
    # print(y.max())

    indexes = np.arange(batch_ind)

    '''
    a = []
    for n in range(0, batch_ind, subseq_size):
        items = indexes[n: n + subseq_size - batch_size - 1]
        for m in items:
            a.append(m)
    indexes = a
    '''

    if(shuffle):
        random.shuffle(indexes)

    # scaler = MinMaxScaler(feature_range=(0, 1))

    # Generate batches
    # Create empty arrays to contain batch of features and labels#

    batch_images = np.zeros((batch_size, img_height, img_width, img_channels), dtype=np.float32)
    batch_poses = np.zeros((batch_size, 7), dtype=np.float32)

    while True:
        # for j in range(0, batch_ind, batch_size):
        # for i in range(0, len(indexes), batch_size):

        for i in range(0, batch_ind, batch_size):
            # batch_images = np.zeros((batch_size, img_height, img_width, img_channels))
            # batch_poses = np.zeros((batch_size, 7))

            index = indexes[i]

            for j in range(batch_size):

                batch_images[j] = preprocess(X[index + j])
                batch_poses[j] = y[index + j]

                # yield batch_images, [batch_pose_x, batch_pose_q, batch_pose_x, batch_pose_q]
                # yield [images_1, images_2], [batch_pose_x, batch_pose_q]
                # yield [images_1, images_2], batch_poses
                # yield [images_1, images_2], [batch_pose_x, batch_pose_q, batch_poses]

            batch_pose_x = batch_poses[:, :3]
            batch_pose_q = batch_poses[:, 3:]
            yield batch_images, [batch_pose_x, batch_pose_q, batch_pose_x, batch_pose_q, batch_pose_x, batch_pose_q]
            # yield batch_images, batch_poses
            # yield batch_images, [batch_poses, batch_poses, batch_poses]
            # yield batch_images, [batch_pose_x, batch_pose_q]
            # yield [images_1, images_1], [batch_pose_x, batch_pose_q, batch_normals]
