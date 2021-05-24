import numpy as np
import cv2
import random
from tqdm import tqdm
import settings


class DataSource(object):
    def __init__(self, images, poses):
        self.images = images
        self.poses = poses


def compute_mean_image(images):
    data_size = len(images)
    X = np.zeros((settings.img_height, settings.img_width, settings.img_channels), dtype=np.float32)

    n = 0
    for i in tqdm(range(data_size)):
        x = cv2.imread(images[i])
        x = cv2.resize(x, (settings.img_width, settings.img_height), interpolation=cv2.INTER_AREA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = x.astype(np.float32)

        X[:, :, 0] += x[:, :, 0]
        X[:, :, 1] += x[:, :, 1]
        X[:, :, 2] += x[:, :, 2]
        n += 1

    mean_img = X / n
    return mean_img


def preprocess(image):
    x = cv2.imread(image)
    x = cv2.resize(x, (settings.img_width, settings.img_height), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.astype(np.float32)
    x = (x - settings.mean_array) / 255.
    return x


def get_data(dataset, directory):
    poses = []
    images = []

    with open(dataset) as f:
        next(f)  # skip the  header line
        for line in f:
            fname, p0, p1, p2, p3, p4, p5, p6 = line.split(',')
            p0 = float(p0)
            p1 = float(p1)
            p2 = float(p2)
            p3 = float(p3)
            p4 = float(p4)
            p5 = float(p5)
            p6 = float(p6)
            poses.append((p0, p1, p2, p3, p4, p5, p6))
            images.append(directory + '/' + fname)
    return DataSource(images, poses)


def generator(data, batch_size):
    batch_ind = len(data.images) - batch_size

    X = np.array(data.images)
    y = np.array(data.poses).astype(np.float32)

    indexes = np.arange(batch_ind)

    if settings.shuffle:
        random.shuffle(indexes)

    # Generate batches
    # Create zero-valued arrays to contain batch of features and labels
    batch_images = np.zeros((batch_size, settings.img_height, settings.img_width, settings.img_channels),
                            dtype=np.float32)
    batch_poses = np.zeros((batch_size, 7), dtype=np.float32)

    while True:
        for i in range(0, batch_ind, batch_size):
            index = indexes[i]

            for j in range(batch_size):
                batch_images[j] = preprocess(X[index + j])
                batch_poses[j] = y[index + j]

            batch_pose_x = batch_poses[:, :3]
            batch_pose_q = batch_poses[:, 3:]
            yield batch_images, [batch_pose_x, batch_pose_q, batch_pose_x, batch_pose_q, batch_pose_x, batch_pose_q]


def linear_decay(epoch):
    lr = 0.5 * settings.lr_0 * (1 - (epoch / settings.epochs))
    return lr
