import numpy as np
from keras.optimizers import Adam
from keras import initializers

# Input image dims
img_width = 400
img_height = 225
img_channels = 3

# Training variables
lr_0 = 1e-4
n_lr = 0.0
optimizer = Adam(lr=lr_0)
conv_init = initializers.glorot_uniform(seed=1)
dense_init = initializers.glorot_uniform(seed=1)
activation = 'relu'
BATCH_SIZE = 8
epochs = 20
seed = 10
shuffle = True
BETA = 1.
BETA_list = []
prev_k = 1.
thr = 0.02
epochs_proportion = 0.5

# Data / results
mean_array = np.zeros((img_height, img_width, img_channels), dtype=np.float32)
mean_file = 'D1_mean_img.npy' # rename accordingly (dataset)
labels_file = 'D1_labels.csv'  # rename accordingly (labels file)
image_dir = '/media/pp/a04a9aa9-a7a5-418e-a995-2492bca016c6/home/michelle/data/D1'  # rename accordingly (dataset)
checkpoint_weights = 'E1_checkpoint_weights.h5'  # rename accordingly (experiment)
trained_weights = 'E1_trained_weights.h5'  # rename accordingly (experiment)
best_weights = 'E1_best_weights.h5'  # rename accordingly (experiment)
training_loss_file = 'E1_training_loss.csv'  # rename accordingly (experiment)
validation_loss_file = 'E1_validation_loss.csv'  # rename accordingly (experiment)
error_test_file = 'E1_error_test.csv' # rename accordingly (experiment)
BETA_file = 'E1_BETAS.csv'  # rename accordingly (experiment)
