import os
import random
import sys
import warnings

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed


def read_images(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, TRAIN_PATH, TEST_PATH):
    train_ids = os.listdir(TRAIN_PATH)
    test_ids = os.listdir(TEST_PATH)

    # Get and resize train images and masks
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):

        path = os.path.join(TRAIN_PATH, id_)
        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask

    # Get and resize test images
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    sizes_test = []
    print('Getting and resizing test images ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = os.path.join(TEST_PATH, id_)
        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[n] = img
    print('Done!')

    return X_train, Y_train, X_test, sizes_test


if __name__ == '__main__':
    IMG_WIDTH = 128
    IMG_HEIGHT = 128
    IMG_CHANNELS = 3
    path = r'C:\Users\omri\Personal\Kaggle\Project'
    TRAIN_PATH = os.path.join(path, 'stage1_train')  # '"C:\Users\omri\Personal\Kaggle\Project\stage1_train/'
    TEST_PATH = '"C:\Users\omri\Personal\Kaggle\Project\stage1_test'

    read_images(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, TRAIN_PATH, TEST_PATH)
