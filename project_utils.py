import sys


def config_paths(user, env_name):
    paths = ['',
             '/home/{0}/{1}/.env/bin'.format(user, env_name),
             '/usr/lib/python35.zip',
             '/usr/lib/python3.5',
             '/usr/lib/python3.5/plat-x86_64-linux-gnu',
             '/usr/lib/python3.5/lib-dynload',
             '/home/{0}/{1}/.env/lib/python3.5/site-packages'.format(user, env_name),
             '/home/{0}/{1}/.env/lib/python3.5/site-packages/IPython/extensions'.format(user, env_name),
             '/home/{0}/.ipython']

    for path in paths:
        sys.path.append(path)


config_paths('omri', 'my_gpu')
from skimage.morphology import label
from skimage.io import imread, imshow, imread_collection, concatenate_images

import pandas as pd
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import numpy as np
from skimage.transform import resize
import warnings
import random
import os
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


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def predict_results(model_name, X_train, X_test, sizes_test):
    model = load_model(model_name, custom_objects={'mean_iou': mean_iou})
    preds_train = model.predict(X_train[:int(X_train.shape[0] * 0.9)], verbose=1)
    preds_val = model.predict(X_train[int(X_train.shape[0] * 0.9):], verbose=1)
    preds_test = model.predict(X_test, verbose=1)

    # Threshold predictions
    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)
    preds_test_t = (preds_test > 0.5).astype(np.uint8)

    # Create list of upsampled test masks
    preds_test_upsampled = []
    for i in range(len(preds_test)):
        preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                           (sizes_test[i][0], sizes_test[i][1]),
                                           mode='constant', preserve_range=True))

    return preds_test_upsampled


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


def create_submission_file(preds_test_upsampled, test_ids, filename):
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(test_ids):
        rle = list(prob_to_rles(preds_test_upsampled[n]))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))

    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv('{}.csv'.format(filename, index=False))
    return sub
