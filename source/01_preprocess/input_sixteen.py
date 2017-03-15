
import os

import numpy as np
import pandas as pd

import cv2
import tifffile as tiff

from skimage import transform as tf

import progressbar

from lib.poc import poc_v2 as poc

#---------------------------
# function
#---------------------------


def normalize(img):

    img = img - np.average(img)
    img = img / np.std(img)

    img *= 30
    img += 127

    return img.astype(np.uint8)


def _align_two_rasters(img1, img2):

    p1 = normalize(img1[10:-10, 10:-10, 0].astype(np.float32))
    p2 = normalize(img2[10:-10, 10:-10, 7].astype(np.float32))

    x, y = poc(p2, p1)
    print('x: {0:.5f} y: {1:.5f}'.format(x, y))

    t_form = tf.SimilarityTransform(translation=(x, y))
    img3 = tf.warp(img2, t_form)

    return img3


def select_and_save(file_names, output_dir):

    # make output directory
    output_dir_path = os.path.join('../../input', output_dir)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    p = progressbar.ProgressBar(max_value=len(file_names))

    for i, name in enumerate(file_names):

        p.update(i+1)

        image_id = name
        img_3 = np.transpose(tiff.imread("../../dataset/three_band/{}.tif".format(image_id)), (1, 2, 0))
        img_a = np.transpose(tiff.imread("../../dataset/sixteen_band/{}_A.tif".format(image_id)), (1, 2, 0))

        raster_size = img_a.shape
        img_3 = cv2.GaussianBlur(img_3.astype(np.float32), (11, 11), 4, 4)
        img_3 = cv2.resize(img_3, (raster_size[1], raster_size[0]), interpolation=cv2.INTER_CUBIC)
        img_3 = img_3[:, :, [2, 1, 0]]

        img_a_new = _align_two_rasters(img_3, img_a)
        img_a_new *= (2 ** 9)

        output_file_name = name + '.npy'
        np.save(os.path.join(output_dir_path, output_file_name), np.transpose(img_a_new, (2, 0, 1)))

#---------------------------
# main
#---------------------------

# train --------------------

train_wkt = pd.read_csv('../../dataset/train_wkt_v4.csv')
train_names = train_wkt.ImageId.unique().tolist()

print('=> Preprocess train images')
select_and_save(train_names, 'train_input_sixteen')

# test ---------------------

sample_submit = pd.read_csv('../../dataset/sample_submission.csv')
test_names = sample_submit.ImageId.unique().tolist()

print('=> Preprocess test images')
select_and_save(test_names, 'test_input_sixteen')
