
# Waterway [0.095 LB]
# https://www.kaggle.com/resolut/dstl-satellite-imagery-feature-detection/waterway-0-095-lb

from __future__ import division

import os

import numpy as np
import pandas as pd

import cv2
import tifffile as tiff
from skimage.transform import resize

import progressbar

#---------------------------
# function
#---------------------------


def CCCI_index(m, rgb):
    RE  = resize(m[5, :, :], (rgb.shape[0], rgb.shape[1]))
    MIR = resize(m[7, :, :], (rgb.shape[0], rgb.shape[1]))
    R = rgb[:, :, 0]
    # canopy chloropyll content index
    CCCI = (MIR-RE)/(MIR+RE)*(MIR-R)/(MIR+R)
    return CCCI


def thresholding(CCCI, threshold):
    # you can look on histogram and pick your favorite threshold value(0.11 is my best)
    binary = (CCCI > threshold).astype(np.float32)
    # print('num of pixel: {0:d}'.format(int(np.sum(binary))))
    if np.sum(binary) <= 500000:
        # print('\t=> clean !')
        binary = np.zeros(binary.shape)
    return binary

#---------------------------
# main
#---------------------------

data = pd.read_csv('../dataset/train_wkt_v4.csv')
data = data[data.MultipolygonWKT != 'MULTIPOLYGON EMPTY']

threshold = 0.11

# train --------------------

print('=> Predict for train images')

tp = 0.0
fp = 0.0
fn = 0.0

for IM_ID in data[data.ClassType == 7].ImageId:

    print('\t{}'.format(IM_ID))

    # read rgb and m bands
    rgb = tiff.imread('../dataset/three_band/{}.tif'.format(IM_ID))
    rgb = np.rollaxis(rgb, 0, 3)
    m = tiff.imread('../dataset/sixteen_band/{}_M.tif'.format(IM_ID))
    
    truth = cv2.imread('../input/train_output_class7/{}.png'.format(IM_ID),
                       cv2.IMREAD_GRAYSCALE).astype(np.float32)
    truth = np.divide(truth, 255.0)

    # get our index
    CCCI = CCCI_index(m, rgb) 

    # thresholding
    binary = thresholding(CCCI, threshold)

    tp += np.sum(truth * binary)
    fp += np.sum((1 - truth) * binary)
    fn += np.sum(truth * (1 - binary))

train_score = tp / (tp + fp + fn)
print('train score: {0:.3f}'.format(train_score))

# test ---------------------

print('=> Predict for test images')

# make output directory
output_dir = os.path.join('../submission/submission_ave',
                          'class7_valid{0:.3f}_threshold{1:.3f}'.format(train_score, threshold))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

sample_submit = pd.read_csv('../dataset/sample_submission.csv')
test_names = sample_submit.ImageId.unique().tolist()

p = progressbar.ProgressBar(max_value=len(test_names))
p.start()

for count, IM_ID in enumerate(test_names):

    p.update(count + 1)

    # read rgb and m bands
    rgb = tiff.imread('../dataset/three_band/{}.tif'.format(IM_ID))
    rgb = np.rollaxis(rgb, 0, 3)
    m = tiff.imread('../dataset/sixteen_band/{}_M.tif'.format(IM_ID))

    # get our index
    CCCI = CCCI_index(m, rgb) 

    # thresholding
    binary = thresholding(CCCI, threshold)

    binary *= 255.0
    cv2.imwrite(os.path.join(output_dir, '{}.png'.format(IM_ID)), binary)
