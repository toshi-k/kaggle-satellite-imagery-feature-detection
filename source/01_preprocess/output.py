
import os
import sys
import warnings

import numpy as np
import pandas as pd

from shapely.wkt import loads

from skimage.draw import polygon

import progressbar

import skimage.io as io
from skimage.io import imread, imsave
io.use_plugin('matplotlib', 'imread')

#---------------------------
# function
#---------------------------


def fill_mask(mask, polygon_array, value, W, H, Xmax, Ymin):
    po = np.array(polygon_array)

    po[:, 0] = po[:, 0] / Xmax * W * W / (W + 1)
    po[:, 1] = po[:, 1] / Ymin * H * H / (H + 1)

    rr, cc = polygon(po[:, 1], po[:, 0], shape=mask.shape)
    mask[rr, cc] = value

    return mask


def fill_polygon(poly, mask, W, H, Xmax, Ymin):
    polygon_array = poly.exterior.coords
    mask = fill_mask(mask, polygon_array, 1, W, H, Xmax, Ymin)

    if hasattr(poly, "interiors"):
        for interior in poly.interiors:
            polygon_array = interior.coords
            mask = fill_mask(mask, polygon_array, 0, W, H, Xmax, Ymin)

    return mask

#---------------------------
# main
#---------------------------

train_wkt = pd.read_csv('../../dataset/train_wkt_v4.csv')

train_images = train_wkt.ImageId.unique()

for cla in range(1, 11):

    print('class index: ' + str(cla) + ' --------------------')
    sys.stdout.flush()

    # make output directory
    output_dir_path = os.path.join('../../input/train_output_class{0:d}'.format(cla))
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    p_bar = progressbar.ProgressBar(max_value=len(train_images))

    for i, train_image in enumerate(train_images):

        p_bar.update(i+1)

        target_wkt = train_wkt.ix[train_wkt.ImageId == train_image, :]
        target_wkt.reset_index(inplace=True, drop=True)

        grid_sizes = pd.read_csv('../../dataset/grid_sizes.csv')
        grid_size = grid_sizes.ix[grid_sizes.iloc[:, 0] == train_image, :]
        Xmax = float(grid_size.Xmax)
        Ymin = float(grid_size.Ymin)

        img = imread('../../dataset/three_band/' + train_image + '.tif')

        W = img.shape[1]
        H = img.shape[0]

        mask = np.zeros((H, W), dtype='float')

        target = target_wkt.MultipolygonWKT[target_wkt.ClassType == cla].tolist()[0]

        if target != 'MULTIPOLYGON EMPTY':

            multi = loads(target)

            for poly in multi:
                mask = fill_polygon(poly, mask, W, H, Xmax, Ymin)

        output_file_name = train_image + '.png'

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(os.path.join(output_dir_path, output_file_name), mask)

    p_bar.finish()
