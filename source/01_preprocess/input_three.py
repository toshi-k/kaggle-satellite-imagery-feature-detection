
import os

import numpy as np
import pandas as pd

import progressbar

from skimage.io import imread, imsave

#---------------------------
# function
#---------------------------


# amaia's normalizing method
# https://www.kaggle.com/aamaia/dstl-satellite-imagery-feature-detection/rgb-using-m-bands-example
def stretch_8bit(bands, lower_percent=2, higher_percent=98):

    out = np.zeros_like(bands)
    for i in range(3):
        a = 0 
        b = 255 
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t

    return out.astype(np.uint8)


def select_and_save(file_names, output_dir):

    # make output directory
    output_dir_path = os.path.join('../../input', output_dir)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    p = progressbar.ProgressBar(max_value=len(file_names))

    for i, name in enumerate(file_names):

        p.update(i+1)

        file_name = name + '.tif'
        img = imread(os.path.join('../../dataset/three_band', file_name))
        img = img.astype("float")

        img = stretch_8bit(img)

        output_file_name = name + '.png'
        imsave(os.path.join(output_dir_path, output_file_name), img)

#---------------------------
# main
#---------------------------

# train --------------------

train_wkt = pd.read_csv('../../dataset/train_wkt_v4.csv')
train_names = train_wkt.ImageId.unique().tolist()

print('=> Preprocess train images')
select_and_save(train_names, 'train_input_three')

# test ---------------------

sample_submit = pd.read_csv('../../dataset/sample_submission.csv')
test_names = sample_submit.ImageId.unique().tolist()

print('=> Preprocess test images')
select_and_save(test_names, 'test_input_three')
