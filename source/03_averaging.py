
import os
import re
import warnings

import numpy as np

from skimage.io import imread, imsave

import progressbar

#---------------------------
# main
#---------------------------

input_dir = '../submission/submission_pre'
output_dir = '../submission/submission_ave'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

re_compiled = re.compile("^.+_valid(.+)_threshold(.+)_.+$")

for cla in [1,2,3,4,5,6,8,9,10]:

    print('class: {0:d}'.format(cla))

    dirs = os.listdir(input_dir)
    sel = [d for d in dirs if re.match("class" + str(cla) + "_", d)]
    print("target directories")
    print(sel)

    scores = [float(re.sub(re_compiled, "\\1", s)) for s in sel]
    score = np.mean(scores)

    thresholds = [float(re.sub(re_compiled, "\\2", s)) for s in sel]
    threshold = np.mean(thresholds)

    new_dir = 'class{0:d}_valid{1:.3f}_threshold{2:.3f}'.format(cla, score, threshold)

    output_dir_single = os.path.join(output_dir, new_dir)
    if not os.path.exists(output_dir_single):
        os.makedirs(output_dir_single)

    files = os.listdir(os.path.join(input_dir, sel[0]))
    # print(files)

    p = progressbar.ProgressBar(max_value=len(files))
    p.start()

    for count_file, file in enumerate(files):

        p.update(count_file + 1)

        img = imread(os.path.join(input_dir, sel[0], file)) / 255.0

        for i in range(1, len(sel), 1):
            img = img + imread(os.path.join(input_dir, sel[i], file)) / 255.0

        img = np.divide(img, len(sel))

        img = img > threshold
        img = img.astype('uint8')
        img = img * 255

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(os.path.join(output_dir_single, file), img)
