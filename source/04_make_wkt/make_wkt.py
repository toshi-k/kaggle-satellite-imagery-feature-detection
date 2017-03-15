
import os
import re
import time
import math
import warnings

import numpy as np
import pandas as pd

import json
import geojson

from shapely.geometry import shape
from shapely.geometry import MultiPolygon, Polygon
from shapely.affinity import scale
from shapely.wkt import dumps

from skimage.draw import polygon
from skimage.io import imread, imsave
from skimage.transform import rescale

from multiprocessing import Process, cpu_count
import multiprocessing
import Queue

import progressbar

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


def gen_mask_image(multi, img, W, H, Xmax, Ymin):

    mask = np.zeros(img.shape, dtype='float')

    if type(multi) == Polygon:
        # Polygon
        mask = fill_polygon(multi, mask, W, H, Xmax, Ymin)
    else:
        # Multi polygon
        for poly in multi:
            mask = fill_polygon(poly, mask, W, H, Xmax, Ymin)

    return mask


def calc_diff(multi, img, W, H, Xmax, Ymin, name=""):

    mask = gen_mask_image(multi, img, W, H, Xmax, Ymin)
    diff = math.sqrt(np.mean((mask - img/255.0) ** 2))
    # print("name: " + name + " diff: {0:.3f}".format(diff))
    
    return diff


def preview_mask(multi, img, W, H, Xmax, Ymin, name=""):

    mask = gen_mask_image(multi, img, W, H, Xmax, Ymin)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imsave('_temp/' + name + '_wkt.png', mask)
        imsave('_temp/' + name + '_original.png', img)


def img_to_wkt(img, scale_rate, target_name, W, H, Xmax, Ymin, t_value, cla):

    W_dash = W * W / (W + 1)
    H_dash = H * H / (H + 1)

    if scale_rate < 0.99:
        img_tiny = rescale(img, scale_rate)
    else:
        img_tiny = img

    bmp_image_path = '_temp/' + target_name + '.bmp'
    target_json_path = '_temp/' + target_name + '.json'

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imsave(bmp_image_path, img_tiny)

    os.system('potrace -a 2 -t ' + str(t_value) + ' -b geojson -i ' + bmp_image_path + ' -o ' + target_json_path)

    f = open(target_json_path)
    data = json.load(f)
    f.close()

    os.remove(target_json_path)
    os.remove(bmp_image_path)

    # type of 'data' is feature collection 
    # we only need focus on features
    features = data['features']

    list_polygons = list()

    for i in range(len(features)):

        shapely_polygon = shape(geojson.loads(json.dumps(features[i]['geometry'])))

        if scale_rate < 0.99:
            shapely_polygon = scale(shapely_polygon, 1/scale_rate, 1/scale_rate, origin=(0, 0))

        list_polygons.append(shapely_polygon.buffer(0.0))

    multi = MultiPolygon(list_polygons)

    multi = scale(multi, 1, -1, 1, origin=(float(W)/2, float(H)/2))
    multi = scale(multi, Xmax / W_dash, Ymin / H_dash, origin=(0, 0))

    if cla != 6:
        multi = multi.simplify(1e-6, preserve_topology=True)
    else:
        multi = multi.simplify(1e-5, preserve_topology=True)

    multi = multi.buffer(0)

    if multi.type == 'Polygon':
        multi = MultiPolygon([multi])

    return multi

# tpex's evaluation code can validate topology more strictly
# https://github.com/cxz/tpex
def is_valid_wkt(multi, p_num=0):

    # wkt_string = dumps(multi)

    # wkt_tmp_name = '_temp/wkt_temp_' + str(p_num) + '.txt'
    # wkt_temp_f = open(wkt_tmp_name, 'w')
    # wkt_temp_f.write(wkt_string)
    # wkt_temp_f.close()

    # result = commands.getoutput('java -cp' + 
    #     ' tpex/target/tpex-1.0-SNAPSHOT-jar-with-dependencies.jar' +
    #     ' io.github.cxz.tpex.App ' + wkt_tmp_name)

    # result = bool(int(result))
    # result = not result

    # return result

    return multi.is_valid


def process_job(files, output_f, cla, p_num=0):

    while 1:

        try:
            i, file = files.get(timeout=1)
        except Queue.Empty:
            break

        p_bar.update(i+1)

        target_image_path = os.path.join(target_class_directory, file)
        img = imread(target_image_path)

        if cla == 9 or cla == 10:
            ref_class_directory = os.path.join('../../submission/submission_ave',
                                               'class7_valid0.651_threshold0.11')
            ref_image_path = os.path.join(ref_class_directory, file)

            before = np.sum(img)

            ref = imread(ref_image_path)
            if np.sum(ref) > 500000:
                img = np.zeros(img.shape)
            after = np.sum(img)

            print("before:{0:10.0f} after:{1:10.0f}".format(before, after))

        target_name = file.split('.')[0]
        Xmax = grid_size.loc[target_name, :].Xmax
        Ymin = grid_size.loc[target_name, :].Ymin

        if np.mean(img[:100, :100])/255.0 > 0.999:
            img[:20, :20] = 0

        if np.mean(img[-100:, :100])/255.0 > 0.999:
            img[-20:, :20] = 0

        if np.mean(img[:100, -100:])/255.0 > 0.999:
            img[:20, -20:] = 0

        if np.mean(img[-100:, -100:])/255.0 > 0.999:
            img[-20:, -20:] = 0

        W = img.shape[1]
        H = img.shape[0]

        scale_rate = 1.0 / 2.0

        if cla in [1, 9, 10]:
            scale_rate = 1.0

        if (cla == 2) and (target_name != '6100_0_2'):
            scale_rate = 1.0

        diff = 99999.0

        while scale_rate > (1.0 / 32):

            t = 20.0 / scale_rate
            multi = img_to_wkt(img, scale_rate, target_name, W, H, Xmax, Ymin, t, cla)

            multi_is_valid = multi.is_valid
            result = is_valid_wkt(multi, p_num)

            if multi_is_valid != result:
                print('shapely: ' + str(multi_is_valid) + ' result: ' + str(result))
                multi_is_valid = result

            diff = calc_diff(multi, img, W, H, Xmax, Ymin, target_name)
            # print("\tdiff: {0:.3f}".format(diff))

            if multi_is_valid and diff < 0.25:
                break

            scale_rate /= 2

        if diff > 0.25:
            print("large diff in {} !".format(target_name))
            print(diff)

        if multi_is_valid:
            preview_mask(multi, img, W, H, Xmax, Ymin, target_name)

        if (str(multi) == 'GEOMETRYCOLLECTION EMPTY') or (not multi_is_valid):
            line = target_name + ',' + str(cla) + ',MULTIPOLYGON EMPTY' + '\n'
        else:
            line = target_name + ',' + str(cla) + ',\"' + dumps(multi) + '\"\n'

        output_f.write(line)

    output_f.flush()

#---------------------------
# main
#---------------------------

tic = time.time()

if not os.path.exists('_temp'):
    os.makedirs('_temp')

grid_size = pd.read_csv('../../dataset/grid_sizes.csv')
grid_size.index = grid_size.iloc[:, 0]

target_directory = '../../submission/submission_ave/'

num_valid_all = 0
num_class = 0

target_classes = range(1, 11)

if target_classes == range(1, 11):
    output_filename = 'wkt_class_all'
else:
    seq_class = '_'.join([str(v) for v in target_classes])
    output_filename = 'wkt_class_' + seq_class

output_filenames = []
for p_num in range(cpu_count()):
    p_output_filename = output_filename + '_' + str(p_num) + '.csv'
    output_f = os.path.join(target_directory, p_output_filename)
    output_filenames.append(output_f)

output_fs = [open(output_file, 'w') for output_file in output_filenames]

scores = []

re_compiled = re.compile("^.+_valid(.+)_threshold(.+).*$")

for cla in target_classes:

    num_class += 1

    print("class: " + str(cla) + " ---------")

    dirs = os.listdir(target_directory)

    sel = [d for d in dirs if re.match("class" + str(cla) + "_", d)]
    assert len(sel) > 0, "target class directory is not exist"
    assert len(sel) == 1, "target class directory is not unique"

    target_class_directory = os.path.join(target_directory, sel[0])

    score = float(re.sub(re_compiled, "\\1", target_class_directory))
    scores.append(score)

    files = os.listdir(target_class_directory)

    print('\ttarget directory: {}'.format(target_class_directory))

    process_files = multiprocessing.Queue()
    for i, fi in enumerate(files):
        process_files.put((i, fi))

    p_bar = progressbar.ProgressBar(max_value=len(files))
    p_bar.start()

    jobs = []
    for mod in range(cpu_count()):
        jobs.append(Process(target=process_job, args=(process_files, output_fs[mod], cla, p_bar)))

    for j in jobs:
        j.start()

    for j in jobs:
        j.join()

for output_f in output_fs:
    output_f.close()

valid_score_mean = float(sum(scores)) / len(scores)
output_filename_new = output_filename + "_valid{0:.3f}".format(valid_score_mean)

os.system('cat ' + ' '.join(output_filenames) + ' > ' + os.path.join(target_directory, output_filename_new + '.csv'))

[os.remove(path) for path in output_filenames]

computational_time = (time.time() - tic) / 60.0
print("\tcomputational time: {0:.0f} [min]".format(computational_time))
