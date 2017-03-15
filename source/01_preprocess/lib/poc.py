
import math

import numpy as np

from skimage import transform as tf
from skimage.io import imread

import progressbar

#------------------------------
# function
#------------------------------


def poc_v1(img1, img2):
    img1_fft = np.fft.fft2(img1)
    img2_fft = np.conj(np.fft.fft2(img2))

    diff = img1_fft * img2_fft / np.absolute(img1_fft) / np.absolute(img2_fft)
    position = np.real(np.fft.ifft2(diff))

    # imsave("result.png", position)

    x = np.argmax(position) % position.shape[1]
    y = np.argmax(position) // position.shape[1]

    x_int = x
    y_int = y

    # subpixel (x)
    if x_int != 0 and x_int+1 != position.shape[1]:
        h = float(position[y_int, x_int-1] - position[y_int, x_int+1])
        c = float(2*position[y_int, x_int-1] - 4*position[y_int, x_int] + 2*position[y_int, x_int+1])
        x += h/c

    # subpixel (y)
    if y_int != 0 and y_int+1 != position.shape[0]:
        h = float(position[y_int-1, x_int] - position[y_int+1, x_int])
        c = float(2*position[y_int-1, x_int] - 4*position[y_int, x_int] + 2*position[y_int+1, x_int])
        y += h/c

    if x > img1.shape[1]/2:
        x = x - img1.shape[1]

    if y > img1.shape[0]/2:
        y = y - img1.shape[0]

    return x, y


def poc_v2(img1, img2):
    img1_fft = np.fft.fft2(img1)
    img2_fft = np.conj(np.fft.fft2(img2))

    diff = img1_fft * img2_fft / np.absolute(img1_fft) / np.absolute(img2_fft)
    position = np.real(np.fft.ifft2(diff))

    c1 = (position.shape[0]+int(1))/2
    c2 = (position.shape[1]+int(1))/2

    d1 = position.shape[0] - c1
    d2 = position.shape[1] - c2

    pos1 = position[:c1, :c2]
    pos2 = position[:c1, c2:]
    pos3 = position[c1:, :c2]
    pos4 = position[c1:, c2:]

    position = np.vstack([np.hstack([pos4, pos3]),
                          np.hstack([pos2, pos1])])

    x = np.argmax(position) % position.shape[1]
    y = np.argmax(position) // position.shape[1]

    # imsave("result.png", position)

    x_int = x
    y_int = y

    # subpixel (x)
    if x_int != 0 and x_int+1 != position.shape[1]:
        h = float(position[y_int, x_int-1] - position[y_int, x_int+1])
        c = float(2*position[y_int, x_int-1] - 4*position[y_int, x_int] + 2*position[y_int, x_int+1])
        x += h/c

    # subpixel (y)
    if y_int != 0 and y_int+1 != position.shape[0]:
        h = float(position[y_int-1, x_int] - position[y_int+1,x_int])
        c = float(2*position[y_int-1, x_int] - 4*position[y_int, x_int] + 2*position[y_int+1, x_int])
        y += h/c

    x -= d2
    y -= d1

    return x, y


def test(test_func):

    lena = imread('lena512.png')

    n = 100

    error_all = np.zeros([n])
    pbar = progressbar.ProgressBar(max_value=n)

    for i in range(n):

        pbar.update(i+1)

        x_true = np.random.random()*6-5
        y_true = np.random.random()*6-5

        # ex) left:5, up:30 => translation=(5, 30)
        t_form = tf.SimilarityTransform(translation=(x_true, y_true))
        lena_shift = tf.warp(lena, t_form)

        a1 = np.random.randint(10, 50)
        a2 = np.random.randint(10, 50)
        a3 = np.random.randint(10, 50)
        a4 = np.random.randint(10, 50)

        img1 = lena[a1:-a2, a3:-a4]
        img2 = lena_shift[a1:-a2, a3:-a4]

        x_est, y_est = test_func(img1, img2)

        # print("x: {0:.3f}, x: {0:.3f}".format(x_true, y_true))
        # print("x: {0:.3f}, y: {0:.3f}".format(x_est, y_est))

        value = math.sqrt((x_true - x_est)**2 + (y_true - y_est)**2)
        error_all[i] = value

    ave = np.average(error_all)
    std = np.std(error_all)

    print("\terror: {0:.3f} +- {1:.3f}".format(ave, std))

#------------------------------
# main
#------------------------------

if __name__ == '__main__':

    print('=> Test poc_v1')
    test(poc_v1)

    print('=> Test poc_v2')
    test(poc_v2)
