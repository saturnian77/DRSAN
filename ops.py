import tensorflow as tf
import numpy as np
import imageio
import math
from glob import glob

def imread(path):
    img = imageio.imread(path).astype(np.float64)
    return img / 255.


def rgb2y(x):
    if x.dtype == np.uint8:
        x = np.float64(x)
        y = 65.481 / 255. * x[:, :, 0] + 128.553 / 255. * x[:, :, 1] + 24.966 / 255. * x[:, :, 2] + 16
        y = np.round(y).astype(np.uint8)
    else:
        y = 65.481 / 255. * x[:, :, 0] + 128.553 / 255. * x[:, :, 1] + 24.966 / 255. * x[:, :, 2] + 16 / 255.
    return y


def psnr_(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 'error'
    if np.max(img1) <= 1.0:
        MAX_VAL = 1.0
    else:
        MAX_VAL = 255.0
    return 20.0 * math.log10(MAX_VAL / math.sqrt(mse))


def imgcut(x, xN):
    h, w, c = x.shape
    x = x[xN:h - xN, xN:w - xN, :]
    return x


def count_param():
    param = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print("Model Parameter: %02.2f M" % (param / 1000000.0))


def imagewrite(img, iter, scale, dataset):
    img = img.astype(np.uint8)
    img = img[0, :, :, :]
    imageio.imsave('./Results/' + dataset + '_' + scale + '_' + ('%03d' % (iter + 1)) + '.png', img)


def load_testimg(dataset, xN):
    lab_path = './psnrtest/HR/' + dataset + '/*.png'
    data_path = './psnrtest/' + xN + '/' + dataset + '/*.png'
    img_list = np.sort(np.asarray(glob(data_path)))
    lab_list = np.sort(np.asarray(glob(lab_path)))
    k = len(img_list)
    imgs = {}
    labs = {}
    for i in range(k):
        img = imread(img_list[i])
        lab = imread(lab_list[i])

        if len(img.shape) < 3:
            img = np.expand_dims(img, axis=2)
            lab = np.expand_dims(lab, axis=2)
            img = np.concatenate((np.concatenate((img, img), axis=2), img), axis=2)
            lab = np.concatenate((np.concatenate((lab, lab), axis=2), lab), axis=2)
        imgs[i] = img
        labs[i] = lab

    return imgs, labs, k

##

def pReLU(_x, name):
    alpha = tf.get_variable(name + 'alpha', _x.get_shape()[-1],
                            initializer=tf.constant_initializer(0.0),
                            dtype=tf.float32)
    out = tf.nn.relu(_x) + alpha * (_x - abs(_x)) * 0.5
    return out


def resmod(feat, param_in, param_out, blockname):
    inp = tf.layers.conv2d(feat, param_in, 1, padding='same')
    inp = pReLU(inp, blockname + 'prelu')
    inp = tf.layers.conv2d(inp, param_out, 1, padding='same')
    out = tf.reduce_mean(inp, axis=[1, 2], keepdims=True)

    return out


def upsample_x2(inputs, feature):
    outputs = tf.layers.conv2d(inputs, feature * 4, 3, padding='same')
    outputs = tf.depth_to_space(outputs, 2)
    return outputs


def upsample_x3(inputs, feature):
    outputs = tf.layers.conv2d(inputs, feature * 9, 3, padding='same')
    outputs = tf.depth_to_space(outputs, 3)
    return outputs


def upsample_x4(inputs, feature):
    outputs = tf.layers.conv2d(inputs, feature * 4, 3, padding='same')
    outputs = tf.depth_to_space(outputs, 2)
    outputs = tf.layers.conv2d(outputs, feature * 4, 3, padding='same')
    outputs = tf.depth_to_space(outputs, 2)
    return outputs