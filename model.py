import tensorflow as tf
import numpy as np
import os
from ops import pReLU, resmod, upsample_x2, upsample_x3, upsample_x4

class DRSAN32(object):
    def __init__(self, x, len, scale):
        self.conv = x
        self.length = len
        self.scale = scale
        self.output = self.build_model()
        
        

    def build_model(self):
        with tf.variable_scope('DRAN'):
            # params
            block_len = self.length
            channel_n = 32

            scale = self.scale
            K = 0

            conv = tf.layers.conv2d(self.conv, channel_n, 3, padding='same')

            convg = conv

            ##
            for i in range(block_len):
                bname = 'conv_block%d' % (i + 1)

                res_att = resmod(conv, 16, 10, bname)
                conv0 = conv

                conv = pReLU(conv, bname + '_pReLU_1')
                conv = tf.layers.conv2d(conv, channel_n, 3, padding='same')
                conv = pReLU(conv, bname + '_pReLU_2')
                conv = tf.layers.conv2d(conv, channel_n, 3, padding='same')

                addterm = conv0 * res_att[:, :, :, 0, None]
                sig1 = tf.nn.sigmoid(conv)
                conv1 = (conv + addterm) * sig1
                conv = conv1

                #####

                conv = pReLU(conv, bname + '_pReLU_3')
                conv = tf.layers.conv2d(conv, channel_n, 3, padding='same')
                conv = pReLU(conv, bname + '_pReLU_4')
                conv = tf.layers.conv2d(conv, channel_n, 3, padding='same')

                addterm = conv0 * res_att[:, :, :, 1, None] + conv1 * res_att[:, :, :, 2, None]
                sig2 = tf.nn.sigmoid(conv)
                conv2 = (conv + addterm) * sig2
                conv = conv2

                #####

                conv = pReLU(conv, bname + '_pReLU_5')
                conv = tf.layers.conv2d(conv, channel_n, 3, padding='same')
                conv = pReLU(conv, bname + '_pReLU_6')
                conv = tf.layers.conv2d(conv, channel_n, 3, padding='same')

                addterm = conv0 * res_att[:, :, :, 3, None] + conv1 * res_att[:, :, :, 4, None] + conv2 * res_att[:, :, :, 5, None]
                sig3 = tf.nn.sigmoid(conv)
                conv3 = (conv + addterm) * sig3
                conv = conv3

                #####

                conv = pReLU(conv, bname + '_pReLU_7')
                conv = tf.layers.conv2d(conv, channel_n, 3, padding='same')
                conv = pReLU(conv, bname + '_pReLU_8')
                conv = tf.layers.conv2d(conv, channel_n, 3, padding='same')

                addterm = conv0 * res_att[:, :, :, 6, None] + conv1 * res_att[:, :, :, 7, None] + conv2 * res_att[:, :, :, 8, None] + conv3 * res_att[:, :, :, 9, None]
                sig4 = tf.nn.sigmoid(conv)
                conv4 = (conv + addterm) * sig4
                conv = conv4

                ####
                conv = tf.concat([conv0, conv1, conv2, conv3, conv4], 3)
                conv = tf.layers.conv2d(conv, channel_n, 1, padding='same')

            ##
            conv = tf.layers.conv2d(conv, channel_n, 3, padding='same')
            conv = conv + convg

            ## Upsampler
            if self.scale == 'x2':
                conv = upsample_x2(conv, channel_n)
            elif self.scale == 'x3':
                conv = upsample_x3(conv, channel_n)
            elif self.scale == 'x4':
                conv = upsample_x4(conv, channel_n)
            ##

            out = tf.layers.conv2d(conv, 3, 3, padding='same')

        return out

class DRSAN48(object):
    def __init__(self, x, len, scale):
        self.conv = x
        self.length = len
        self.scale = scale
        self.output = self.build_model()

    def build_model(self):
        with tf.variable_scope('DRAN'):
            # params
            block_len = self.length
            channel_n = 48

            scale = self.scale
            K = 0

            conv = tf.layers.conv2d(self.conv, channel_n, 3, padding='same')

            convg = conv

            ##
            for i in range(block_len):
                bname = 'conv_block%d' % (i + 1)

                res_att = resmod(conv, 16, 6, bname)
                conv0 = conv

                conv = pReLU(conv, bname + '_pReLU_1')
                conv = tf.layers.conv2d(conv, channel_n, 3, padding='same')
                conv = pReLU(conv, bname + '_pReLU_2')
                conv = tf.layers.conv2d(conv, channel_n, 3, padding='same')

                addterm = conv0 * res_att[:, :, :, 0, None]
                sig1 = tf.nn.sigmoid(conv)
                conv1 = (conv + addterm) * sig1
                conv = conv1

                #####

                conv = pReLU(conv, bname + '_pReLU_3')
                conv = tf.layers.conv2d(conv, channel_n, 3, padding='same')
                conv = pReLU(conv, bname + '_pReLU_4')
                conv = tf.layers.conv2d(conv, channel_n, 3, padding='same')

                addterm = conv0 * res_att[:, :, :, 1, None] + conv1 * res_att[:, :, :, 2, None]
                sig2 = tf.nn.sigmoid(conv)
                conv2 = (conv + addterm) * sig2
                conv = conv2

                #####

                conv = pReLU(conv, bname + '_pReLU_5')
                conv = tf.layers.conv2d(conv, channel_n, 3, padding='same')
                conv = pReLU(conv, bname + '_pReLU_6')
                conv = tf.layers.conv2d(conv, channel_n, 3, padding='same')

                addterm = conv0 * res_att[:, :, :, 3, None] + conv1 * res_att[:, :, :, 4, None] + conv2 * res_att[:, :, :, 5, None]
                sig3 = tf.nn.sigmoid(conv)
                conv3 = (conv + addterm) * sig3
                conv = conv3

                ####
                conv = tf.concat([conv0, conv1, conv2, conv3], 3)
                conv = tf.layers.conv2d(conv, channel_n, 1, padding='same')

            ##
            conv = tf.layers.conv2d(conv, channel_n, 3, padding='same')
            conv = conv + convg

            ## Upsampler
            if self.scale == 'x2':
                conv = upsample_x2(conv, channel_n)
            elif self.scale == 'x3':
                conv = upsample_x3(conv, channel_n)
            elif self.scale == 'x4':
                conv = upsample_x4(conv, channel_n)
            ##

            out = tf.layers.conv2d(conv, 3, 3, padding='same')

        return out