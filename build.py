import tensorflow as tf
import model
import numpy as np
from ops import imread, rgb2y, psnr_, imgcut, count_param, imagewrite, load_testimg

class Build(object):
    def __init__(self, ckpt_path, modelname, scale, dataset):
        self.savefolder = ckpt_path
        self.modelname = modelname
        self.scale = scale
        self.dataset = dataset
        self.conf = tf.ConfigProto()
        self.input = tf.placeholder(tf.float32, [None, None, None, 3])

        if self.modelname[0:2] == '32':
            if self.modelname[2:3] == 's':
                self.MODEL = model.DRSAN32(self.input, 4, scale)
            elif self.modelname[2:3] == 'm':
                self.MODEL = model.DRSAN32(self.input, 8, scale)
            elif self.modelname[2:3] == 'l':
                self.MODEL = model.DRSAN32(self.input, 10, scale)
            else:
                print('ERROR: Model length is invalid.')
        elif self.modelname[0:2] == '48':
            if self.modelname[2:3] == 's':
                self.MODEL = model.DRSAN48(self.input, 4, scale)
            elif self.modelname[2:3] == 'm':
                self.MODEL = model.DRSAN48(self.input, 8, scale)
            else:
                print('ERROR: Model length is invalid.')
        else:
            print('ERROR: Model name is invalid.')



    def test(self):
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        with tf.Session(config=self.conf) as sess:

            sess.run(self.init)

            print("Searching Checkpoint...")
            ckpt = tf.train.get_checkpoint_state(self.savefolder)
            if ckpt:
                ckpt_list = ckpt.all_model_checkpoint_paths
                self.saver.restore(sess, ckpt_list[0])
                print("Checkpoint Restored")
                count_param()

                ######## DATASET IMAGES COUNT
                timg, labs, dataset_len = load_testimg(self.dataset, self.scale)  # B100 Set5 Set14 Urban100
                avg_psnr = 0.0

                for i in range(dataset_len):
                    test_img = timg[i]
                    tlab = labs[i]
                    output_ = sess.run([self.MODEL.output], feed_dict={self.input: test_img[np.newaxis, :, :, :]})
                    output_ = np.round(255 * np.clip(output_[0], 0.0, 1.0))
                    #imagewrite(output_, i, self.scale, self.dataset)
                    output_ = output_[0, :, :, :] / 255.0
                    if self.scale == 'x2':
                        cutedge = 2
                    elif self.scale == 'x3':
                        cutedge = 3
                    elif self.scale == 'x4':
                        cutedge = 4
                    #h, w, c = output_.size
                    ho,wo,co = output_.shape
                    output_ = imgcut(output_, cutedge)
                    tlab = imgcut(tlab[0:ho,0:wo,:], cutedge)
                    psnr_mean_ = psnr_(rgb2y(output_), rgb2y(tlab))
                    avg_psnr = avg_psnr + psnr_mean_

                print("Average PSNR: %02.2f dB" % (avg_psnr / (dataset_len * 1.0)))
            else:
                print("Checkpoint does not exist")