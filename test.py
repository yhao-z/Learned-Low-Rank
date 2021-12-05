import tensorflow as tf
import os
from model import LR_Net
from dataset_tfrecord import get_dataset, get_dataset_singCoil
import argparse
import scipy.io as scio
import mat73
import numpy as np
from datetime import datetime
import time
from tools import video_summary

from tools import tempfft, mse, loss_LR, calc_SNR, fft2c_mri, ifft2c_mri
from tools import mask3d
import matplotlib.pyplot as plt  # plt 用于显示图片
import datetime

# tf.debugging.set_log_device_placement(True)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.debugging.set_log_device_placement(True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', metavar='int', nargs=1, default=['50'], help='number of epochs')
    parser.add_argument('--batch_size', metavar='int', nargs=1, default=['1'], help='batch size')
    parser.add_argument('--learning_rate', metavar='float', nargs=1, default=['0.001'], help='initial learning rate')
    parser.add_argument('--niter', metavar='int', nargs=1, default=['20'], help='number of network iterations')
    parser.add_argument('--gpu', metavar='int', nargs=1, default=['0'], help='GPU No.')
    parser.add_argument('--multiCoil', metavar='int', nargs=1, default=['0'])
    parser.add_argument('--weight', metavar='str', nargs=1, default=['models/2021-12-03T14-12-44_ocmr_lr_0.001/epoch-50/ckpt'], help='modeldir in ./models')

    args = parser.parse_args()

    # GPU setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu[0]
    GPUs = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(GPUs[0], True)

    mode = 'test'
    batch_size = int(args.batch_size[0])
    num_epoch = int(args.num_epoch[0])
    learning_rate = float(args.learning_rate[0])
    niter = int(args.niter[0])
    multicoil = int(args.multiCoil[0])
    weight_file = args.weight[0]

    # prepare dataset
    if multicoil == 1:
        dataset = get_dataset(mode, batch_size, shuffle=True)
    else:
        dataset = get_dataset_singCoil(mode, batch_size, shuffle=False)
    tf.print('dataset loaded.')

    # initialize network
    net = LR_Net(niter)
    net.load_weights(weight_file)
    tf.print('network initialized.')

    for step, sample in enumerate(dataset):

        # forward
        t0 = time.time()
        k0 = None
        csm = None
        with tf.GradientTape() as tape:
            if multicoil:
                k0, csm, label = sample
            else:
                k0, label = sample
                csm = None

            if k0 is None:
                continue
            if k0.shape[0] < batch_size:
                continue

            # label = np.array(label)
            # max_label = np.max(np.abs(label[:]))
            # label = tf.constant(label / max_label)
            # k0 = fft2c_mri(label)

            label_abs = tf.abs(label)
            plt.figure(1)
            plt.subplot(1, 3, 1)
            plt.imshow(label_abs[0, 0, :, :])
            plt.axis('off')  # 关掉坐标轴为 off
            plt.title('label')  # 图像题目

            # generate under-sampling mask (random)
            if multicoil:
                nb, nc, nt, nx, ny = k0.get_shape()
                mask = mask3d(nx, ny, nt)
                mask = np.transpose(mask, (2, 0, 1))
                mask = np.tile(mask, (nc, 1, 1, 1))
                mask = tf.constant(np.complex64(mask + 0j))
            else:
                nb, nt, nx, ny = k0.get_shape()
                mask = mask3d(nx, ny, nt)
                scio.savemat('mask%d.mat' % step, {'sampling_mask': mask})
                mask = np.transpose(mask, (2, 0, 1))
                mask = tf.constant(np.complex64(mask + 0j))

            plt.figure(1)
            plt.subplot(1, 3, 3)
            plt.imshow(tf.abs(mask[0, :, :]))
            plt.axis('off')  # 关掉坐标轴为 off
            plt.title('mask')  # 图像题目

            k0 = k0 * mask

            recon = net(k0, mask, csm, label)
            recon_abs = tf.abs(recon)
            plt.figure(1)
            plt.subplot(1, 3, 2)
            plt.imshow(recon_abs[0, 0, :, :])
            plt.axis('off')  # 关掉坐标轴为 off
            plt.title('recon')  # 图像题目

            print(calc_SNR(recon, label))
            # plt.show()
