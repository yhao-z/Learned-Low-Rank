import tensorflow as tf
import os
from model import LR_Net
from dataset_tfrecord import get_dataset_singCoil
import argparse
import numpy as np
import time

from tools import mse, calc_SNR
from tools import mask3d
import matplotlib.pyplot as plt  # plt 用于显示图片
import datetime

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', metavar='int', nargs=1, default=['50'], help='number of epochs')
    parser.add_argument('--batch_size', metavar='int', nargs=1, default=['1'], help='batch size')
    parser.add_argument('--learning_rate', metavar='float', nargs=1, default=['0.001'], help='initial learning rate')
    parser.add_argument('--niter', metavar='int', nargs=1, default=['5'], help='number of network iterations')
    parser.add_argument('--gpu', metavar='int', nargs=1, default=['0'], help='GPU No.')

    args = parser.parse_args()

    # GPU setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu[0]
    GPUs = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(GPUs[0], True)

    mode = 'train'
    batch_size = int(args.batch_size[0])
    num_epoch = int(args.num_epoch[0])
    learning_rate = float(args.learning_rate[0])
    niter = int(args.niter[0])

    logdir = './logs'
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
    model_id = TIMESTAMP + '_ocmr_lr_' + str(learning_rate)
    log_dir_sp = os.path.join(logdir, mode, model_id + '/')
    summary_writer = tf.summary.create_file_writer(log_dir_sp)


    modeldir = os.path.join('./models/', model_id)
    os.makedirs(modeldir)

    # prepare dataset
    dataset = get_dataset_singCoil(mode, batch_size, shuffle=True)
    tf.print('dataset loaded.')

    # initialize network
    net = LR_Net(niter)
    tf.print('network initialized.')

    learning_rate_org = learning_rate
    learning_rate_decay = 0.95

    optimizer = tf.optimizers.Adam(learning_rate_org)

    # Iterate over epochs.
    total_step = 0
    param_num = 0
    loss = 0

    for epoch in range(num_epoch):
        for step, sample in enumerate(dataset):

            # forward
            t0 = time.time()
            with tf.GradientTape() as tape:
                k0, label = sample

                if k0 is None:
                    continue
                if k0.shape[0] < batch_size:
                    continue

                # display the image using matplotlib
                # label_abs = tf.abs(label)
                # plt.figure(1)
                # plt.imshow(label_abs[0, 0, :, :])
                # plt.axis('off')  # 关掉坐标轴为 off
                # plt.title('label')  # 图像题目
                # plt.show()

                # generate under-sampling mask (random)
                nb, nt, nx, ny = k0.get_shape()
                mask = mask3d(nx, ny, nt)
                mask = np.transpose(mask, (2, 0, 1))
                mask = tf.constant(np.complex64(mask + 0j))

                k0 = k0 * mask

                recon = net(k0, mask)
                recon_abs = tf.abs(recon)

                loss = mse(recon, label)

            # backward
            grads = tape.gradient(loss, net.trainable_weights)
            optimizer.apply_gradients(zip(grads, net.trainable_weights))

            # record loss
            with summary_writer.as_default():
                tf.summary.scalar('loss/total', loss.numpy(), step=total_step)
                tf.summary.scalar('SNR', calc_SNR(recon, label), step=total_step)

            # calculate parameter number
            if total_step == 0:
                param_num = np.sum([np.prod(v.get_shape()) for v in net.trainable_variables])

            # log output
            tf.print('Epoch', epoch + 1, '/', num_epoch, 'Step', step, 'loss =', loss.numpy(), loss.numpy(), 'time',
                     time.time() - t0, 'lr = ', learning_rate, 'param_num', param_num)
            total_step += 1

        # learning rate decay for each epoch
        learning_rate = learning_rate_org * learning_rate_decay ** (epoch + 1)  # (total_step / decay_steps)
        optimizer = tf.optimizers.Adam(learning_rate)

        # save model each epoch
        # if epoch in [0, num_epoch-1, num_epoch]:
        model_epoch_dir = os.path.join(modeldir, 'epoch-' + str(epoch + 1), 'ckpt')
        net.save_weights(model_epoch_dir, save_format='tf')
