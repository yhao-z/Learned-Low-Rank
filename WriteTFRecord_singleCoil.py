import mat73
import scipy.io
import tensorflow as tf
import glob
import os
import numpy as np


def fft2c_mri(x):
    # nb nx ny nt
    X = np.fft.ifftshift(x, 1)
    X = np.transpose(X, (0, 2, 1))  # permute to make nx dimension the last one.
    X = np.fft.fft(X)
    X = np.transpose(X, (0, 2, 1))  # permute back to original order.
    nt, nx, ny = np.float32(x.shape)
    nx = np.complex64(nx + 0j)
    ny = np.complex64(ny + 0j)
    X = np.fft.fftshift(X, 1) / np.sqrt(nx)

    X = np.fft.ifftshift(X, 2)
    X = np.fft.fft(X)
    X = np.fft.fftshift(X, 2) / np.sqrt(ny)

    return X


def _float_feature(value):
    """Return a float_list form a float/double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Return a int64_list from a bool/enum/int/uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# 创建图像数据的Example
def data_example(data_dir):
    try:
        label = np.array(mat73.loadmat(data_dir)['label'])
    except:
        label = np.array(scipy.io.loadmat(data_dir)['label'])
    label = np.transpose(label, (2, 0, 1))  # nx, ny, nt -> nt, nx, ny

    label = np.array(label)
    max_label = np.max(np.abs(label[:]))
    label = tf.constant(label / max_label)
    k = fft2c_mri(label)

    label = np.array(label)
    k = np.array(k)

    label_shape = label.shape
    label = label.flatten()

    k_shape = k.shape
    k = k.flatten()

    feature = {
        'k_real': _float_feature(k.real.tolist()),
        'k_imag': _float_feature(k.imag.tolist()),
        'label_real': _float_feature(label.real.tolist()),
        'label_imag': _float_feature(label.imag.tolist()),
        'k_shape': _int64_feature(list(k_shape)),
        'label_shape': _int64_feature(list(label_shape))
    }

    exam = tf.train.Example(features=tf.train.Features(feature=feature))

    return exam


writer = tf.io.TFRecordWriter('I:/OCMR - Cardiovascular MRI/OCMR_getlabel_SingleCoil/OCMR_singCoil_test.tfrecord')
data_dirs = glob.glob(os.path.join('I:/OCMR - Cardiovascular MRI/OCMR_getlabel_SingleCoil/test/', '*.mat'))
for data_dir in data_dirs:
    print(data_dir)
    exam = data_example(data_dir)
    writer.write(exam.SerializeToString())
writer.close()



