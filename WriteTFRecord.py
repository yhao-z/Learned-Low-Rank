import mat73
import scipy.io
import tensorflow as tf
import glob
import os
import numpy as np


def _float_feature(value):
    """Return a float_list form a float/double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Return a int64_list from a bool/enum/int/uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# 创建图像数据的Example
def data_example(data_dir):
    try:
        k = np.array(mat73.loadmat(data_dir)['kData'])
    except:
        k = np.array(scipy.io.loadmat(data_dir)['kData'])
    k_shape = k.shape
    k = k.flatten()

    try:
        csm = np.array(mat73.loadmat(data_dir)['csm'])
    except:
        csm = np.array(scipy.io.loadmat(data_dir)['csm'])
    csm_shape = csm.shape
    csm = csm.flatten()

    feature = {
        'k_real': _float_feature(k.real.tolist()),
        'k_imag': _float_feature(k.imag.tolist()),
        'csm_real': _float_feature(csm.imag.tolist()),
        'csm_imag': _float_feature(csm.imag.tolist()),
        'k_shape': _int64_feature(list(k_shape)),
        'csm_shape': _int64_feature(list(csm_shape))
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


writer = tf.io.TFRecordWriter('I:/OCMR - Cardiovascular MRI/OCMR_test.tfrecord')
data_dirs = glob.glob(os.path.join('I:/OCMR - Cardiovascular MRI/OCMR_data_and_csm - zyh_ESPIRiT/test/', '*.mat'))
for data_dir in data_dirs:
    print(data_dir)

    try:
        csm = np.array(mat73.loadmat(data_dir)['csm'])
    except:
        csm = np.array(scipy.io.loadmat(data_dir)['csm'])  # kx,ky,coils
    csm = np.transpose(csm, (2, 0, 1))  # coils, kx, ky
    RO = csm.shape[1]
    # csm = csm[:, int(np.ceil(RO / 4)):int(np.ceil(RO / 4 * 3)), :]
    csm_shape = csm.shape
    csm = csm.flatten()

    try:
        k = np.array(mat73.loadmat(data_dir)['kData'])
    except:
        k = np.array(scipy.io.loadmat(data_dir)['kData'])
    k = np.squeeze(k)

    if len(k.shape) == 4:
        k = np.transpose(np.squeeze(k), (2, 3, 0, 1))  # kx, ky, coils, kt -> coils,kt,kx,ky
        # k = k[:, :, int(np.ceil(RO / 4)):int(np.ceil(RO / 4 * 3)), :]
        k_shape = k.shape
        k = k.flatten()

        feature = {
            'k_real': _float_feature(k.real.tolist()),
            'k_imag': _float_feature(k.imag.tolist()),
            'csm_real': _float_feature(csm.real.tolist()),
            'csm_imag': _float_feature(csm.imag.tolist()),
            'k_shape': _int64_feature(list(k_shape)),
            'csm_shape': _int64_feature(list(csm_shape))
        }

        exam = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(exam.SerializeToString())

    elif len(k.shape) == 5:
        # kx, ky, coils, kt, slices(avg) -> slices(avg), coils, kt, kx, ky
        k = np.transpose(np.squeeze(k), (4, 2, 3, 0, 1))
        for i in range(k.shape[0]):
            ki = k[i, ]
            # ki = ki[:, :, int(np.ceil(RO / 4)):int(np.ceil(RO / 4 * 3)), :]
            ki_shape = ki.shape
            ki = ki.flatten()
            feature = {
                'k_real': _float_feature(ki.real.tolist()),
                'k_imag': _float_feature(ki.imag.tolist()),
                'csm_real': _float_feature(csm.real.tolist()),
                'csm_imag': _float_feature(csm.imag.tolist()),
                'k_shape': _int64_feature(list(ki_shape)),
                'csm_shape': _int64_feature(list(csm_shape))
            }

            exam = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(exam.SerializeToString())
    else:
        print('the shape of kData is not considered in this code, please adjust the codes')
writer.close()
