import tensorflow as tf
import numpy as np


def mask3d(nx, ny, nz, center_r=[15, 15, 0], undersampling=0.5):
    # create undersampling mask
    mask_shape = np.array([nx, ny, nz])
    Npts = mask_shape.prod()  # total number of data points
    k = int(round(Npts * undersampling))  # undersampling
    ri = np.random.choice(Npts, k, replace=False)  # index for undersampling
    ma = np.zeros(Npts)  # initialize an all zero vector
    ma[ri] = 1  # set sampled data points to 1
    mask = ma.reshape(mask_shape)

    flag_centerfull = 1
    # x center, k-space index range
    if center_r[0] > 0:
        cxr = np.arange(-center_r[0], center_r[0] + 1) + mask_shape[0] // 2
    elif center_r[0] is 0:
        cxr = np.arange(mask_shape[0])
    else:
        flag_centerfull = 0
    # y center, k-space index range
    if center_r[1] > 0:
        cyr = np.arange(-center_r[1], center_r[1] + 1) + mask_shape[1] // 2
    elif center_r[1] is 0:
        cyr = np.arange(mask_shape[1])
    else:
        flag_centerfull = 0
    # z center, k-space index range
    if center_r[2] > 0:
        czr = np.arange(-center_r[2], center_r[2] + 1) + mask_shape[2] // 2
    elif center_r[2] is 0:
        czr = np.arange(mask_shape[2])
    else:
        flag_centerfull = 0

    # full sampling in the center kspace
    if flag_centerfull is not 0:
        mask[np.ix_(cxr, cyr, czr)] = \
            np.ones((cxr.shape[0], cyr.shape[0], czr.shape[0]))  # center k-space is fully sampled
    return mask


def calc_SNR(y, y_):
    y = np.array(y).flatten()
    y_ = np.array(y_).flatten()
    err = np.linalg.norm(y_ - y) ** 2
    snr = 10 * np.log10(np.linalg.norm(y_) ** 2 / err)

    return snr


def tempfft(input, inv):
    if len(input.shape) == 4:
        nb, nt, nx, ny = np.float32(input.shape)
        nt = tf.constant(np.complex64(nt + 0j))

        if inv:
            x = tf.transpose(input, perm=[0, 2, 3, 1])
            # x = tf.signal.fftshift(x, 3)
            x = tf.signal.ifft(x)
            x = tf.transpose(x, perm=[0, 3, 1, 2])
            x = x * tf.sqrt(nt)
        else:
            x = tf.transpose(input, perm=[0, 2, 3, 1])
            x = tf.signal.fft(x)
            # x = tf.signal.fftshift(x, 3)
            x = tf.transpose(x, perm=[0, 3, 1, 2])
            x = x / tf.sqrt(nt)
    else:
        nb, nt, nx, ny, _ = np.float32(input.shape)
        nt = tf.constant(np.complex64(nt + 0j))

        if inv:
            x = tf.transpose(input, perm=[0, 2, 3, 4, 1])
            # x = tf.signal.fftshift(x, 4)
            x = tf.signal.ifft(x)
            x = tf.transpose(x, perm=[0, 4, 1, 2, 3])
            x = x * tf.sqrt(nt)
        else:
            x = tf.transpose(input, perm=[0, 2, 3, 4, 1])
            x = tf.signal.fft(x)
            # x = tf.signal.fftshift(x, 4)
            x = tf.transpose(x, perm=[0, 4, 1, 2, 3])
            x = x / tf.sqrt(nt)
    return x


def mse(recon, label):
    if recon.dtype == tf.complex64:
        residual_cplx = recon - label
        residual = tf.stack([tf.math.real(residual_cplx), tf.math.imag(residual_cplx)], axis=-1)
        mse = tf.reduce_mean(residual ** 2)
    else:
        residual = recon - label
        mse = tf.reduce_mean(residual ** 2)
    return mse


def fft2c_mri(x):
    # nb nx ny nt
    X = tf.signal.ifftshift(x, 2)
    X = tf.transpose(X, perm=[0, 1, 3, 2])  # permute to make nx dimension the last one.
    X = tf.signal.fft(X)
    X = tf.transpose(X, perm=[0, 1, 3, 2])  # permute back to original order.
    nb, nt, nx, ny = np.float32(x.shape)
    nx = tf.constant(np.complex64(nx + 0j))
    ny = tf.constant(np.complex64(ny + 0j))
    X = tf.signal.fftshift(X, 2) / tf.sqrt(nx)

    X = tf.signal.ifftshift(X, 3)
    X = tf.signal.fft(X)
    X = tf.signal.fftshift(X, 3) / tf.sqrt(ny)

    return X


def ifft2c_mri(X):
    # nb nx ny nt
    x = tf.signal.ifftshift(X, 2)
    x = tf.transpose(x, perm=[0, 1, 3, 2])  # permute a to make nx dimension the last one.
    x = tf.signal.ifft(x)
    x = tf.transpose(x, perm=[0, 1, 3, 2])  # permute back to original order.
    nb, nt, nx, ny = np.float32(X.shape)
    nx = tf.constant(np.complex64(nx + 0j))
    ny = tf.constant(np.complex64(ny + 0j))
    x = tf.signal.fftshift(x, 2) * tf.sqrt(nx)

    x = tf.signal.ifftshift(x, 3)
    x = tf.signal.ifft(x)
    x = tf.signal.fftshift(x, 3) * tf.sqrt(ny)

    return x


def sos(x):
    # x: nb, ncoil, nt, nx, ny; complex64
    x = tf.math.reduce_sum(tf.abs(x ** 2), axis=1)
    x = x ** (1.0 / 2)
    return x


def softthres(x, thres):
    x_abs = tf.abs(x)
    coef = tf.nn.relu(x_abs - thres) / (x_abs + 1e-10)
    coef = tf.cast(coef, tf.complex64)
    return coef * x
