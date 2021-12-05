import tempfile
import os
import tensorflow as tf
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.io as scio


def video_summary(name, video, step=None, fps=10):
    name = tf.constant(name).numpy().decode('utf-8')
    video = np.array(video)
    if video.dtype in (np.float32, np.float64):
        video = np.clip(255 * video, 0, 255).astype(np.uint8)
    B, T, H, W, C = video.shape
    try:
        frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
        summary = tf.compat.v1.Summary()
        image = tf.compat.v1.Summary.Image(
            height=B * H, width=T * W, colorspace=C)
        image.encoded_image_string = encode_gif(frames, fps)
        summary.value.add(tag=name + '/gif', image=image)
        tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
    except (IOError, OSError) as e:
        print('GIF summaries require ffmpeg in $PATH.', e)
        frames = video.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
        tf.summary.image(name + '/grid', frames, step)


def encode_gif(frames, fps):
    from subprocess import Popen, PIPE
    h, w, c = frames[0].shape
    pxfmt = {1: 'gray', 3: 'rgb24'}[c]
    cmd = ' '.join([
        f'ffmpeg -y -f rawvideo -vcodec rawvideo',
        f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
        f'[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
        f'-r {fps:.02f} -f gif -'])
    proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in frames:
        proc.stdin.write(image.tostring())
    out, err = proc.communicate()
    if proc.returncode:
        raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
    del proc
    return out


def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2) ** 2)


def cartesian_mask(shape, acc, sample_n=10, centred=False):
    """
    Sampling density estimated from implementation of kt FOCUSS
    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..
    """
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = normal_pdf(Nx, 0.5 / (Nx / 10.) ** 2)
    lmda = Nx / (2. * acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1. / Nx

    if sample_n:
        pdf_x[Nx // 2 - sample_n // 2:Nx // 2 + sample_n // 2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx // 2 - sample_n // 2:Nx // 2 + sample_n // 2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    if not centred:
        mask = mymath.ifftshift(mask, axes=(-1, -2))

    return mask


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


def loss_LR(y, y_):
    pred = tf.stack([tf.math.real(y), tf.math.imag(y)], axis=-1)
    label = tf.stack([tf.math.real(y_), tf.math.imag(y_)], axis=-1)

    loss = tf.reduce_mean(tf.math.square(pred - label))

    return loss


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


class Emat_xyt():
    def __init__(self):
        super(Emat_xyt, self).__init__()

    def mtimes(self, b, inv, mask, csm=None):
        if csm == None:
            if inv:
                x = self._ifft2c_mri_singlecoil(b * mask)
            else:
                x = self._fft2c_mri_singlecoil(b) * mask
        else:
            csm = tf.expand_dims(csm, 2)
            if inv:
                x = self._ifft2c_mri_multicoil(b * mask)
                x = x * tf.math.conj(csm)
                x = tf.reduce_sum(x, 1)  # / tf.cast(tf.reduce_sum(tf.abs(csm)**2, 1), dtype=tf.complex64)
            else:
                b = tf.expand_dims(b, 1) * csm
                x = self._fft2c_mri_multicoil(b) * mask

        return x

    def _fft2c_mri_multicoil(self, x):
        # nb nt nx ny -> nb, nc, nt, nx, ny
        X = tf.signal.fftshift(x, 3)
        X = tf.transpose(X, perm=[0, 1, 2, 4, 3])  # permute to make nx dimension the last one.
        X = tf.signal.fft(X)
        X = tf.transpose(X, perm=[0, 1, 2, 4, 3])  # permute back to original order.
        nb, nc, nt, nx, ny = np.float32(x.shape)
        nx = tf.constant(np.complex64(nx + 0j))
        ny = tf.constant(np.complex64(ny + 0j))
        X = tf.signal.fftshift(X, 3) / tf.sqrt(nx)
        X = tf.signal.fftshift(X, 4)
        X = tf.signal.fft(X)
        X = tf.signal.fftshift(X, 4) / tf.sqrt(ny)

        return X

    def _ifft2c_mri_multicoil(self, X):
        # nb nt nx ny -> nb, nc, nt, nx, ny
        x = tf.signal.fftshift(X, 3)
        x = tf.transpose(x, perm=[0, 1, 2, 4, 3])  # permute a to make nx dimension the last one.
        x = tf.signal.ifft(x)
        x = tf.transpose(x, perm=[0, 1, 2, 4, 3])  # permute back to original order.
        nb, nc, nt, nx, ny = np.float32(X.shape)
        nx = tf.constant(np.complex64(nx + 0j))
        ny = tf.constant(np.complex64(ny + 0j))

        x = tf.signal.fftshift(x, 3) * tf.sqrt(nx)

        x = tf.signal.fftshift(x, 4)
        x = tf.signal.ifft(x)
        x = tf.signal.fftshift(x, 4) * tf.sqrt(ny)

        return x

    def _fft2c_mri_singlecoil(self, x):
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

    def _ifft2c_mri_singlecoil(self, X):
        # nb nt nx ny
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


