import tensorflow as tf
from tensorflow.keras import layers
from tools import fft2c_mri, ifft2c_mri
global label

class LR_Net(tf.keras.Model):
    def __init__(self, niter):
        super(LR_Net, self).__init__(name='LR_Net')
        self.niter = niter
        self.celllist = []

    def build(self, input_shape):
        for i in range(self.niter-1):
            self.celllist.append(LRCell(input_shape, i))
        self.celllist.append(LRCell(input_shape, self.niter-1, is_last=True))

    def call(self, d, mask, csm, label):
        """
        d: undersampled k-space
        csm: coil sensitivity map
        mask: sampling mask
        label: to test the SNRs, when test passed, it can be delete
        """
        # nb, nc, nt, nx, ny = d.shape
        x_rec = ifft2c_mri(d)
        A = tf.zeros_like(x_rec)
        L = tf.zeros_like(x_rec)
        data = [x_rec, L, A, d, csm, mask, label]

        for i in range(self.niter):
            data = self.celllist[i](data)

        x_rec = data[0]

        return x_rec


class LRCell(layers.Layer):
    def __init__(self, input_shape, i, is_last=False):
        super(LRCell, self).__init__()
        if len(input_shape) == 4:
            self.nb, self.nt, self.nx, self.ny = input_shape
        else:
            self.nb, nc, self.nt, self.nx, self.ny = input_shape

        if is_last:
            self.thres_coef = tf.Variable(tf.constant(1, dtype=tf.float32), trainable=True, name='thres_coef %d' % i)
            self.mu = tf.Variable(tf.constant(0.1, dtype=tf.float32), trainable=True, name='mu %d' % i)
            self.eta = tf.Variable(tf.constant(1, dtype=tf.float32), trainable=False, name='eta %d' % i)
        else:
            self.thres_coef = tf.Variable(tf.constant(1, dtype=tf.float32), trainable=True, name='thres_coef %d' % i)
            self.mu = tf.Variable(tf.constant(0.1, dtype=tf.float32), trainable=True, name='mu %d' % i)
            self.eta = tf.Variable(tf.constant(1, dtype=tf.float32), trainable=True, name='eta %d' % i)

    def call(self, data, **kwargs):
        x_rec, L, A, d, csm, mask, label = data

        x_temp = x_rec + L
        A = self.lowrank_step(x_temp)
        x_rec = self.x_step(L, A, d, csm, mask)
        L = self.L_step(L, x_rec, A)

        data[0] = x_rec
        data[1] = L
        data[2] = A

        return data

    def x_step(self, L, A, d, csm, mask):
        temp = A - L
        k_rec = fft2c_mri(temp)  # tf.cast(tf.nn.relu(self.mu), tf.complex64)
        k_rec = tf.math.scalar_mul(tf.cast(tf.nn.relu(self.mu), tf.complex64), d) + k_rec
        k_rec = tf.math.divide_no_nan(k_rec, tf.math.scalar_mul(tf.cast(tf.nn.relu(self.mu), tf.complex64), mask) + 1)
        x_rec = ifft2c_mri(k_rec)
        return x_rec

    def lowrank_step(self, x):
        [batch, Nt, Nx, Ny] = x.get_shape()
        M = tf.reshape(x, [batch, Nt, Nx * Ny])
        St, Ut, Vt = tf.linalg.svd(M)
        thres = self.thres_coef
        # thres = tf.sigmoid(self.thres_coef) * St[:, 0]
        # thres = tf.expand_dims(thres, -1)
        St = tf.nn.relu(St - thres)
        St = tf.linalg.diag(St)

        St = tf.dtypes.cast(St, tf.complex64)
        Vt_conj = tf.transpose(Vt, perm=[0, 2, 1])
        Vt_conj = tf.math.conj(Vt_conj)
        US = tf.linalg.matmul(Ut, St)
        M = tf.linalg.matmul(US, Vt_conj)
        A = tf.reshape(M, [batch, Nt, Nx, Ny])

        return A

    def L_step(self, L, x_rec, A):
        eta = tf.cast(tf.nn.relu(self.eta), tf.complex64)
        return L + tf.math.scalar_mul(eta, x_rec - A)
