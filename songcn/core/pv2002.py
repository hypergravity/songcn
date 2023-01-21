import datetime
import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz


class PV2002:
    """
    P&V (2002) solver.
    """

    def __init__(self, S_lambda_x, remainders, Lrel=1, osr=10, zero_wing=1):
        self.S_lambda_x = S_lambda_x
        self.n_lambda, self.n_width = self.S_lambda_x.shape
        self.f_lambda_sum = np.einsum("ij->i", S_lambda_x)
        self.f_lambda_0 = self.f_lambda_1 = None  # 1D spectrum, initialized with None
        self.g_j_0 = self.g_j_1 = None  # spatial profile, initialized with None
        self.remainders = remainders
        self.osr = osr
        self.L = np.median(self.f_lambda_sum) ** 2 * Lrel
        self.zero_wing = zero_wing
        self.j_max = osr * (self.n_width + 1)
        self.B_j_k = self.eval_B_j_k(j_max=self.j_max)
        self.w_lambda_x_j = self.eval_w(remainders, osr=osr, n_pixel=self.n_width)
        self.A_j_k = None
        self.R_k = None
        self.C_lambda = None
        self.D_lambda = None
        self.S_rec = None
        self.n_iter = 0
        self.f_lambda_history = []
        self.g_j_history = []
        self.lambda_good = np.ones(self.n_lambda, dtype=bool)

        # iteration info
        self.dt = 0.
        self.d_g_j = 0.
        self.d_f_lambda = 0.
        self.maxdev_S = 0.
        self.meddev_S = 0.

    def iterate(self):
        logging.info("Start to iterate ...")
        t_0 = datetime.datetime.now()

        # prepare to iterate
        self.f_lambda_0 = self.f_lambda_1
        self.g_j_0 = self.g_j_1

        if self.f_lambda_0 is None:
            self.f_lambda_0 = self.f_lambda_sum
            self.f_lambda_history.append(self.f_lambda_0)
            self.g_j_0 = np.ones((self.n_width + 1) * self.osr, dtype=float) / (self.n_width + 1)

        # solve new spatial profile
        self.A_j_k = self.eval_A_j_k(self.f_lambda_0[self.lambda_good], self.w_lambda_x_j[self.lambda_good])
        self.R_k = self.eval_R_k(self.f_lambda_0[self.lambda_good], self.w_lambda_x_j[self.lambda_good],
                                 self.S_lambda_x[self.lambda_good])
        self.g_j_1 = np.linalg.solve(self.A_j_k + self.L * self.B_j_k, self.R_k)
        if self.zero_wing:
            self.g_j_1[:self.osr * self.zero_wing] = 0
            self.g_j_1[-self.osr * self.zero_wing:] = 0
        self.g_j_1 *= self.osr / self.g_j_1.sum()
        # solve new spectrum with new spatial profile
        self.C_lambda = self.eval_C_lambda(self.S_lambda_x, self.w_lambda_x_j, self.g_j_1)
        self.D_lambda = self.eval_D_lambda(self.w_lambda_x_j, self.g_j_1)
        self.f_lambda_1 = self.C_lambda / self.D_lambda
        # reconstruct image
        self.S_rec = np.einsum("i,ijk,k->ij", self.f_lambda_1, self.w_lambda_x_j, self.g_j_1)
        # record iteration number
        self.n_iter += 1
        self.f_lambda_history.append(self.f_lambda_1)
        self.g_j_history.append(self.g_j_1)

        self.dt = datetime.datetime.now() - t_0
        self.d_g_j = np.linalg.norm(self.g_j_1 - self.g_j_0, np.inf)
        self.d_f_lambda = np.linalg.norm(self.f_lambda_1 - self.f_lambda_0, np.inf)
        self.maxdev_S = np.linalg.norm(self.S_lambda_x - self.S_rec, np.inf)
        self.meddev_S = np.median(np.max(np.abs(self.S_lambda_x - self.S_rec), axis=1))

        self.lambda_good &= np.all(np.abs(self.S_lambda_x - self.S_rec) < 5 * self.meddev_S, axis=1)

        print(f"Finish {self.n_iter}th iteration: D(t)={self.dt.total_seconds():.2f} sec!"
              f" D(g_j) = {self.d_g_j:.2e},"
              f" D(f_lambda) = {self.d_f_lambda:.2e},"
              f" MedDev(S)={self.meddev_S:.2e},"
              f" MaxDev(S)={self.maxdev_S:.2e},"
              f" N_clip={np.sum(~self.lambda_good)}")
        return

    @staticmethod
    def eval_B_j_k_(j_max=320):
        c = np.zeros(j_max, dtype=float)
        r = np.zeros(j_max, dtype=float)
        c[0] = 2
        c[1] = -1
        r[1] = -1
        B_j_k = toeplitz(c, r)
        B_j_k[0, 0] = 1
        B_j_k[-1, -1] = 1
        return B_j_k

    @staticmethod
    def eval_B_j_k(j_max=320):
        B_j_k = np.diag(np.ones(j_max) * 2., k=0) \
                + np.diag(np.ones(j_max - 1) * -1., k=1) \
                + np.diag(np.ones(j_max - 1) * -1., k=-1)
        B_j_k[0, 0] = 1
        B_j_k[-1, -1] = 1
        return B_j_k

    @staticmethod
    def eval_A_j_k(f_lambda, w_lambda_x_j):
        """
        Evaluate A_j_k.

        Parameters
        ----------
        f_lambda : ndarray
            (n_lambda,)
        w_lambda_x_j : ndarray
            (n_lambda, n_width, (n_width+1)*osr)

        Returns
        -------
        ndarray
            ((n_width+1)*osr, (n_width+1)*osr)
        """
        return np.einsum("i,ijk,ijh->kh", np.square(f_lambda), w_lambda_x_j, w_lambda_x_j)
        # return np.einsum("i,ijkh->kh", f_lambda**2, np.einsum("...i,...j->...ij", w_lambda_x_j, w_lambda_x_j))

    @staticmethod
    def eval_R_k(f_lambda, w_lambda_x_j, S_lambda_x):
        """
        Evaluate R_k.

        Parameters
        ----------
        f_lambda : ndarray
            (n_lambda,)
        w_lambda_x_j : ndarray
            (n_lambda, n_width, (n_width+1)*osr)
        S_lambda_x : ndarray
            (n_lambda, n_width)

        Returns
        -------
        ndarray
            ((n_width+1)*osr, (n_width+1)*osr)
        """
        return np.einsum("ij,i,ijk->k", S_lambda_x, f_lambda, w_lambda_x_j)

    @staticmethod
    def eval_C_lambda(S_lambda_x, w_lambda_x_j, g_j):
        return np.einsum("ij,ijk,k->i", S_lambda_x, w_lambda_x_j, g_j)

    @staticmethod
    def eval_D_lambda(w_lambda_x_j, g_j):
        """
        Evaluate D_lambda.

        Parameters
        ----------
        w_lambda_x_j : ndarray
            (n_lambda, n_width, (n_width+1)*osr)
        g_j : ndarray
            ((n_width+1)*osr,)

        Returns
        -------
        ndarray
            (n_lambda,)
        """
        return np.einsum("ij->i", np.square(np.einsum("ijk,k->ij", w_lambda_x_j, g_j)))

    @staticmethod
    def eval_w_lambda(remainder=0.0247, osr=10, n_pixel=31):
        # evaluate start index of period
        start_ind = int(np.floor(remainder * osr))
        # construct period
        period = np.ones((osr + 1,), dtype=float) / osr
        period[-1] = np.mod(remainder, 1. / osr)
        period[0] = 1. / osr - period[-1]
        # construct w_lambda
        w_lambda = np.zeros((n_pixel, (n_pixel + 1) * osr), dtype=float)
        for i_pixel in range(n_pixel):
            w_lambda[i_pixel, start_ind + i_pixel * osr:start_ind + (i_pixel + 1) * osr + 1] = period
        return w_lambda[::-1]

    @staticmethod
    def eval_w(remainders, osr=10, n_pixel=31):
        """
        Evaluate W_lambda_x from aperture center remainders, over-sampling rate, and aperture width.

        Parameters
        ----------
        remainders : array-like
            The aperture center remainder.
        osr : int, optional
            The over-sampling rate. The default is 10.
        n_pixel : int, optional
            The aperture width (pixels). The default is 31.

        Returns
        -------
        np.ndarray
            W_lambda_x.

        """
        return np.array(
            [
                PV2002.eval_w_lambda(remainder=remainder, osr=osr, n_pixel=n_pixel) for remainder in remainders
            ]
        )


def test_downsample_spectral_profile():
    # %timeit eval_w_lambda(remainder=0.247, osr=10, n_pixel=31)
    # %timeit eval_w(remainders=ap.ap_center_remainder[:256], osr=10, n_pixel=31).shape
    n_pixel = 31
    osr = 10
    test_g = np.ones((n_pixel + 1) * osr).reshape(-1, 1)
    test_w = PV2002.eval_w(remainders=np.random.uniform(0, 1, size=256), osr=10, n_pixel=31)
    print(np.linalg.norm(test_w @ test_g - 1))
    assert np.linalg.norm(test_w @ test_g - 1) < 1e-10
    # (256, 31, 320) @ (320,) = (256, 31)
    return


def generate_gaussian_profile(npix=10):
    # 5 sigma
    xx = np.linspace(-5, 5, npix)
    yy = np.exp(-xx ** 2)
    return yy / yy.sum()


def test_pv2002():
    # load data
    data = joblib.load("/Users/cham/PycharmProjects/songcn/songcn/data/test_pv2002/spec2d.z")
    ap_xx = data['ap_xx']
    ap_yy = data['ap_yy']
    ap_img = data['ap_img']
    ap_remainders = data['ap_remainders']
    ap_img[10, 5:11] = 100000
    pv = PV2002(S_lambda_x=ap_img, remainders=ap_remainders, osr=5, Lrel=1)

    pv.iterate()
    pv.iterate()
    pv.iterate()
    pv.iterate()
    pv.iterate()

    plt.figure()
    for i, g in enumerate(pv.g_j_history):
        plt.plot(g + i * 0.001)
        print(g.sum())

    plt.figure()
    for f in pv.f_lambda_history:
        plt.plot(f)
        print(np.median(f))

    plt.figure()
    plt.imshow(pv.S_lambda_x - pv.S_rec, aspect="auto")
    plt.colorbar()

    plt.figure()
    plt.plot((pv.S_lambda_x / pv.f_lambda_sum[:, None]).T)

    S_rec = np.einsum("ijk,k->ij", pv.w_lambda_x_j, pv.g_j_1)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(pv.S_lambda_x)
    axs[1].imshow(S_rec)


if __name__ == "__main__":
    test_downsample_spectral_profile()
