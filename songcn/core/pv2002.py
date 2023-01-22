import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz


class PV2002:
    """
    P&V (2002) solver.

    Parameters
    ----------
    S_lambda_x : np.ndarray
        (n_lambda, n_width), the spectral image.
    remainders : np.ndarray
        (n_lambda,), the aperture center remainders.
    osr : int
        The over sampling rate.
    Lrel : float
        The relative regularization parameter. The default value is 1.
    zero_wing : int
        The number of edge pixels which are set to zero. The default value is 1.
    clip_sigma : float
        Clip the rows with deviations larger than {clip_sigma} * median deviation.
    xx : np.ndarray
        The X coordinates.
    yy : np.ndarray
        The Y coordinates.

    Attributes
    ----------
    S_lambda_x : np.ndarray
        (n_lambda, n_width), the spectral image.
    f_lambda_sum : np.ndarray
        (n_lambda,), the sum-extracted 1D spectrum.
    f_lambda_0 : np.ndarray
        (n_lambda,), the optimally extracted 1D spectrum in the last iteration.
    f_lambda_1 : np.ndarray
        (n_lambda,), the optimally extracted 1D spectrum in the current iteration.
    g_j_0 : np.ndarray
        ((n_width+1) * osr,), the optimally extracted spatial profile in the last iteration.
    g_j_1 : np.ndarray
        ((n_width+1) * osr,), the optimally extracted spatial profile in the current iteration.

    """

    def __init__(self, S_lambda_x, remainders, osr=10, Lrel=1, zero_wing=1, clip_sigma=3, xx=None, yy=None):
        # important attributes
        self.xx = xx
        self.yy = yy
        self.S_lambda_x = S_lambda_x
        self.n_lambda, self.n_width = self.S_lambda_x.shape
        self.f_lambda_sum = np.einsum("ij->i", S_lambda_x) - remainders * S_lambda_x[:, 0] - (
                1 - remainders) * S_lambda_x[:, -1]
        self.f_lambda_0 = self.f_lambda_1 = None  # 1D spectrum, initialized with None
        self.g_j_0 = self.g_j_1 = None  # spatial profile, initialized with None
        self.remainders = remainders
        self.osr = osr
        self.L = np.median(self.f_lambda_sum) ** 2 * Lrel
        self.zero_wing = zero_wing
        # derived quantities
        self.j_max = osr * (self.n_width + 1)
        self.B_j_k = self.eval_B_j_k(j_max=self.j_max)
        self.w_lambda_x_j = self.eval_w_lambda_x_j(remainders, osr=osr, n_pixel=self.n_width)
        self.A_j_k = None
        self.R_k = None
        self.C_lambda = None
        self.D_lambda = None
        self.S_rec = None
        # pixel mask
        self.lambda_good = np.ones(self.n_lambda, dtype=bool)
        # iteration info
        self.n_iter = 0
        self.f_lambda_history = []
        self.g_j_history = []
        self.dt = 0.
        self.d_g_j = 0.
        self.d_f_lambda = 0.
        self.maxdev_S = 0.
        self.meddev_S = 0.
        self.clip_sigma = clip_sigma

    def reduce(self, n_iter=10, tol_f_lambda=1e-3, tol_g_j=1e-10):
        """
        Reduce spectrum image iteratively until reaching tolerances.

        Parameters
        ----------
        n_iter: int
            The max iteration number.
        tol_f_lambda : float
            The tolerance of 1D spectrum.
        tol_g_j : float
            The tolerance of profile (slit function).
        """
        for i_iter in range(n_iter):
            self.iterate()
            if np.linalg.norm(self.f_lambda_1 - self.f_lambda_0, np.inf) < tol_f_lambda \
                    and np.linalg.norm(self.g_j_1 - self.g_j_0, np.inf) < tol_g_j:
                break

    def iterate(self):
        """ Do one iteration with P&V 2002 algorithm. """
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
        self.R_k = self.eval_R_k(
            self.S_lambda_x[self.lambda_good], self.f_lambda_0[self.lambda_good], self.w_lambda_x_j[self.lambda_good]
        )
        self.g_j_1 = np.linalg.solve(self.A_j_k + self.L * self.B_j_k, self.R_k)
        if self.zero_wing:
            self.g_j_1[:self.osr * self.zero_wing] = 0
            self.g_j_1[-self.osr * self.zero_wing:] = 0
        self.g_j_1[self.g_j_1 < 0] = 0
        self.g_j_1 *= self.osr / self.g_j_1.sum()
        # solve new spectrum with new spatial profile
        self.C_lambda = self.eval_C_lambda(self.S_lambda_x, self.w_lambda_x_j, self.g_j_1)
        self.D_lambda = self.eval_D_lambda(self.w_lambda_x_j, self.g_j_1)
        self.f_lambda_1 = self.C_lambda / self.D_lambda
        # reconstruct image
        self.S_rec = np.einsum("i,ijk,k->ij", self.f_lambda_1, self.w_lambda_x_j, self.g_j_1)
        # record iteration number
        self.n_iter += 1
        # append results in history
        self.f_lambda_history.append(self.f_lambda_1)
        self.g_j_history.append(self.g_j_1)
        # verbose info
        self.dt = datetime.datetime.now() - t_0
        # evaluate variations in spectrum and profile
        self.d_g_j = np.linalg.norm(self.g_j_1 - self.g_j_0, np.inf)
        self.d_f_lambda = np.linalg.norm(self.f_lambda_1 - self.f_lambda_0, np.inf)
        # evaluate deviation from input images
        self.maxdev_S = np.linalg.norm(self.S_lambda_x - self.S_rec, np.inf)
        self.meddev_S = np.median(np.max(np.abs(self.S_lambda_x - self.S_rec), axis=1))

        self.lambda_good &= np.all(np.abs(self.S_lambda_x - self.S_rec) < self.clip_sigma * self.meddev_S, axis=1)

        print(f"Finish {self.n_iter}th iteration: D(t)={self.dt.total_seconds():.2f} sec! \n"
              f"    - D(g_j)      = {self.d_g_j:.2e},\n"
              f"    - D(f_lambda) = {self.d_f_lambda:.2e},\n"
              f"    - MedDev(S)   = {self.meddev_S:.2e},\n"
              f"    - MaxDev(S)   = {self.maxdev_S:.2e},\n"
              f"    - N_clip      = {np.sum(~self.lambda_good)} / {self.n_lambda}")
        return

    @staticmethod
    def eval_B_j_k_(j_max=320):
        """ Evaluate B_j_k, the tri-diagonal matrix with toeplitz method. (Deprecated) """
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
        """
        Evaluate B_j_k, the tri-diagonal matrix.

        Parameters
        ----------
        j_max : int
            The max j value, which is `(n_width + 1 * osr)`.

        Returns
        -------
        np.ndarray
            The B_j_k.
        """
        B_j_k = np.diag(np.ones(j_max) * 2., k=0) + \
                np.diag(np.ones(j_max - 1) * -1., k=1) + \
                np.diag(np.ones(j_max - 1) * -1., k=-1)
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
        # return np.einsum("i,ijkh->kh", f_lambda**2, np.einsum("...i,...j->...ij", w_lambda_x_j, w_lambda_x_j))
        return np.einsum("i,ijk,ijh->kh", np.square(f_lambda), w_lambda_x_j, w_lambda_x_j)

    @staticmethod
    def eval_R_k(S_lambda_x, f_lambda, w_lambda_x_j):
        """
        Evaluate R_k.

        Parameters
        ----------
        S_lambda_x : ndarray
            (n_lambda, n_width)
        f_lambda : ndarray
            (n_lambda,)
        w_lambda_x_j : ndarray
            (n_lambda, n_width, (n_width+1)*osr)

        Returns
        -------
        ndarray
            ((n_width+1)*osr, (n_width+1)*osr)
        """
        return np.einsum("ij,i,ijk->k", S_lambda_x, f_lambda, w_lambda_x_j)

    @staticmethod
    def eval_C_lambda(S_lambda_x, w_lambda_x_j, g_j):
        """
        Evaluate C_lambda.

        Parameters
        ----------
        S_lambda_x : ndarray
            (n_lambda, n_width)
        w_lambda_x_j : ndarray
            (n_lambda, n_width, (n_width+1)*osr)
        g_j : ndarray
            ((n_width+1)*osr,)

        Returns
        -------
        ndarray
            ((n_width+1)*osr, (n_width+1)*osr)
        """
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
        """ Evaluate w_lambda_x_j at a given lambda. """
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
    def eval_w_lambda_x_j(remainders, osr=10, n_pixel=31):
        """
        Evaluate w_lambda_x_j from aperture center remainders, over-sampling rate, and aperture width.

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

    def plot_result(self, profile_shift=0.002):
        assert self.xx is not None and self.yy is not None

        fig = plt.figure(figsize=(10, 12))

        ax0 = fig.add_subplot(3, 2, 1, title="Input image")
        I = ax0.imshow(self.S_lambda_x, aspect="auto", interpolation="nearest", cmap=plt.cm.jet,
                       vmin=np.min(self.S_rec),
                       vmax=np.max(self.S_rec))
        plt.colorbar(I, ax=ax0)
        ax0.set_xlabel("Rectified $X$ [pixel]")
        ax0.set_ylabel("$Y$ [pixel]")

        ax1 = fig.add_subplot(3, 2, 2, title="Reconstructed image")
        I = ax1.imshow(
            self.S_rec, aspect="auto", interpolation="nearest", cmap=plt.cm.jet, vmin=np.min(self.S_rec),
            vmax=np.max(self.S_rec)
        )
        plt.colorbar(I, ax=ax1)
        ax1.set_xlabel("Rectified $X$ [pixel]")
        ax1.set_ylabel("$Y$ [pixel]")

        ax0 = fig.add_subplot(3, 2, 3, projection='3d', title="Input image")
        ax1 = fig.add_subplot(3, 2, 4, projection='3d', title="Reconstructed image")
        ax0.plot_surface(
            self.xx, self.yy, self.S_lambda_x, cmap=plt.cm.jet, vmin=np.min(self.S_rec), vmax=np.max(self.S_rec)
        )
        ax1.plot_surface(
            self.xx, self.yy, self.S_rec, cmap=plt.cm.jet, vmin=np.min(self.S_rec), vmax=np.max(self.S_rec),
        )
        ax0.set_ylabel("$X$ (spatial direction) [pixel]")
        ax0.set_xlabel("$Y$ (dispersion direction) [pixel]")
        ax1.set_ylabel("$X$ (spatial direction) [pixel]")
        ax1.set_xlabel("$Y$ (dispersion direction) [pixel]")
        ax0.set_zlim(ax1.get_zlim())
        ax0.set_zlabel("Counts [ADU]")
        ax1.set_zlabel("Counts [ADU]")
        ax0.view_init(30, -10, 0)
        ax1.view_init(30, -10, 0)

        # plot profile
        ax_g = fig.add_subplot(3, 2, 5, title="Spatial profile")
        # data
        ax_g.plot(
            np.arange(self.n_width).reshape(-1, 1) + 1 - self.remainders,
            self.S_lambda_x.T / self.f_lambda_sum,
            color="gray", zorder=-1
        )
        # estimated profile
        for i, g in enumerate(self.g_j_history):
            ax_g.plot(np.arange(len(g)) / self.osr - .5 + .5 / self.osr,  # this is tricky
                      g[::-1] + (self.n_iter - 1 - i) * profile_shift)
            print("sum(g): ", g.sum())
        ax_g.set_xlabel("Corrected $X$ [pixel]")
        ax_g.set_ylabel("Profile density $\\times$ Over sampling rate")

        # plot spectrum
        ax_f = fig.add_subplot(3, 2, 6, title="Extracted spectrum")
        for i, f in enumerate(self.f_lambda_history):
            if i == 0:
                plt.plot(f, label=f"Iter {i}", color="k")
            else:
                plt.plot(f, label=f"Iter {i}")
            print("median(f): ", np.median(f))
        ax_f.legend()
        ax_f.set_xlabel("$\\lambda$ [pixel]")
        ax_f.set_ylabel("Counts [ADU]")

        fig.tight_layout()
        plt.show()
        # fig.subplots_adjust(top=0.95)
        return fig


def test_downsample_spectral_profile():
    # %timeit eval_w_lambda(remainder=0.247, osr=10, n_pixel=31)
    # %timeit eval_w(remainders=ap.ap_center_remainder[:256], osr=10, n_pixel=31).shape
    n_pixel = 31
    osr = 10
    test_g = np.ones((n_pixel + 1) * osr).reshape(-1, 1)
    test_w = PV2002.eval_w_lambda_x_j(remainders=np.random.uniform(0, 1, size=256), osr=10, n_pixel=31)
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
    # ap_img[10, 5:8] = 50000
    pv = PV2002(S_lambda_x=ap_img, remainders=ap_remainders, osr=5, Lrel=10, xx=ap_xx, yy=ap_yy)

    pv.reduce(n_iter=10, tol_f_lambda=1e-3, tol_g_j=1e-10)
    fig = pv.plot_result()
    return fig


if __name__ == "__main__":
    test_downsample_spectral_profile()
    fig = test_pv2002()
