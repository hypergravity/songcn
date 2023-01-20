import joblib
import numpy as np
from scipy.linalg import toeplitz


class PV2002:

    def __init__(self, S_lambda_x, remainders, Lambda=0.1, osr=10):
        self.S_lambda_x = S_lambda_x
        self.n_row, self.n_col = self.S_lambda_x.shape
        self.f_lambda_sum = self.f_lambda = S_lambda_x.sum(axis=0)
        self.remainders = remainders
        self.Lambda = Lambda
        self.osr = osr
        self.j_max = osr * (self.n_col + 1)
        self.B_j_k = self.eval_B_j_k(j_max=self.j_max)
        self.w_lambda_x_j = self.eval_w(remainders, osr=osr, n_pixel=self.n_col)

    def step(self):
        self.g_j = 0

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
    def eval_A_jk(f_lambda, w_lambda_x_j):
        pass

    @staticmethod
    def eval_R_k(f_lambda, w_lambda_x_j, S_lambda_x):
        pass

    @staticmethod
    def eval_C_lambda(f_lambda, w_lambda_x_j, g_j):
        pass

    @staticmethod
    def eval_D_lambda(w_lambda_x_j, g_j):
        pass

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
        return w_lambda

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


def test_pv2002():
    # load data
    data = joblib.load("/Users/cham/PycharmProjects/songcn/songcn/data/test_pv2002/spec2d.z")
    ap_xx = data['ap_xx']
    ap_yy = data['ap_yy']
    ap_img = data['ap_img']
    ap_remainders = data['ap_remainders']
    pv = PV2002(S_lambda_x=ap_img, remainders=ap_remainders, Lambda=0.1, osr=10)


if __name__ == "__main__":
    test_downsample_spectral_profile()
