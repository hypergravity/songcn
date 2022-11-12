import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


def cost1d(p, x, y, pw=2):
    res = np.polynomial.chebyshev.chebval(x, p) - y
    return np.abs(res) ** (pw / 2)


def cost2d(p, x, y, z, deg=(1, 1), pw=2):
    res = np.polynomial.chebyshev.chebval2d(x, y, p.reshape(deg[0] + 1, deg[1] + 1)) - z
    return np.abs(res) ** (pw / 2)


class PolyFit1D:
    """
    1D polynomial fitter.

    Parameters
    ----------
    x : np.ndarray
        X data.
    y : np.ndarray
        Y data.
    deg : int
        The degree of polynomial in X dimension.
    pw : int
        The cost type is ``(abs(model-obs))**pw``.
    robust : bool
        If True, use median instead of mean to estimate initial parameters.
    """
    x_mean = 0
    x_scale = 0
    y_mean = 0
    y_scale = 0

    x_sc = 0
    y_sc = 0

    opt = None
    p = None
    deg = 0
    rms = 0

    def __init__(self, x, y, deg=1, pw=1, robust=True):
        # determine data scale
        if robust:
            self.x_scale = np.ptp(np.percentile(x, q=[84, 16])) / 2
            self.x_mean = np.percentile(x, q=50)
            self.y_scale = np.ptp(np.percentile(y, q=[84, 16])) / 2
            self.y_mean = np.percentile(y, q=50)
        else:
            self.x_scale = np.std(x)
            self.x_mean = np.mean(x)
            self.y_scale = np.std(y)
            self.y_mean = np.mean(y)

        # normalize data
        self.x_sc = (x - self.x_mean) / self.x_scale
        self.y_sc = (y - self.y_mean) / self.y_scale

        # fit data
        p0 = np.zeros((deg + 1,), dtype=float)
        self.opt = least_squares(cost1d, p0, method="lm", args=(self.x_sc, self.y_sc, pw))
        self.p = self.opt.x

        self.pw = pw
        self.robust = robust
        self.deg = deg

    def __call__(self, x):
        x_sc = (x - self.x_mean) / self.x_scale
        y_sc = np.polynomial.chebyshev.chebval(x_sc, self.p)
        return y_sc * self.y_scale + self.y_mean


class PolyFit2D:
    """
    2D polynomial fitter.

    Parameters
    ----------
    x : np.ndarray
        X data.
    y : np.ndarray
        Y data.
    z : np.ndarray
        Z data.
    deg : tuple
        The degrees of polynomials in X and Y dimensions.
    pw : int
        The cost type is ``(abs(model-obs))**pw``.
    robust : bool
        If True, use median instead of mean to estimate initial parameters.
    """
    x_mean = 0
    x_scale = 0
    y_mean = 0
    y_scale = 0
    z_mean = 0
    z_scale = 0

    x_sc = 0
    y_sc = 0
    z_sc = 0

    opt = None
    p = None
    deg = 0
    rms = 0

    def __init__(self, x, y, z, deg=(1, 2), pw=1, robust=True):
        # determine data scale
        if robust:
            self.x_scale = np.ptp(np.percentile(x, q=[84, 16])) / 2
            self.x_mean = np.percentile(x, q=50)
            self.y_scale = np.ptp(np.percentile(y, q=[84, 16])) / 2
            self.y_mean = np.percentile(y, q=50)
            self.z_scale = np.ptp(np.percentile(z, q=[84, 16])) / 2
            self.z_mean = np.percentile(z, q=50)
        else:
            self.x_scale = np.std(x)
            self.x_mean = np.mean(x)
            self.y_scale = np.std(y)
            self.y_mean = np.mean(y)
            self.z_scale = np.std(z)
            self.z_mean = np.mean(z)

        # normalize data
        self.x_sc = (x - self.x_mean) / self.x_scale
        self.y_sc = (y - self.y_mean) / self.y_scale
        self.z_sc = (z - self.z_mean) / self.z_scale

        # fit data
        p0 = np.zeros((deg[0] + 1, deg[1] + 1), dtype=float).flatten()
        self.opt = least_squares(cost2d, p0, method="lm", args=(self.x_sc, self.y_sc, self.z_sc, deg, pw))
        self.p = self.opt.x

        self.pw = pw
        self.robust = robust
        self.deg = deg

    def __call__(self, x, y):
        x_sc = (x - self.x_mean) / self.x_scale
        y_sc = (y - self.y_mean) / self.y_scale
        z_sc = np.polynomial.chebyshev.chebval2d(
            x_sc, y_sc, self.p.reshape(self.deg[0] + 1, self.deg[1] + 1))
        return z_sc * self.z_scale + self.z_mean


def test_poly1d():
    xx = np.linspace(1001, 1002, 100)
    yy = (xx - 1001.5) ** 2 * 100 + 100 + np.random.normal(0, 1, xx.shape)

    fig = plt.figure()
    plt.plot(xx, yy, 'x')
    poly1d1 = PolyFit1D(xx, yy, deg=2, pw=1)
    plt.plot(xx, poly1d1(xx), 'r-', label="deg=2, pw=1")
    poly1d2 = PolyFit1D(xx, yy, deg=2, pw=2)
    plt.plot(xx, poly1d2(xx), 'r-', label="deg=2, pw=2")
    poly1dnp = np.polyfit(xx, yy, deg=2)
    plt.plot(xx, np.polyval(poly1dnp, xx), color="gray", label="numpy.polyfit")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    fig.tight_layout()


def test_poly2d():
    xx = np.random.uniform(1001, 1002, 1000)
    yy = np.random.uniform(2001, 2002, 1000)
    zz = (xx - 1001.5) ** 2 - (yy - 2001.5) ** 2 + 100

    xgrid, ygrid = np.meshgrid(
        np.linspace(1001, 1002, 100),
        np.linspace(2001, 2002, 100),
    )

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xx, yy, zz, s=5, c="cyan", label="Data")
    poly2d = PolyFit2D(xx, yy, zz, deg=(2, 2), pw=2)

    zpred = poly2d(xgrid, ygrid)
    ax.plot_surface(
        xgrid, ygrid, zpred, rstride=1, cstride=1, cmap=plt.cm.coolwarm,
        linewidth=0, antialiased=False, alpha=.5, label="Poly2dFitter")
    # ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()


if __name__ == "__main__":
    test_poly1d()
    test_poly2d()
