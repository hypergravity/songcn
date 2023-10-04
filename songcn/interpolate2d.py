from typing import Union

import matplotlib.pyplot as plt
import numpy as np

"""
Well, it seems no exact solution can be found. 

References
----------
https://en.wikipedia.org/wiki/Optical_flow
"""


def cumsum_nd(x: np.ndarray):
    cs = np.copy(x)
    for i_dim in range(cs.ndim):
        cs = np.cumsum(cs, axis=i_dim)
    return cs


def cumsum_lastd(x: np.ndarray):
    return np.cumsum(x, axis=-1)


def diff_nd(cs: np.ndarray):
    df = np.copy(cs)
    for i_dim in range(df.ndim):
        df = np.diff(df, axis=i_dim, prepend=0)
    return df


def diff_lastd(cs: np.ndarray):
    return np.diff(cs, axis=-1, prepend=0)


def test_cumsum_diff():
    a = np.ones((2, 3, 4))
    cs = cumsum_nd(a)
    df = diff_nd(cs)
    print(a)
    print(cs)
    print(df)


def gaussian_2d():
    return


# suppose the sampling is good enough
# we DO NOT over-sample distributions any more for ND cases
def interpolate_distribution_nd(
    *p: np.ndarray, f: Union[float, np.ndarray] = 0.5
) -> np.ndarray:
    n_p = len(p)
    p_shape = p[0].shape
    p_ndim = p[0].ndim
    cs = np.zeros(shape=(n_p, *p_shape), dtype=float)

    # evaluate CDFs
    for i_p in range(n_p):
        cs[i_p] = cumsum_nd(p[i_p])

    # determine scales
    scale = cs[:, *[-1 for i in range(p_ndim)]]

    # determine weights
    if isinstance(f, float):
        f = np.ones(n_p, float) * f
    else:
        f = np.array(f)
    # normalize weights
    f /= np.sum(f)

    print(f, scale)
    print(f.shape, scale.shape, cs.shape)
    # weighted mean of all CDFs
    cs_interp = np.mean(np.einsum("n,n...->n... ", f * scale, cs), axis=0)
    print(cs_interp.shape)
    # convert CDF to PDF
    df_interp = diff_nd(cs_interp)
    print(df_interp.shape)
    return df_interp


def gaussian_2d(
    *xi,  # n_dim*n_point
    mu: np.ndarray = np.array([[0.0], [0.0]]),  # n_dim*n_point
    sigma: np.ndarray = np.array([[1.0, 0.5], [0.5, 1.0]]),  # n_dim*n_dim
):
    # dimension
    ndim = len(xi)
    # xi shape
    x_shape = xi[0].shape
    # number of points
    # n_point = np.prod(x_shape)

    # vectorize coordinates
    x = np.array(xi).reshape(ndim, -1)  # n*p

    # inverse covariance
    inv_sigma = np.linalg.inv(sigma)  # n_dim*n_dim

    # difference
    diff = np.array(x) - mu.reshape(ndim, -1)

    # exponent
    exponent = np.einsum("np,nm,mp->p", diff, inv_sigma, diff)
    # exponent = -0.5 * np.dot(np.dot(diff.T, inv_sigma), diff)

    # normalization factor
    norm = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(sigma)))
    # gaussian value
    value = norm * np.exp(-exponent)

    # reshape
    return value.reshape(x_shape)


def test_interpolate_distribution_nd():
    x1grid = np.linspace(-3, 3, 100)
    x2grid = np.linspace(-3, 3, 100)
    im0 = gaussian_2d(*np.meshgrid(x1grid, x2grid), mu=np.array([-1, -1]))
    im1 = gaussian_2d(*np.meshgrid(x1grid, x2grid), mu=np.array([1, 1]))
    im_interp = interpolate_distribution_nd(im0, im1, f=1.0)

    plt.imshow(cumsum_nd(im0), origin="lower")

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    axs[0].imshow(cumsum_nd(im0))
    axs[2].imshow(cumsum_nd(im1))
    axs[1].imshow(cumsum_nd(im_interp))

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    axs[0].imshow(cumsum_lastd(im0))
    axs[2].imshow(cumsum_lastd(im1))
    axs[1].imshow(cumsum_lastd(im_interp))

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    axs[0].imshow(im0)
    axs[2].imshow(im1)
    axs[1].imshow(im_interp)


def test_optical_flow():
    x1grid = np.linspace(-3, 3, 100)
    x2grid = np.linspace(-3, 3, 100)
    gray1 = gaussian_2d(*np.meshgrid(x1grid, x2grid), mu=np.array([-1, -1])).astype(
        np.float32
    )
    gray2 = gaussian_2d(*np.meshgrid(x1grid, x2grid), mu=np.array([1, 1])).astype(
        np.float32
    )

    gray1 = (gray1 / gray1.max() * 255).astype(np.uint8)
    gray2 = (gray2 / gray2.max() * 255).astype(np.uint8)

    import cv2

    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 0.5, 10, 100, 100, 5, 1.2, 0
    )
    interpolated_frame = cv2.remap(gray1, flow, None, cv2.INTER_LINEAR)

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    axs[0].imshow(gray1)
    axs[2].imshow(gray2)
    axs[1].imshow(interpolated_frame)

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    axs[0].imshow(cumsum_nd(gray1))
    axs[2].imshow(cumsum_nd(gray2))
    axs[1].imshow(cumsum_nd(interpolated_frame))
