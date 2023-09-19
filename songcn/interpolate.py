from collections.abc import Iterable
from typing import Union

import matplotlib.pyplot as plt
import numpy as np


def gaussian1d(x, a=1.0, b=0.0, c=1.0):
    return a * np.exp(-0.5 * ((x - b) / c) ** 2)


def interpolate_distribution(
    p1: np.ndarray, p2: np.ndarray, f: Union[float, np.ndarray] = 0.5
):
    # construct x coordinates
    n_element = len(p1)
    x = np.arange(n_element, dtype=float)

    # normalize PDFs
    p1_sum = np.sum(p1)
    p2_sum = np.sum(p2)
    p1_norm = p1 / p1_sum
    p2_norm = p2 / p2_sum

    # PDF -> CDF
    c1 = np.cumsum(p1_norm)  # 1x
    c2 = np.cumsum(p2_norm)  # 1x

    # stack CDF1 and CDF2
    c_12 = np.sort(np.hstack((c1, c2)))  # 2x
    x_1to12 = np.interp(c_12, c1, x)  # 2x
    x_2to12 = np.interp(c_12, c2, x)  # 2x

    if not isinstance(f, Iterable):
        x_interp = x_2to12 * f + x_1to12 * (1.0 - f)  # 2x, interpolated x
        c_interp = np.interp(x, x_interp, c_12)  # 1x, interpolated cdf
        p_interp = (p2_sum * f + p1_sum * (1.0 - f)) * np.hstack(
            (c_interp[0], np.diff(c_interp))
        )
        return p_interp

    else:
        p_interp = np.zeros((len(f), n_element))
        for i, _f in enumerate(f):
            x_interp = x_2to12 * _f + x_1to12 * (1.0 - _f)  # 2x, interpolated x
            c_interp = np.interp(x, x_interp, c_12)  # 1x, interpolated cdf
            p_interp[i] = (p2_sum * _f + p1_sum * (1.0 - _f)) * np.hstack(
                (c_interp[0], np.diff(c_interp))
            )
        return p_interp


def plot_demo(ax_pdf, ax_cdf, xx, p0, p1):
    ax_pdf.plot(xx, p0, "-", c="tab:gray", lw=2, label="distribution 1")
    ax_pdf.plot(xx, p1, "-", c="tab:gray", lw=2, label="distribution 2")
    ax_pdf.plot(
        xx,
        interpolate_distribution(p0, p1, f=0.3),
        "-",
        lw=2,
        label="interpolated f=0.3",
    )
    ax_pdf.plot(
        xx,
        interpolate_distribution(p0, p1, f=0.6),
        "-",
        lw=2,
        label="interpolated f=0.6",
    )
    # ax_pdf.legend(loc="upper left", fontsize=10)

    ax_cdf.plot(xx, np.cumsum(p0), "-", c="tab:gray", lw=2, label="distribution 1")
    ax_cdf.plot(xx, np.cumsum(p1), "-", c="tab:gray", lw=2, label="distribution 2")
    ax_cdf.plot(
        xx,
        np.cumsum(interpolate_distribution(p0, p1, f=0.3)),
        "-",
        lw=2,
        label="interpolated f=0.3",
    )
    ax_cdf.plot(
        xx,
        np.cumsum(interpolate_distribution(p0, p1, f=0.6)),
        "-",
        lw=2,
        label="interpolated f=0.6",
    )
    ax_cdf.legend(loc="upper left")


if __name__ == "__main__":
    xx = np.linspace(0, 10, 1000)
    p0 = gaussian1d(xx, a=1, b=3, c=0.5)
    p1 = gaussian1d(xx, a=1, b=6, c=0.5)
    p2 = gaussian1d(xx, a=2, b=6, c=0.5)
    p3 = gaussian1d(xx, a=2, b=6, c=1)
    p4 = 0.5 * (p0 + p1)
    p5 = np.ones_like(p0) * np.mean(p0)

    fig, axs = plt.subplots(2, 5, figsize=(16, 7), sharex="all", sharey="row")
    plot_demo(axs[0, 0], axs[1, 0], xx, p0, p1)
    plot_demo(axs[0, 1], axs[1, 1], xx, p0, p2)
    plot_demo(axs[0, 2], axs[1, 2], xx, p0, p3)
    plot_demo(axs[0, 3], axs[1, 3], xx, p0, p4)
    plot_demo(axs[0, 4], axs[1, 4], xx, p0, p5)

    for i in range(5):
        axs[1, i].set_xlabel("X")
    axs[0, 0].set_ylabel("PDF")
    axs[1, 0].set_ylabel("CDF")
    fig.tight_layout()
