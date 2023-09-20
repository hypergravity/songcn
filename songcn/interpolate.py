from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable


def gaussian1d(x, a=1.0, b=0.0, c=1.0):
    return a * np.exp(-0.5 * ((x - b) / c) ** 2)


def interpolate_distribution(
    p1: np.ndarray, p2: np.ndarray, f: Union[float, np.ndarray] = 0.5
):
    """Interpolate between distribution ``p1`` and ``p2``.

    Parameters
    ----------
    p1: np.ndarray
        Distribution One.
    p2: np.ndarray
        Distribution Two.
    f: Union[float, np.ndarray]
        The distance fraction defined by ``(p_interp-p1)/(p2-p1)``.
        ``p_interp`` approaches ``p0 when f approaches 0.

    """
    # construct x coordinates
    p1 = np.insert(p1, 0, 0.0)
    p2 = np.insert(p2, 0, 0.0)
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

    if isinstance(f, float):
        assert 0 <= f <= 1, f"Valid f values are between 0 and 1. f={f}"
        x_interp = x_2to12 * f + x_1to12 * (1.0 - f)  # 2x, interpolated x
        c_interp = np.interp(x, x_interp, c_12)  # 1x, interpolated cdf
        p_interp = (p2_sum * f + p1_sum * (1.0 - f)) * np.hstack(
            (c_interp[0], np.diff(c_interp))
        )
        return p_interp[1:]

    else:
        assert isinstance(f, np.ndarray), f"type(f) is {type(f)}"
        assert np.all(f >= 0) and np.all(
            f <= 1
        ), f"Valid f values are between 0 and 1. f={f}"
        p_interp = np.zeros((len(f), n_element))
        for i, _f in enumerate(f):
            x_interp = x_2to12 * _f + x_1to12 * (1.0 - _f)  # 2x, interpolated x
            c_interp = np.interp(x, x_interp, c_12)  # 1x, interpolated cdf
            p_interp[i] = (p2_sum * _f + p1_sum * (1.0 - _f)) * np.hstack(
                (c_interp[0], np.diff(c_interp))
            )
        return p_interp[:, 1:]


def plot_demo(ax_clr, ax_pdf, ax_cdf, xx, p0, p1):
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
    ax_cdf.legend(loc="upper left", framealpha=0, fontsize=10)

    f = np.linspace(0, 1, 30)
    cmap = colormaps["jet"]
    for i in range(len(f)):
        ax_clr.plot(
            xx,
            interpolate_distribution(p0, p1, f=f[i]),
            "-",
            color=cmap(f[i]),
            alpha=0.5,
            lw=1,
        )
        ax_clr.plot(
            xx,
            interpolate_distribution(p0, p1, f=f[i]),
            "-",
            color=cmap(f[i]),
            alpha=0.5,
            lw=1,
        )

    divider = make_axes_locatable(ax_clr)
    cax = divider.append_axes("top", size="3%", pad=0.03)

    # ax_clr.legend(loc="upper left", framealpha=0, fontsize=10)
    sm = plt.cm.ScalarMappable(cmap="jet")
    sm.set_array([0, 1])  # 设置一个空数组
    plt.colorbar(sm, cax=cax, location="top", ticks=None)


def test_performance():
    xx = np.linspace(0, 10, 1000)
    p0 = gaussian1d(xx, a=1, b=3, c=0.5)
    p1 = gaussian1d(xx, a=1, b=6, c=0.5)
    f = np.linspace(0, 1, 100)
    return interpolate_distribution(p0, p1, f)


def test_demo1():
    plt.rcParams["font.size"] = 12
    xx = np.linspace(0, 10, 1000)
    p0 = gaussian1d(xx, a=1, b=3, c=0.5)
    p1 = gaussian1d(xx, a=1, b=6, c=0.5)
    p2 = gaussian1d(xx, a=2, b=6, c=0.5)
    p3 = gaussian1d(xx, a=1.5, b=6, c=1)
    p4 = 0.5 * (p0 + p1)
    p5 = np.ones_like(p0) * np.mean(p0)

    fig, axs = plt.subplots(3, 5, figsize=(16, 10), sharex="all", sharey="row")
    plot_demo(axs[0, 0], axs[1, 0], axs[2, 0], xx, p0, p1)
    plot_demo(axs[0, 1], axs[1, 1], axs[2, 1], xx, p0, p2)
    plot_demo(axs[0, 2], axs[1, 2], axs[2, 2], xx, p0, p3)
    plot_demo(axs[0, 3], axs[1, 3], axs[2, 3], xx, p0, p4)
    plot_demo(axs[0, 4], axs[1, 4], axs[2, 4], xx, p0, p5)

    for i in range(5):
        axs[-1, i].set_xlabel("X")
    axs[0, 0].set_ylabel("Interpolate with f from 0 to 1")
    axs[1, 0].set_ylabel("PDF")
    axs[2, 0].set_ylabel("CDF")

    fig.tight_layout()
    fig.savefig("figs/demo_interpolate_distribution.pdf")
    fig.savefig("figs/demo_interpolate_distribution.png")


def test_demo2():
    cmap = colormaps["jet"]
    f = np.linspace(0, 1, 100)
    xx = np.linspace(0, 10, 1000)
    p0 = gaussian1d(xx, a=1, b=3, c=0.5)
    p1 = gaussian1d(xx, a=1, b=6, c=0.5)
    p_interp = interpolate_distribution(p0, p1, f)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for i in range(len(f)):
        ax.plot(xx, p_interp[i], "-", color=cmap(f[i]), alpha=0.5)
    plt.colorbar(cmap, ax=ax)


def test_demo3():
    """compare to value interpolation"""
    plt.rcParams["font.size"] = 12
    # %%
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].plot([1, 4], [1, 4], "s-", color="gray", ms=10, lw=2, mec="k", label="data")
    axs[0].plot([2, 3], [2, 3], "s", ms=10, lw=2, mec="k", label="interpolated values")
    axs[0].legend(framealpha=0)
    axs[0].set_title("interpolate values")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")

    xx = np.linspace(0, 10, 1000)
    p0 = gaussian1d(xx, a=1, b=3, c=0.5)
    p1 = gaussian1d(xx, a=1, b=6, c=0.5)
    axs[1].plot(xx, p0, "-", color="gray", ms=10, lw=2, mec="k", label="data")
    axs[1].plot(xx, p1, "-", color="gray", ms=10, lw=2, mec="k")

    axs[1].plot(
        xx, 0.5 * (p0 + p1), "", ms=10, lw=3, mec="k", label="interpolated values"
    )
    # axs[1].legend(fontsize=10, framealpha=0)
    axs[1].set_title("interpolate distributions?")
    axs[1].set_xlabel("X")
    fig.tight_layout()
    fig.savefig("figs/demo_interpolation_problem.pdf")
    fig.savefig("figs/demo_interpolation_problem.png")


if __name__ == "__main__":
    test_demo1()
    test_demo3()
