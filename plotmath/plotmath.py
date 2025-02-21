import pathlib

import numpy as np
import matplotlib.pyplot as plt
import shutil
import platform
import warnings

# Check if LaTeX is available
if platform.system() == "Windows":
    latex_available = shutil.which("latex.exe") is not None
else:
    latex_available = shutil.which("latex") is not None


if latex_available:
    try:
        plt.rc("text", usetex=True)
    except (FileNotFoundError, RuntimeError):
        plt.rc("text", usetex=False)
else:
    plt.rc("text", usetex=False)


colors = ["#029386", "#C875C4", "#E50000", "blue", "purple", "orange"]
# colors = np.random.permutation(colors)
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)


def show():
    plt.show()


def _get_figure_and_axis():
    fig, ax = plt.subplots()
    ax.spines["left"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["bottom"].set_position("zero")
    ax.spines["top"].set_color("none")

    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    ax.set_xlabel(r"$x$", fontsize=16, loc="right")
    ax.set_ylabel(r"$y$", fontsize=16, loc="top", rotation="horizontal")

    return fig, ax


def _get_figures_and_axes(n, m, figsize):
    figs, axes = plt.subplots(n, m, figsize=figsize)
    for ax in axes.flat:
        ax.spines["left"].set_position("zero")
        ax.spines["right"].set_color("none")
        ax.spines["bottom"].set_position("zero")
        ax.spines["top"].set_color("none")

        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

        ax.set_xlabel(r"$x$", fontsize=16, loc="right")
        ax.set_ylabel(r"$y$", fontsize=16, loc="top", rotation="horizontal")

    return figs, axes


def _set_ticks(xmin, xmax, ymin, ymax, xstep, ystep):

    xticks = list(np.arange(xmin + xstep, xmax, xstep))

    if 0 in xticks:
        xticks.remove(0)
    plt.xticks(xticks, fontsize=16)

    yticks = list(np.arange(ymin + ystep, ymax, ystep))

    if 0 in yticks:
        yticks.remove(0)
    plt.yticks(yticks, fontsize=16)

    return None


def _set_multiple_ticks(xmin, xmax, ymin, ymax, xstep, ystep, axes):
    xticks = list(np.arange(xmin, xmax, xstep))
    if 0 in xticks:
        xticks.remove(0)
    xticklabels = [f"${i}$" for i in xticks]

    yticks = list(np.arange(ymin, ymax, ystep))
    if 0 in yticks:
        yticks.remove(0)

    yticklabels = [f"${i}$" for i in yticks]

    for ax in axes.flat:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=16)

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=16)

    return axes


def plot(
    functions,
    fn_labels=True,
    xmin=-6,
    xmax=6,
    ymin=-6,
    ymax=6,
    xstep=1,
    ystep=1,
    ticks=True,
    alpha=0.7,
    grid=True,
):
    fig, ax = _get_figure_and_axis()

    if ticks:
        _set_ticks(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            xstep=xstep,
            ystep=ystep,
        )
    else:
        plt.xticks([])
        plt.yticks([])

    # domain = [xmin, xmax]
    # try:
    #     tmp = [f(xmin) for f in functions]
    #     tmp = [f(xmax) for f in functions]
    # except:
    #     raise ValueError(
    #         f"One of the provided functions is not defined on the provided domain: {domain}"
    #     )

    x = np.linspace(xmin, xmax, int(2**12))

    if isinstance(fn_labels, bool) and fn_labels:  # If True, automatically set labels
        fn_labels = [f"${fn.__name__}$" for fn in functions]
    elif isinstance(fn_labels, bool) and not fn_labels:  # If False, disable labels
        fn_labels = None
    # Otherwise the code will apply user-specified labels

    if fn_labels is not None:
        for f, label in zip(functions, fn_labels):
            ax.plot(x, f(x), lw=2, alpha=alpha, label=label)

        ax.legend(fontsize=16)

    else:
        for f in functions:
            ax.plot(x, f(x), lw=2, alpha=alpha)

    plt.ylim(ymin, ymax)
    plt.xlim(xmin, xmax)

    if grid:
        plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    return fig, ax


def multiplot(
    functions,
    fn_labels=False,
    xmin=-6,
    xmax=6,
    ymin=-6,
    ymax=6,
    xstep=1,
    ystep=1,
    ticks=True,
    alpha=0.7,
    grid=True,
    rows=2,
    cols=2,
    figsize=(8, 6),
):
    figs, axes = _get_figures_and_axes(rows, cols, figsize)

    if ticks:
        axes = _set_multiple_ticks(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            xstep=xstep,
            ystep=ystep,
            axes=axes,
        )
    else:
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])

    x = np.linspace(xmin, xmax, int(2**12))

    if isinstance(fn_labels, bool) and fn_labels:  # If True, automatically set labels
        fn_labels = [f"${fn.__name__}$" for fn in functions]
    elif isinstance(fn_labels, bool) and not fn_labels:  # If False, disable labels
        fn_labels = None
    else:
        fn_labels = [f"${label}$" for label in fn_labels]

    for f, ax, label in zip(functions, axes.flat, fn_labels):
        ax.plot(x, f(x), lw=2, alpha=alpha, label=label)
        ax.legend(fontsize=16)

    for ax in axes.flat:
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)

    if grid:
        for ax in axes.flat:
            ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    return figs, axes


def savefig(dirname, fname):
    dir = pathlib.Path(dirname)
    dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{dir}/{fname}")

    return None


def plot_polygon(*points, ax, color="teal", alpha=0.1):
    points = [*points]
    points.append(points[0])

    x, y = zip(*points)
    ax.fill(x, y, color=color, alpha=0.1)
    ax.plot(x, y, color="black", alpha=1, lw=1.5)
    ax.plot(x, y, "ko", markersize=8, alpha=0.7)

    return None


def histogram(
    xmin,
    frequencies,
    binsizes,
    xlabel=None,
    ylabel=None,
    rotation=0,
    fontsize=16,
    norwegian=True,
):
    heights = [f / b for f, b in zip(frequencies, binsizes)]
    fig, ax = _get_figure_and_axis()

    x = [xmin]
    for i, binsize in enumerate(binsizes):
        x.append(x[i] + binsize)

    xticks = x[:]
    if 0 in xticks:
        xticks.remove(0)

    ax.set_xticks(xticks)
    ax.set_xticklabels([f"${i}$" for i in xticks], fontsize=fontsize, rotation=0)

    yticks = list(np.arange(1, max(heights) + 1, 1))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"${int(i)}$" for i in yticks], fontsize=fontsize, rotation=0)

    if ylabel:
        ax.set_ylabel(
            ylabel,
            fontsize=fontsize,
            rotation=0,
            loc="top",
        )
    else:
        if norwegian:
            ax.set_ylabel(
                r"$\displaystyle \frac{\mathrm{Frekvens}}{\mathrm{Klassebredde}}$",
                fontsize=fontsize,
                rotation=0,
                loc="top",
            )
        else:
            ax.set_ylabel(
                r"$\displaystyle \frac{\mathrm{Frequency}}{\mathrm{Binsize}}$",
                fontsize=fontsize,
                rotation=0,
                loc="top",
            )

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=16, loc="right")
    else:
        ax.set_xlabel(r"$x$", fontsize=16, loc="right")

    plt.xlim(-1, x[-1] + 4)
    plt.ylim(-0.5, max(heights) + 1)

    for i in range(len(x) - 1):
        ax.plot(
            [x[i], x[i + 1]], [heights[i], heights[i]], color="teal", lw=2, alpha=0.7
        )
        ax.plot([x[i], x[i]], [0, heights[i]], color="teal", lw=2, alpha=0.7)
        ax.plot([x[i + 1], x[i + 1]], [0, heights[i]], color="teal", lw=2, alpha=0.7)

        ax.fill(
            [x[i], x[i + 1], x[i + 1], x[i]],
            [0, 0, heights[i], heights[i]],
            color="teal",
            alpha=0.3,
        )

    ax.grid(False)

    plt.subplots_adjust(
        left=0.25,
        bottom=0.068,
        top=0.885,
    )

    return fig, ax
