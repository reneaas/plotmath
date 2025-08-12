import plotmath
import numpy as np


def f(x):
    return x**2 * np.cos(x)


fix, ax = plotmath.plot(
    functions=[f],
    xmin=-6,
    xmax=6,
    ymin=-12,
    ymax=12,
    xstep=1,
    ystep=2,
)

plotmath.savefig(
    dirname="../figures",
    fname="example_2.svg",
)

plotmath.show()
