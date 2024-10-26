import plotmath


def f(x):
    return x**2 - 4


def g(x):
    return x + 2


fix, ax = plotmath.make_figure(
    functions=[f, g],
    xmin=-6,
    xmax=6,
    ymin=-6,
    ymax=6,
)

plotmath.savefig(
    dirname="../figures",
    fname="example_3.svg",
)

plotmath.show()
