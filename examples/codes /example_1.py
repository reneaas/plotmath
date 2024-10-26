import plotmath


def f(x):
    return x**2 - x - 2


fix, ax = plotmath.make_figure(
    functions=[f],
)

plotmath.savefig(
    dirname="../figures",
    fname="example_1.svg",
)

plotmath.show()
