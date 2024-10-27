import plotmath


def f(x):
    return x**2 - x - 2


fix, ax = plotmath.plot(
    functions=[f],
)

plotmath.savefig(
    dirname="../figures",
    fname="example_1.svg",
)

plotmath.show()
