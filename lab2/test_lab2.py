from math import isclose

import numpy as np

from math_util import DIFFERENTIATION_STEP, partial_derivative


def test_partial_derivative_hyperbola():
    fn = lambda x: x ** 3

    assert isclose(
        partial_derivative(fn, [-2.], 0),
        12.,
        abs_tol=DIFFERENTIATION_STEP,
    )

    assert isclose(
        partial_derivative(fn, [2.], 0),
        12.,
        abs_tol=DIFFERENTIATION_STEP,
    )

    assert isclose(
        partial_derivative(fn, [0.], 0),
        0.,
        abs_tol=DIFFERENTIATION_STEP,
    )


def test_partial_derivative_paraboloid():
    fn = lambda x, y: x ** 2 + y ** 2

    assert isclose(
        partial_derivative(fn, [1., 2.], 0),
        2.,
        abs_tol=DIFFERENTIATION_STEP,
    )

    assert isclose(
        partial_derivative(fn, [1., 2.], 1),
        4.,
        abs_tol=DIFFERENTIATION_STEP,
    )

    assert isclose(
        partial_derivative(fn, [0., 2.], 0),
        0.,
        abs_tol=DIFFERENTIATION_STEP,
    )


def test_partial_derivative_with_numpy_arrays():
    fn = lambda x, y: x ** 2 + y ** 2

    assert isclose(
        partial_derivative(fn, np.array([1., 2.]), 1),
        4.,
        abs_tol=DIFFERENTIATION_STEP,
    )
