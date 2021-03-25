from alg import Fibonacci


def test_fibonacci_pre_cached():
    f = Fibonacci()
    assert f.fib(10) == 89
    assert f.n(34) == 8


def test_fibonacci_not_cached():
    f = Fibonacci()
    assert f.n(34) == 8


def test_fibonacci_float():
    f = Fibonacci()
    assert f.n(34.1) == 9
