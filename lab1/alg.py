from bisect import bisect_left
from math import sqrt


minimizers = []


def minimizer(fn):
    """Декоратор для регистрации алгоритмов"""
    minimizers.append(fn)
    return fn


@minimizer
def dichotomy_method(f, a0, b0, eps):
    """Метод дихотомии"""
    a = a0
    b = b0
    delta = eps / 2

    interval_length = abs(b - a)
    algo_iters = 0

    while interval_length > eps:
        x1 = (a + b - delta) / 2
        x2 = (a + b + delta) / 2
        y1 = f(x1)
        y2 = f(x2)
        if y1 > y2:
            a = x1
        elif y1 < y2:
            b = x2
        else:
            a = x1
            b = x2

        interval_length = abs(b - a)
        algo_iters += 1

    return (a + b) / 2.0, algo_iters


@minimizer
def golden_ratio_method(f, a0, b0, eps):
    """Метод золотого сечения"""
    a = a0
    b = b0
    interval_length = abs(b - a)

    phi = (3 - sqrt(5)) / 2
    x1 = a + phi * interval_length
    x2 = b - phi * interval_length
    y1 = f(x1)
    y2 = f(x2)

    algo_iters = 0

    while interval_length > eps:
        if y1 >= y2:
            a = x1
            x1 = x2
            x2 = b - phi * (b - a)
            y1 = y2
            y2 = f(x2)
        else:
            b = x2
            x2 = x1
            x1 = a + phi * (b - a)
            y2 = y1
            y1 = f(x1)
        interval_length = b - a
        algo_iters += 1
    return (a + b) / 2.0, algo_iters


class Fibonacci:
    """Класс для работы с числами Фибоначчи"""

    def __init__(self):
        self._cache = [1, 1]

    def _append_next(self):
        next_fib = sum(self._cache[-2:])
        self._cache.append(next_fib)

    def fib(self, n):
        """Найти n-ое число Фибоначчи, n >= 0"""

        while n >= len(self._cache):
            self._append_next()
        return self._cache[n]

    def n(self, fib):
        """Найти номер числа Фибоначчи, ближайшего сверху к fib"""

        if fib <= self._cache[-1]:
            return bisect_left(self._cache, fib)

        while fib > self._cache[-1]:
            self._append_next()
        return len(self._cache) - 1
