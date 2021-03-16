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
    intervals = [(a, b)]

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
        intervals.append((a, b))

    return intervals


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

    intervals = [(a, b)]

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
        intervals.append((a, b))
    return intervals


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


@minimizer
def fibonacci_method(f, a, b, eps):
    """Метод Фибоначчи"""

    fib = Fibonacci()

    fib_iters = (b - a) / eps
    n = fib.n(fib_iters) - 2

    x1 = a + fib.fib(n) / fib.fib(n + 2) * (b - a)
    x2 = a + fib.fib(n + 1) / fib.fib(n + 2) * (b - a)
    y1 = f(x1)
    y2 = f(x2)

    intervals = [(a, b)]

    for k in range(2, n + 1):
        if y1 > y2:
            a = x1
            x1, y1 = x2, y2
            x2 = a + fib.fib(n - k + 2) / fib.fib(n - k + 3) * (b - a)
            y2 = f(x2)
        else:
            b = x2
            x2, y2 = x1, y1
            x1 = a + fib.fib(n - k + 1) / fib.fib(n - k + 3) * (b - a)
            y1 = f(x1)
        intervals.append((a, b))

    return intervals


# todo метод парабол работает корректно только при шаге 0.01
# нужно убедиться, что это точно не баг, а фича метода
PARABOLA_STEP = 0.01

# todo протестировать
@minimizer
def parabola_method(f, a0, b0, eps):
    """Метод парабол"""

    global PARABOLA_STEP

    x2 = a0
    step = PARABOLA_STEP * a0
    intervals = []

    while True:

        x1 = x2 - step
        x3 = x2 + step

        f1 = f(x1)
        f2 = f(x2)
        f3 = f(x3)

        x_min = x2 - 0.5 * ((x2 - x1) ** 2 * (f2 - f3) - (x2 - x3) ** 2 * (f2 - f1)) / (
            (x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1)
        )
        intervals.append((x1, x3))

        if abs(x_min - x2) < eps:
            x2 = x_min
            break

        x2 = x_min

    return intervals


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


# todo протестировать
@minimizer
def brent1_method(f, a0, b0, eps):
    """Комбинированный метод Брента"""

    intervals = []

    a = a0
    c = b0

    K = (3 - sqrt(5)) / 2
    x = w = v = (a + c) / 2
    f_x = f_w = f_v = f(x)
    d = e = c - a

    while d > eps:
        g = e
        e = d

        u = 0
        if (
            (x != w)
            and (x != v)
            and (w != v)
            and (f_x != f_w)
            and (f_x != f_v)
            and (f_w != f_v)
        ):

            x1 = v
            x2 = x
            x3 = w

            f1 = f_v
            f2 = f_x
            f3 = f_w

            u = x2 - 0.5 * ((x2 - x1) ** 2 * (f2 - f3) - (x2 - x3) ** 2 * (f2 - f1)) / (
                (x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1)
            )

        if (a + eps <= u) and (u <= c - eps) and (abs(u - x) < 0.5 * g):
            intervals.append((x, u))
            d = abs(u - x)
        else:
            if x < 0.5 * (c + a):
                u = x + K * (c - x)
                d = c - x
                intervals.append((x, c))
            else:
                u = x - K * (x - a)
                d = x - a
                intervals.append((a, x))

            if abs(u - x) < eps:
                u = x + sign(u - x) * eps

            f_u = f(u)
            if f_u <= f_x:
                if u >= x:
                    a = x
                else:
                    c = x
                v = w
                w = x
                x = u
                f_v = f_w
                f_w = f_x
                f_x = f_u
            else:
                if u >= x:
                    c = u
                else:
                    a = u
                if (f_u <= f_w) or (w == x):
                    v = w
                    w = u
                    f_v = f_w
                    f_w = f_u
                elif (f_u <= f_v) or (v == x) or (v == w):
                    v = u
                    f_v = f_u

    return intervals
