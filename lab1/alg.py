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

    for k in range(2, n + 2):
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

def get_xs(f, x1, step, a0, b0):

    if ((x1 < a0) or (x1 > b0)):
        x1 = (a0 + b0) / 2

    x2 = x1 + step
    x2 = max(a0, min(b0, x2))

    f1 = f(x1)
    f2 = f(x2)
    if f1 > f2:
        return x2, max(a0, min(b0,  x1 + 2 * step)), f1, f2
    else:
        return x2, max(a0, min(b0,  x1 -  step)), f1, f2

def get_min_x_f(f1, f2, f3, x1, x2, x3):
    """Выбрать точку с минимальным значением функции"""

    f_min, x_min = min((f1, x1), (f2, x2), (f3, x3))
    return x_min, f_min


def square_approximation(f, f1, f2, f3, x1, x2, x3, step, a0, b0):

    #f1 = f(x1)
    #f2 = f(x2)
    #f3 = f(x3)

    x_min, f_min = get_min_x_f(f1, f2, f3, x1, x2, x3)

    if (x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1) == 0:
        x_1, x_2, x_3 = get_xs(f, x_min, step, a0, b0)
        return square_approximation(f, f1, f2, f3, x_1, x_2, x_3, step, a0, b0)

    else:
        u = 0.5 * ((x2 ** 2 - x3 ** 2) * f1 + (x3 ** 2 - x1 ** 2) * f2 + (x1 ** 2 - x2 ** 2) * f3) / (
            (x2 - x3) * f1 + (x3 - x1) * f2 + (x1 - x2) * f3
        )
        fu = f(u)
        return u, fu



@minimizer
def parabola_method2(f, a0, b0, eps):
    """Метод квадратичной аппроксимации"""

    eps /= 2
    intervals = []
    intervals.append((a0, b0))
    step = 0.05  # min(0.05, max(abs(b0 - a0) * eps, 10 * eps))

    x1 = (a0 + b0) / 2
    x2, x3, f1, f2 = get_xs(f, x1, step, a0, b0)
    f3 = f(x3)

    while True:

        x_min, f_min = get_min_x_f(f1, f2, f3, x1, x2, x3)
        u, fu = square_approximation(f, f1, f2, f3, x1, x2, x3, step, a0, b0)

        # if (abs(x3 - x1) < eps / 2):
        #     intervals.append((x1 - eps, x3 + eps))
        #     break

        # if ((abs(u - x_min) < eps) and (abs(x3 - x1) < eps)):
        #     intervals.append((u - eps, u + eps))
        #     break

        if (abs((f_min - fu) / fu) < eps) and (abs((x_min - u) / u) < eps):
            intervals.append((u - eps, u + eps))
            break

        else:
            if (u >= x1) and (u <= x3):
                if (f_min + eps < fu) and (x_min < x2):
                    x2 = x_min
                else:
                    x2 = u

                assert((x2 >= a0) and (x2 <= b0))
                x1 = max(a0, x2 - step)
                x3 = min(b0, x2 + step)
                f1 = f(x1)
                f2 = f(x2)
                f3 = f(x3)

            else:
                x1 = u
                # if (u >= a0) and (u <= b0):
                #     x1 = u
                # else:
                #     x1 = x_min
                x2, x3, f1, f2 = get_xs(f, x1, step, a0, b0)
                f3 = f(x3)

            intervals.append((x1, x3))

    return intervals


@minimizer
def parabola_method(f, a0, b0, eps):
    """Метод парабол"""

    intervals = []
    intervals.append((a0, b0))

    x1 = a0
    x3 = b0
    f1 = f(x1)
    f3 = f(x3)

    x2 = (x1 + x3) / 2
    f2 = f(x2)

    while abs(x3 - x1) > eps:

        u = x2 - 0.5 * ((x2 - x1) ** 2 * (f2 - f3) - (x2 - x3) ** 2 * (f2 - f1)) / (
            (x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1)
        )
        fu = f(u)

        if x2 < u:
            left_x, left_f = x2, f2
            right_x, right_f = u, fu
        else:
            left_x, left_f = u, fu
            right_x, right_f = x2, f2

        if left_f < right_f:
            x3, f3 = right_x, right_f
            x2, f2 = left_x, left_f
        else:
            x1, f1 = left_x, left_f
            x2, f2 = right_x, right_f

        intervals.append((x1, x3))

    return intervals


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


@minimizer
def brent_method(f, a0, b0, eps):
    """Комбинированный метод Брента"""

    intervals = []
    intervals.append((a0, b0))

    a = a0
    c = b0

    K = (3 - sqrt(5)) / 2
    x = w = v = (a + c) / 2
    f_x = f_w = f_v = f(x)
    d = e = c - a

    while d > eps:
        g = e
        e = d

        # todo не все итерации добавляются, а только уникальные. норм?
        if intervals[-1] != (a, c):
            intervals.append((a, c))

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
            d = abs(u - x)
        else:
            # todo опечатка в коде? я поставил +, а не -, как дано
            if x < 0.5 * (c + a):
                u = x + K * (c - x)
                d = c - x
            else:
                u = x - K * (x - a)
                d = x - a

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
