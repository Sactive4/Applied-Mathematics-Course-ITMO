from math import sin, log1p

import math


minimizers = []


def minimizer(fn):
    """Декоратор для регистрации алгоритмов"""
    minimizers.append(fn)


def f(x):
    """Исходная функция"""
    return math.sin(x) - math.log1p(x) ** 2 - 1


@minimizer
def dichotomy_method(a0, b0, eps):
    """Метод дихотомии"""
    a = a0
    b = b0
    delta = eps / 2

    interval_length = abs(b - a)
    algo_iters = 0
    func_calls = 0

    while interval_length > eps:
        x1 = (a + b - delta) / 2
        x2 = (a + b + delta) / 2
        if f(x1) > f(x2):
            a = x1
        elif f(x1) < f(x2):
            b = x2
        else:
            a = x1
            b = x2

        interval_length = abs(b - a)
        algo_iters += 1
        func_calls += 2

    return (a + b) / 2.0, algo_iters, func_calls


# TODO Метод золотого сечения - пока в разработке
@minimizer
def golden_ratio_method(a0, b0, eps):
    """Метод золотого сечения"""
    a = a0
    b = b0
    interval_length = abs(b - a)

    phi = (math.sqrt(5) - 1) / 2
    x1 = a + phi * interval_length
    x2 = b - phi * interval_length
    y1 = f(x1)
    y2 = f(x2)

    algo_iters = 0
    func_calls = 2

    while interval_length > eps:
        if y1 >= y2:
            a = x1
            x1 = x2
            x2 = a + phi * (b - a)
            y1 = y2
            y2 = f(x2)
            func_calls += 1
        else:
            b = x2
            x2 = x1
            x1 = b - phi * (b - a)
            y2 = y1
            y1 = f(x1)
            func_calls += 1
        interval_length = b - a
        algo_iters += 1
    return (a + b) / 2.0, func_calls, algo_iters


if __name__ == "__main__":
    a0 = 0.1
    b0 = 7.0
    eps = 0.01

    for minimizer in minimizers:
        res, call_count, iter_count = minimizer(a0, b0, eps)
        print(
            f"Метод: {minimizer.__name__}",
            f"Результат: {res:.3f}",
            f"Вызовов функции: {call_count}",
            f"Итераций: {iter_count}",
            sep="\n",
            end="\n\n",
        )
