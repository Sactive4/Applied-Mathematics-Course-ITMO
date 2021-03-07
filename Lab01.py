from math import sin, log1p

import math

# ===== Исходная функция =====
def f(x):
    return math.sin(x) - math.log1p(x) ** 2 - 1

# ===== Метод дихотомии =====
def dichotomy_method(a0, b0, eps):
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
# ===== Метод золотого сечения =====
def golden_ratio_method(a0, b0, eps):
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

a0 = 0.1
b0 = 7.0
eps = 0.01

x1, f1, alg1 = dichotomy_method(a0, b0, eps)
print(x1)

x2, f2, alg2 = golden_ratio_method(a0, b0, eps)
print(x2)












