import mpmath as mp
mp.dps = 7; mp.pretty = True

DIFF_STEP = 1 / 2 ** 20
DIFF_STEP = 0.001

def partial_derivative(f, x, i):
    """Найти частную производную функции f в точке x по аргументу i
    df / dx_{i}
    f - дифференцируемая функция
    x - вектор координат точки, len(x) совпадает с числом параметров f
    i - номер координаты, по которой дифференцирование, 0 <= i < len(x)
    """
    mask = [0] * len(x)
    mask[i] = 1

    return mp.diff(f, x, mask)


def second_partial_derivative(f, x, i1, i2):
    """Найти вторую частную производную функции f в точке x по аргументам i1 и i2
    d^2f / ( dx_{i1} dx_{i2} )
    f - дифференцируемая функция
    x - вектор координат точки, len(x) совпадает с числом параметров f
    i1 - первая координата, по которой дифференцирование, 0 <= i < len(x)
    i2 - вторая координата, по которой дифференцирование, 0 <= i < len(x)
    """

    mask = [0] * len(x)
    mask[i1] += 1
    mask[i2] += 1

    return mp.diff(f, x, mask)


def gradient(f, x):
    """Найти градиент функции f в точке x
    f - дифференцируемая функция
    x - вектор координат точки, len(x) совпадает с числом параметров f
    """

    result = x.copy()  # для сохранения типа и размерности вектора

    for i in range(len(x)):
        result[i] = partial_derivative(f, x, i)

    return result


def hessian_matrix(f, x):
    """Найти найти матрицу Гессе функции f в точке x
    f - дифференцируемая функция
    x - вектор координат точки, len(x) совпадает с числом параметров f
    """

    return [
        [float(second_partial_derivative(f, x, i, j)) for j in range(len(x))]
        for i in range(len(x))
    ]


def norm(x):
    """Норма вектора x"""

    return sum(map(lambda num: num ** 2, x)) ** (1 / 2)


def normalize(x):
    """Найти нормализованный вектор x"""

    y = x.copy()
    length = norm(y)
    for i in range(len(y)):
        y[i] /= length
    return y


def norm_gradient(f, x):
    """Найти нормализованный градиент функции f в точке x
    f - дифференцируемая функция
    x - вектор координат точки, len(x) совпадает с числом параметров f
    """

    return normalize(gradient(f, x))
