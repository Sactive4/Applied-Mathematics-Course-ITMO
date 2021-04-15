DIFF_STEP = 1 / 2 ** 20


def partial_derivative(f, x, i):
    """Найти частную производную функции f в точке x по аргументу i
    f - дифференцируемая функция
    x - вектор координат точки, len(x) совпадает с числом параметров f
    i - номер координаты, по которой дифференцирование, 0 <= i < len(x)
    """

    x1 = x.copy()
    x2 = x.copy()
    x1[i] = x1[i] + DIFF_STEP
    x2[i] = x1[i] - DIFF_STEP
    f1 = f(*x1)
    f2 = f(*x2)

    return (f1 - f2) / (x1[i] - x2[i])


def second_partial_derivative(f, x, i1, i2):
    x1 = x.copy()
    x2 = x.copy()
    x1[i2] = x1[i2] + DIFF_STEP
    x2[i2] = x1[i2] - DIFF_STEP
    d1 = partial_derivative(f, x1, i1)
    d2 = partial_derivative(f, x2, i1)

    return (d1 - d2) / (x1[i2] - x2[i2])


def gradient(f, x):
    """Найти градиент функции f в точке x
    f - дифференцируемая функция
    x - вектор координат точки, len(x) совпадает с числом параметров f
    """

    result = x.copy()  # для сохранения типа и размерности вектора

    for i in range(len(x)):
        result[i] = partial_derivative(f, x, i)

    return result


def norm(x):
    return sum(map(lambda num: num ** 2, x)) ** (1 / 2)


def normalize(x):
    y = x.copy()
    length = norm(y)
    for i in range(len(y)):
        y[i] /= length
    return y


def norm_gradient(f, x):
    return normalize(gradient(f, x))
