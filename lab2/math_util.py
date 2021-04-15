DIFFERENTIATION_STEP = 1e-9


def partial_derivative(f, x, i):
    """Найти частную производную функции f в точке x по аргументу i
    f - дифференцируемая функция
    x - вектор координат точки, len(x) совпадает с числом параметров f
    i - номер координаты, по которой дифференцирование, 0 <= i < len(x)
    """

    f_x = f(*x)
    x1 = x.copy()
    x1[i] = x1[i] + DIFFERENTIATION_STEP
    f_x1 = f(*x1)

    return (f_x1 - f_x) / (x1[i] - x[i])


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
