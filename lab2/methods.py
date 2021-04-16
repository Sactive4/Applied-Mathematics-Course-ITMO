import numpy as np
from primathlab1.alg import golden_ratio_method

from math_util import gradient, hessian_matrix, norm, norm_gradient


# предельное число итераций (это значит, что алгоритм не сходится)
M = 50000


def gradient_method(f, x0, eps, choose_step):
    """Найти минимум методом градиентного спуска
    f - многомерная функция
    x0 - начальный вектор-точка
    eps - требуемая точность
    choose_step - (f, curr_x, prev_step, grad) - метод вычисления шага. Примеры:
        - постоянный:  return 2 (метод может расходиться)
        - дробный шаг:  return prev_step / 3
        - наискорейший спуск: return ...argmin lambda {f(curr_x - lambda * grad)}
    """

    # x_kx1 - x_{k+1} - следующая точка оптимизации
    # x_k   - x_{k}   - текушая точка оптимизации
    # lambda_k - текущий шаг
    # lambda_kx1 - следующий шаг

    x_k = x0
    x_kx1 = x0
    lambda_k = lambda_kx1 = float("inf")

    k = 0

    while k < M:

        # следующая итерация, поменять значения
        x_k = x_kx1
        lambda_k = lambda_kx1

        # вычислить следующий шаг
        grad = norm_gradient(f, x_k)
        lambda_kx1 = choose_step(f=f, curr_x=x_k, prev_step=lambda_k, grad=grad)
        x_kx1 = x_k - lambda_kx1 * grad

        length = norm(x_kx1 - x_k)

        if (length < eps) and (abs(f(*x_k) - f(*x_kx1)) < eps):
            break

        k += 1

    return x_kx1, k


def lambda_quickest_descent(f, curr_x, prev_step, grad, **kwargs):
    fu = lambda x: f(*(curr_x - x * grad))
    u = golden_ratio_method(fu, 0, 1000, 0.0001)
    v = sum(u[-1]) / 2
    return v


def quickest_descent_gradient_method(f, x0, eps):
    """Найти минимум методом наискорейшего градиентного спуска
    f - многомерная функция
    x0 - начальный вектор-точка
    eps - требуемая точность
    """
    return gradient_method(f, x0, eps, lambda_quickest_descent)


def check_monotonic(f, curr_x, grad, step):
    curr_f = f(*curr_x)
    next_f = f(*(curr_x - step * grad))
    return next_f <= curr_f


def reduce_until_monotonic(f, curr_x, grad, step):
    while not check_monotonic(f, curr_x, grad, step):
        step /= 2

    return step


def lambda_const_checked(step):
    return lambda f, curr_x, grad, **kwargs: reduce_until_monotonic(
        f, curr_x, grad, step
    )


class LambdaRatioChecked:
    def __init__(self, step, coef):
        self._curr_step = step
        self._coef = coef

    def __call__(self, f, curr_x, grad, **kwargs):
        step = reduce_until_monotonic(f, curr_x, grad, self._curr_step)
        self._curr_step *= self._coef
        return step


def lambda_const(step):
    return lambda **kwargs: step


def lambda_ratio(step, coef):
    return (
        lambda prev_step, **kwargs: step
        if prev_step == float("inf")
        else coef * prev_step
    )


def conjugate_method(f, x0, eps, lambda_b):
    """Найти минимум общим сопряженным методом
    f - многомерная функция
    x0 - начальный вектор-точка
    eps - требуемая точность
    b - лямбда для подсчета beta
    """

    k = 0

    x_prev = x0
    x_k = x0
    x_next = x0

    grad = grad_prev = norm_gradient(f, x_k)
    d = -gradient(f, x_k)

    while True:

        k += 1
        if k >= M:
            break

        x_prev = x_k
        x_k = x_next

        grad_prev = grad
        grad = norm_gradient(f, x_k)

        b = lambda_b(grad, grad_prev)
        d = -grad + b * d

        f_min = lambda t: f(*(x_k + t * d))
        u = golden_ratio_method(f_min, -100, 100, 0.001)
        t = sum(u[-1]) / 2

        x_next = x_k + t * d

        if (norm(x_next - x_k) < eps) and (abs(f(*x_next) - f(*x_k)) < eps):
            return x_k, k

        # if (norm(grad) < eps):
        #     return x_k, k

    return x_k, k


def conjugate_gradient_method(f, x0, eps):
    """Найти минимум методом сопряженных градиентов (Флетчера-Ривса)
    f - многомерная функция
    x0 - начальный вектор-точка
    eps - требуемая точность
    ОСТОРОЖНО! Его применение ограничено квадратичными функциями! т.е. H > 0
    """

    # todo: проверить гессиан на положительность

    return conjugate_method(
        f, x0, eps, lambda grad, grad_prev: (norm(grad) ** 2) / (norm(grad_prev) ** 2)
    )


def conjugate_direction_method(f, x0, eps):
    """Найти минимум методом сопряженных направлений (Полака-Райбера)
    f - многомерная функция
    x0 - начальный вектор-точка
    eps - требуемая точность
    """

    # todo: переписать эту строчку --- grad * (grad - grad_prev)
    # здесь нужно скалярное произведение
    return conjugate_method(
        f,
        x0,
        eps,
        lambda grad, grad_prev: (grad * (grad - grad_prev)) / (norm(grad) ** 2),
    )


def newton_method(f, x0, eps):
    """Найти минимум методом Ньютона-Рафсона
    f - многомерная функция
    x0 - начальный вектор-точка
    eps - требуемая точность
    """

    k = 0
    x = np.array(x0)
    prev_x = np.array([float("inf")] * len(x))

    grad = gradient(f, x)

    while (
        norm(grad) >= eps
        and k < M
        and norm(prev_x - x) >= eps
        and abs(f(*x) - f(*prev_x)) >= eps
    ):
        hessian = np.array(hessian_matrix(f, x))

        try:
            inv_hessian = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:  # TODO: вырожденная матрица - что делать?
            direction = -grad
        else:
            if np.linalg.det(inv_hessian) > 0:
                direction = -np.matmul(inv_hessian, np.atleast_2d(grad).transpose())
                direction = direction.transpose()[0]
            else:
                direction = -grad

        prev_x = x.copy()
        x += direction  # TODO: добавить выбор шага
        grad = gradient(f, x)
        k += 1

    return x, k
