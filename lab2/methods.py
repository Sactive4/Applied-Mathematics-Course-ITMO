import numpy as np
from math_util import gradient, hessian_matrix, norm, norm_gradient, normalize
from steps import lambda_quickest_descent_golden
from primathlab1.alg import golden_ratio_method, fibonacci_method


# предельное число итераций (это значит, что алгоритм не сходится)
M = 50000


def gradient_method_const(f, x0, eps, t_0, alpha=1.):
    """Найти минимум методом градиентного спуска
    f - многомерная функция
    x0 - начальный вектор-точка
    eps - требуемая точность
    t_0 - начальный шаг
    alpha - множитель <= 1
    """

    trajectory = [x0]

    x_k = x0
    x_kx1 = x0
    t_k = t_0

    k = 0

    while k < M:
        k += 1

        # следующая итерация, поменять значения
        x_k = x_kx1

        # вычислить следующий шаг
        grad = gradient(f, x_k)

        if norm(grad) < eps:
            break

        grad = normalize(grad)

        #t_k = t_0 * alpha ** k
        t_k *= alpha

        if k % 30 == 0:
            t_k = t_0

        x_kx1 = x_k - t_k * grad

        while f(*x_kx1) - f(*x_k) >= 0:
            t_k /= 2
            x_kx1 = x_k - t_k * grad

        trajectory.append(x_kx1)

        length = norm(x_kx1 - x_k)
        if length < eps and abs(f(*x_k) - f(*x_kx1)) < eps:
            break

    return trajectory


def gradient_method_quick_golden(f, x0, eps):
    """Найти минимум методом наискорейшего градиентного спуска
    f - многомерная функция
    x0 - начальный вектор-точка
    eps - требуемая точность
    """

    trajectory = [x0]

    x_k = x0
    x_kx1 = x0

    k = 0

    while k < M:
        k += 1

        # следующая итерация, поменять значения
        x_k = x_kx1

        # вычислить следующий шаг
        grad = gradient(f, x_k)

        if norm(grad) < eps:
            break

        grad = normalize(grad)

        f_min = lambda t: f(*(x_k - t * grad))
        u = golden_ratio_method(f_min, 0., 1000., eps)
        t_k = sum(u[-1]) / 2

        x_kx1 = x_k - t_k * grad
        trajectory.append(x_kx1)

        length = norm(x_kx1 - x_k)
        if length < eps and abs(f(*x_k) - f(*x_kx1)) < eps:
            break

    return trajectory


def gradient_method_quick_fibonacci(f, x0, eps):
    """Найти минимум методом наискорейшего градиентного спуска
    f - многомерная функция
    x0 - начальный вектор-точка
    eps - требуемая точность
    """

    trajectory = [x0]

    x_k = x0
    x_kx1 = x0

    k = 0

    while k < M:
        k += 1

        # следующая итерация, поменять значения
        x_k = x_kx1

        # вычислить следующий шаг
        grad = gradient(f, x_k)

        if norm(grad) < eps:
            break

        grad = normalize(grad)

        f_min = lambda t: f(*(x_k - t * grad))
        u = fibonacci_method(f_min, 0., 1000., eps)
        t_k = sum(u[-1]) / 2

        x_kx1 = x_k - t_k * grad
        trajectory.append(x_kx1)

        length = norm(x_kx1 - x_k)
        if length < eps and abs(f(*x_k) - f(*x_kx1)) < eps:
            break

    return trajectory


def conjugate_method(f, x0, eps, lambda_b):
    """Найти минимум общим сопряженным методом
    f - многомерная функция
    x0 - начальный вектор-точка
    eps - требуемая точность
    choose_step - лямбда для подсчета шага
    b - лямбда для подсчета beta
    """

    trajectory = [x0]

    k = 0

    x_k = x0
    x_next = x0

    t = 0

    grad = grad_prev = norm_gradient(f, x_k)
    d = -norm_gradient(f, x_k)

    while k < M:

        x_k = x_next

        grad_prev = grad
        grad = gradient(f, x_k)

        if norm(grad) < eps:
            break

        grad = normalize(grad)
        
        k += 1

        b = lambda_b(grad, grad_prev)
        d = -grad + b * d

        f_min = lambda ttt: f(*(x_k + ttt * d))
        u = golden_ratio_method(f_min, 0., 10000000., eps)
        t = sum(u[-1]) / 2

        x_next = x_k + t * d

        trajectory.append(x_next)

        if norm(x_next - x_k) < eps and abs(f(*x_next) - f(*x_k)) < eps:
            break

    return trajectory


def conjugate_gradient_method(f, x0, eps):
    """Найти минимум методом сопряженных градиентов (Флетчера-Ривса)
    f - многомерная функция
    x0 - начальный вектор-точка
    eps - требуемая точность
    ОСТОРОЖНО! Его применение ограничено квадратичными функциями! т.е. H > 0
    """

    hessian = hessian_matrix(f, x0)
    if not np.all(np.linalg.eigvals(hessian) > 0):
        return None

    return conjugate_method(
        f, x0, eps, lambda grad, grad_prev: norm(grad) ** 2 / norm(grad_prev) ** 2
    )


def conjugate_direction_method(f, x0, eps, choose_step=lambda_quickest_descent_golden):
    """Найти минимум методом сопряженных направлений (Полака-Райбера)
    f - многомерная функция
    x0 - начальный вектор-точка
    eps - требуемая точность
    """

    return conjugate_method(
        f,
        x0,
        eps,
        lambda grad, grad_prev: np.dot(grad, grad - grad_prev) / norm(grad) ** 2,
    )


def newton_method(f, x0, eps):
    """Найти минимум методом Ньютона-Рафсона
    f - многомерная функция
    x0 - начальный вектор-точка
    eps - требуемая точность
    """

    trajectory = [x0]

    k = 0
    x = np.array(x0)
    prev_x = x0

    grad = gradient(f, x)

    while (
        norm(grad) >= eps
        and k < M
        and prev_x is x0 or (
            norm(prev_x - x) >= eps
            and abs(f(*x) - f(*prev_x)) >= eps
        )
    ):
        hessian = np.array(hessian_matrix(f, x))

        try:
            inv_hessian = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            print("Гессиан вырожден!!!")
            return None
            #direction = -grad
        else:
            if np.linalg.det(inv_hessian) > 0:
                direction = -np.matmul(inv_hessian, np.atleast_2d(grad).transpose())
                direction = direction.transpose()[0]
            else:
                direction = -grad

        prev_x = x.copy()
        x += direction

        grad = gradient(f, x)
        k += 1

        trajectory.append(x)

    return trajectory
