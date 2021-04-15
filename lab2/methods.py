from math_util import norm, norm_gradient
import numpy as np
from primathlab1.alg import golden_ratio_method


INF = 100000000

def gradient_method(f, x0, eps, lambda_f):
    """Найти минимум методом градиентного спуска
    f - многомерная функция
    x0 - начальный вектор-точка
    eps - требуемая точность
    lambda_f(f, x_k, lambda_k) - метод вычисления шага. Примеры:
        - постоянный:  return 2 (метод может расходиться)
        - дробный шаг:  return lambda_k / 3
        - наискорейший спуск: return ...argmin lambda {f(x_k - lambda * gradient(x_k))}
    """

    # x_kx1 - x_{k+1} - следующая точка оптимизации
    # x_k   - x_{k}   - текушая точка оптимизации
    # lambda_k - текущий шаг
    # lambda_kx1 - следующий шаг

    x_k = x0
    x_kx1 = x0
    lambda_k = INF
    lambda_kx1 = INF

    k = 0

    while True:
        # следующая итерация, поменять значения
        x_k = x_kx1
        lambda_k = lambda_kx1

        # вычислить следующий шаг
        lambda_kx1 = lambda_f(f, x_k, lambda_k)

        #print(x_k)
        grad = norm_gradient(f, x_k)
        x_kx1 = x_k - lambda_kx1 * grad

        length = norm(x_kx1 - x_k)
        k += 1
        if (length < eps) and (abs(f(*x_k) - f(*x_kx1)) < eps):
            break
        if k > 100000:
            break
    
    return x_kx1


def lambda_const(f, x_k, lambda_k):
    return 0.5

def lambda_ratio(f, x_k, lambda_k):
    if (lambda_k == INF):
        return 0.5
    return lambda_k * 0.97

def lambda_quickest_descent(f, x_k, lambda_k):
    fu = lambda x: f(*(x_k - x * norm_gradient(f, x_k)))
    u = golden_ratio_method(fu, 0, 1000, 0.0001)
    v = sum(u[-1]) / 2
    return v


fn1 = lambda x, y: x ** 2 + y ** 2
# ответ 0, 0
fn2 = lambda x, y: 22 * ((x-100) ** 4) + 8 * (y ** 4)
# ответ 100, 0

def test_function(fn, x, eps, step):

    fn_answer1 = gradient_method(fn, x, eps, lambda x, y, z: step)
    print(fn_answer1)
    fn_answer2 = gradient_method(fn, x, eps, lambda_ratio)
    print(fn_answer2)
    fn_answer3 = gradient_method(fn, x, eps, lambda_quickest_descent)
    print(fn_answer3)

test_function(fn1, np.array([200., 10.]), 0.001, 0.5)
test_function(fn2, np.array([200., 10.]), 0.001, 1)
# ставлю


