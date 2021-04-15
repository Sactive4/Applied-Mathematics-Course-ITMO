from math_util import gradient, norm
import numpy as np

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

    while True:
        # следующая итерация, поменять значения
        x_k = x_kx1
        lambda_k = lambda_kx1

        # вычислить следующий шаг
        lambda_kx1 = lambda_f(f, x_k, lambda_k)

        #print(x_k)

        x_kx1 = x_k - lambda_kx1 * gradient(f, x_k)

        length = norm(x_kx1 - x_k)
        if (length < eps) and (abs(f(*x_k) - f(*x_kx1)) < eps):
            break
    
    return x_kx1


def lambda_const(f, x_k, lambda_k):
    return 0.5

def lambda_ratio(f, x_k, lambda_k):
    if (lambda_k == INF):
        return 0.5
    return lambda_k * 0.97

def lambda_quickest_descent(f, x_k, lambda_k):
    



fn = lambda x, y: x ** 2 + y ** 2

fn_answer1 = gradient_method(fn, np.array([18888., 5743.]), 0.001, lambda_const)
print(fn_answer1)
fn_answer2 = gradient_method(fn, np.array([18888., 5743.]), 0.001, lambda_ratio)
print(fn_answer2)
# ставлю


