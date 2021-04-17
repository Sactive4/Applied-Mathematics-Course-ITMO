import numpy as np


from methods import (
    gradient_method_const,
    gradient_method_quick_golden,
    gradient_method_quick_fibonacci,
    conjugate_gradient_method,
    conjugate_direction_method,
    newton_method,
    M,
)
from steps import (
    lambda_const,
    lambda_ratio,
    lambda_quickest_descent_fibonacci,
    lambda_quickest_descent_golden,
)
import inspect
from plot import plot_3d_with_trajectory


def check_correct(x, answer, eps):
    for i in range(len(x)):
        if abs(x[i] - answer[i]) > eps:
            return False
    return True


def print_correct(trajectory, answer, eps, title):
    if trajectory:
        iter_count = len(trajectory) - 1

        if iter_count < M:
            final_point = trajectory[-1]

            if check_correct(final_point, answer, eps):
                print("+", title, iter_count)
            else:
                print("--", title, iter_count)
            
            return
        
    print("--", title, "diverges")


def test(fn, x, eps, step, alpha, answer):
    print(inspect.getsource(fn))

    print_correct(gradient_method_const(fn, x, eps, step), answer, eps, "DESCENT const step")
    print_correct(gradient_method_const(fn, x, eps, step, alpha), answer, eps, "DESCENT ratio")
    print_correct(gradient_method_quick_golden(fn, x, eps), answer, eps, "QUICK DESCENT golden")
    print_correct(gradient_method_quick_fibonacci(fn, x, eps), answer, eps, "QUICK DESCENT fibonacci")
    print_correct(conjugate_gradient_method(fn, x, eps), answer, eps, "CONJUGATED GRADS")
    print_correct(conjugate_direction_method(fn, x, eps), answer, eps, "CONJUGATED DIRS")
    print_correct(newton_method(fn, x, eps), answer, eps, "NEWTON")

    print("")



# Примеры рисования графиков

def plot_function(fn, trajectory, title):
    x_min = min((point[0] for point in trajectory)) - 1
    y_min = min((point[1] for point in trajectory)) - 1
    x_max = max((point[0] for point in trajectory)) + 1
    y_max = max((point[1] for point in trajectory)) + 1
    plot_3d_with_trajectory(fn, trajectory, x_min, x_max, y_min, y_max, title=title)

def test_plot_function(fn, x, eps, step, alpha):
    plot_function(fn, gradient_method_const(fn, x, eps, step), "DESCENT const step")
    plot_function(fn, gradient_method_const(fn, x, eps, step, alpha), "DESCENT const ratio")
    plot_function(fn, gradient_method_quick_golden(fn, x, eps), "QUICK DESCENT golden")
    plot_function(fn, gradient_method_quick_fibonacci(fn, x, eps), "QUICK DESCENT fibonacci")
    plot_function(fn, conjugate_gradient_method(fn, x, eps), "CONJUGATE GRAD")
    plot_function(fn, conjugate_direction_method(fn, x, eps), "CONJUGATE DIRS")
    plot_function(fn, newton_method(fn, x, eps), "NEWTON")


def test_and_plot(fn, x, eps, step, alpha, answer):
    test(fn, x, eps, step, alpha, answer)
    test_plot_function(fn, x, eps, step, alpha)
    
fn1 = lambda x, y: x ** 2 + y ** 2
# ответ 0, 0
fn2 = lambda x, y: 22 * ((x-100) ** 4) + 8 * (y ** 4)
# ответ 100, 0

fn3 = lambda x, y: x ** 4 + y ** 4 + 2 * x * x * y * y - 4 * x + 3
# ответ 1, 1

# test_function(fn1, np.array([200000.0, 200000.]), 0.001, 100., 0.95, np.array([0., 0.]))
# test_function(fn1, np.array([1000., 1000.]), 0.001, 0.5, 0.95, np.array([0., 0.]))
# test_function(fn1, np.array([100., 100.]), 0.001, 0.5, 0.95, np.array([0., 0.]))
#test_function(fn1, np.array([7., 5.]), 0.001, 0.2, 0.5, np.array([0., 0.]))

#test_and_plot(fn1, np.array([7., 5.]), 0.001, 0.2, 0.5, np.array([0., 0.]))

#test_function(fn3, np.array([100., 100.]), 0.001, 3, 0.95, np.array([1., 0.]))
#test_function(fn3, np.array([100., 100.]), 0.001, 0.5, 0.95, np.array([1., 0.]))

#test_plot_function(fn1, np.array([7., 5.]), 0.001, 0.5, 0.95)

#test(lambda x,y: x*x+y, np.array([70., 50.]), 0.001, 5., 0.95, np.array([0., 0.]))
test(fn1, np.array([70., 50.]), 0.001, 5., 0.95, np.array([0., 0.]))
test_and_plot(fn3, np.array([100., 100.]), 0.001, 3., 0.95, np.array([1., 0.]))


# tr1 =  gradient_method_const(fn1, np.array([7., 5.]), 0.001, 0.5)
# tr1 = gradient_method_const(fn1, np.array([7., 5.]), 0.001, 0.5, 0.95)
# plot_3d_with_trajectory(fn1, tr1, -8, 8, -8, 8, title="CONST STEP CHECKED, fn1")

# tr2 = gradient_method(fn2, np.array([200., 10.]), 0.001, lambda_const(2))
# plot_3d_with_trajectory(fn1, tr2, 95, 205, -50, 60, title="CONST STEP, fn2")
