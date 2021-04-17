from methods import *
import inspect
from plot import plot_3d_with_trajectory


def check_correct(x, answer, eps):
    for i in range(len(x)):
        if abs(x[i] - answer[i]) > eps:
            return False
    return True

def print_correct(trajectory, answer, eps, title):
    iter_count = len(trajectory) - 1
    final_point = trajectory[-1]

    if check_correct(final_point, answer, eps):
        print("+", title, iter_count)
    else:
        if iter_count == M:
            print("--", title, "diverges")
        else:
            print("--", title, iter_count)

def test_function(fn, x, eps, step, alpha, answer):
    print(inspect.getsource(fn))
    print_correct(gradient_method(fn, x, eps, lambda_const(step)), answer, eps, "CONST STEP")
    print_correct(gradient_method(fn, x, eps, lambda_ratio(step, alpha)), answer, eps, "CONST RATIO")
    print_correct(gradient_method(fn, x, eps, lambda_quickest_descent), answer, eps, "QUICKEST DESCENT")
    print_correct(conjugate_gradient_method(fn, x, eps), answer, eps, "CONJUGATED GRADS")
    print_correct(conjugate_direction_method(fn, x, eps), answer, eps, "CONJUGATED DIRS")
    print_correct(newton_method(fn, x, eps), answer, eps, "NEWTON")
    print("")

fn1 = lambda x, y: x ** 2 + y ** 2
# ответ 0, 0
fn2 = lambda x, y: 22 * ((x-100) ** 4) + 8 * (y ** 4)
# ответ 100, 0

# todo: существуют методы нахождения оптимальных шагов и коэф. (см. вики)
test_function(fn1, np.array([7., 5.]), 0.001, 0.5, 0.95, np.array([0., 0.]))
test_function(fn2, np.array([200., 10.]), 0.001, 2., 0.75, np.array([100., 0.]))

# quickest_descent_gradient_method(f, x0, eps)
# gradient_method(fn, x, eps, lambda_const(step)
# gradient_method(fn, x, eps, lambda_ratio(step, alpha))


# Примеры рисования графиков

#tr1 = gradient_method(fn1, np.array([7., 5.]), 0.001, lambda_const(0.5))
#plot_3d_with_trajectory(fn1, tr1, -8, 8, -8, 8, title="CONST STEP CHECKED, fn1")

tr2 = gradient_method(fn2, np.array([200., 10.]), 0.001, lambda_const(2))
plot_3d_with_trajectory(fn1, tr2, 95, 205, -50, 60, title="CONST STEP, fn2")
