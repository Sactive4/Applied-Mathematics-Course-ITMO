from methods import *


def check_correct(x, answer, eps):
    for i in range(len(x)):
        if abs(x[i] - answer[i]) > eps:
            return False
    return True

def print_correct(x, answer, eps, title):
    if check_correct(x[0], answer, eps):
        print("+ " + title + " " + str(x[1]))
    else:
        if x[1] == M:
            print("-- " + title + " diverges")
        else:
            print("-- " + title + " " + str(x[1]))

def test_function(fn, x, eps, step, alpha, answer):
    print_correct(gradient_method(fn, x, eps, lambda_const(step)), answer, eps, "CONST STEP")
    print_correct(gradient_method(fn, x, eps, lambda_const_checked(step)), answer, eps, "CONST STEP CHECKED")
    print_correct(gradient_method(fn, x, eps, lambda_ratio(step, alpha)), answer, eps, "CONST RATIO")
    print_correct(gradient_method(fn, x, eps, LambdaRatioChecked(step, alpha)), answer, eps, "CONST RATIO CHECKED")
    print_correct(gradient_method(fn, x, eps, lambda_quickest_descent), answer, eps, "QUICKEST DESCENT")
    print_correct(conjugate_gradient_method(fn, x, eps), answer, eps, "CONJUGATED GRADS")
    print_correct(conjugate_direction_method(fn, x, eps), answer, eps, "CONJUGATED DIRS")
    print_correct(newton_method(fn, x, eps), answer, eps, "NEWTON")

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




