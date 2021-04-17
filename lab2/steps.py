from primathlab1.alg import golden_ratio_method, fibonacci_method

### SERVICE FUNCTIONS

def check_monotonic(f, curr_x, grad, step):
    curr_f = f(*curr_x)
    next_f = f(*(curr_x - step * grad))
    return next_f <= curr_f


def reduce_until_monotonic(f, curr_x, grad, step):
    while not check_monotonic(f, curr_x, grad, step):
        step /= 2

    return step


class LambdaRatio:
    def __init__(self, step, coef):
        self._curr_step = step
        self._coef = coef

    def __call__(self, f, curr_x, grad, **kwargs):
        step = reduce_until_monotonic(f, curr_x, grad, self._curr_step)
        self._curr_step *= self._coef
        return step


def lambda_quickest_descent(f, curr_x, prev_step, grad, linear_minimizer):
    fu = lambda t : f(*(curr_x - t * grad))
    u = linear_minimizer(fu, 0., 1000., 0.0001)
    t = sum(u[-1]) / 2
    return t


### LAMBDAS

def lambda_const(step):
    return lambda f, curr_x, grad, **kwargs: reduce_until_monotonic(
        f, curr_x, grad, step
    )


def lambda_ratio(step, coef):
    return LambdaRatio(step, coef)


def lambda_quickest_descent_golden(f, curr_x, prev_step, grad, **kwargs):
    return lambda_quickest_descent(f, curr_x, prev_step, grad, golden_ratio_method)


def lambda_quickest_descent_fibonacci(f, curr_x, prev_step, grad, **kwargs):
    return lambda_quickest_descent(f, curr_x, prev_step, grad, fibonacci_method)