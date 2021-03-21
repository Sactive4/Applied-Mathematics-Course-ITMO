import inspect
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from math import log, sin

import matplotlib.pyplot as plt

from alg import minimizers


class CallCounter:
    """Обёртка для подсчёта числа вызовов функции"""

    def __init__(self, fn):
        self._fn = fn
        self._count = 0

    def __call__(self, *args, **kwargs):
        self._count += 1
        return self._fn(*args, **kwargs)

    def get_count(self):
        return self._count

    def reset(self):
        self._count = 0


def f(x):
    """Исходная функция"""
    #return sin(x)
    return sin(x) - log(x ** 2) - 1

def f1(x):
    """Исходная функция"""
    return sin(x)


parser = ArgumentParser(
    description="Найти локальный минимум функции несколькими способами",
)
parser.add_argument("-l", type=float, default=0.1, help="Левая граница интервала")
parser.add_argument("-r", type=float, default=7.0, help="Правая граница интервала")
parser.add_argument("-e", "--eps", type=float, default=0.001, help="Точность")
parser.add_argument(
    "--plot",
    default=False,
    action=BooleanOptionalAction,
    help="Нарисовать графики длин интервалов",
)

args = parser.parse_args()

def analysis(l, r, eps, f):

    if not l < r:
        print("Левая граница должна быть меньше правой", file=sys.stderr)
        sys.exit(1)

    print(
        "Исследуемая функция:",
        inspect.getsource(f).split("\n")[2].lstrip()[7:],
    )

    print(f"Исследуемый интервал: ({l}, {r})")
    print(f"Точность: {eps}")

    print()

    for algo in minimizers:
        fn_counted = CallCounter(f)
        intervals = algo(fn_counted, l, r, eps)
        res = sum(intervals[-1]) / 2.0
        iter_count = len(intervals)
        print(
            f"Метод: {algo.__name__}",
            f"Результат: {res:.3f}",
            f"Вызовов функции: {fn_counted.get_count()}",
            f"Итераций: {iter_count}",
            sep="\n",
            end="\n\n",
        )
        if args.plot:
            lengths = [abs(b - a) for a, b in intervals]
            plt.plot(range(iter_count), lengths, ".-", label=algo.__name__)
            plt.plot()

    if args.plot:
        plt.legend()
        plt.gca().xaxis.get_major_locator().set_params(integer=True)
        plt.title("Изменение длин интервалов в процессе работы алгоритмов")
        plt.xlabel("Номер итерации")
        plt.ylabel("Длина интервала")
        plt.show()

# analysis(args.l, args.r, args.eps, f)

analysis(2.15, 7.15, 0.001, f1)