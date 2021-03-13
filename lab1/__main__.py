import inspect
import sys
from argparse import ArgumentParser
from math import log, sin

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
    return sin(x) - log(x ** 2) - 1


parser = ArgumentParser(
    description="Найти локальный минимум функции несколькими способами",
)
parser.add_argument("-l", type=float, default=0.1, help="Левая граница интервала")
parser.add_argument("-r", type=float, default=7.0, help="Правая граница интервала")
parser.add_argument("-e", "--eps", type=float, default=0.001, help="Точность")

args = parser.parse_args()

if not args.l < args.r:
    print("Левая граница должна быть меньше правой", file=sys.stderr)
    sys.exit(1)

print(
    "Исследуемая функция:",
    inspect.getsource(f).split("\n")[2].lstrip()[7:],
)

print(f"Исследуемый интервал: ({args.l}, {args.r})")
print(f"Точность: {args.eps}")

print()

for algo in minimizers:
    fn_counted = CallCounter(f)
    res, iter_count = algo(fn_counted, args.l, args.r, args.eps)
    print(
        f"Метод: {algo.__name__}",
        f"Результат: {res:.3f}",
        f"Вызовов функции: {fn_counted.get_count()}",
        f"Итераций: {iter_count}",
        sep="\n",
        end="\n\n",
    )
