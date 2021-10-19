# Task solution
import json

# import numpy as np
import scipy.optimize
import numpy as np

from pydantic import BaseModel
from enum import Enum
from math import isclose
from typing import Optional


class TaskType(str, Enum):
    min = "min"
    max = "max"


class ConstraintSign(str, Enum):
    le = "<="
    ge = ">="
    eq = "="


    def reverse(self):
        if self is ConstraintSign.le:
            return ConstraintSign.ge
        elif self is ConstraintSign.ge:
            return ConstraintSign.le
        return self


class Constraint(BaseModel):
    a: list[float]
    sign: ConstraintSign
    b: float


    def mul_in_place(self, k):
        self.a = [x * k for x in self.a]
        self.b *= k

        if k < 0:
            self.sign = self.sign.reverse()


class Task(BaseModel):
    type: TaskType
    f: list[float]
    constraints: list[Constraint]
    start: Optional[list[float]]
    answer: list[float] = None


    @staticmethod
    def load(filename):
        with open(filename) as f:
            data = json.load(f)
            return Task(**data)
    

    def to_min_in_place(self):
        if self.type is TaskType.min:
            return

        self.f = [-x for x in self.f]
        self.type = TaskType.min

    def to_max_in_place(self):
        if self.type is TaskType.max:
            return

        self.f = [-x for x in self.f]
        self.type = TaskType.max
    

    def remove_ge_in_place(self):
        for c in self.constraints:
            if c.sign is ConstraintSign.ge:
                c.mul_in_place(-1)
    

    def to_canonical(self):
        task = self.copy(deep=True)

        # Задачу минимизации приводим к задаче максимизации
        # путём умножения функции на -1

        task.to_max_in_place()

        # Ограничения >= приведём к ограничениям <= путём
        # умножения их на -1

        task.remove_ge_in_place()

        # Неравенства приведём к равенствам путём введения
        # новых переменных

        for c in task.constraints:
            if c.sign is ConstraintSign.eq:
                continue

            task.f.append(0.)
            c.a.append(1.)

            for c_ in task.constraints:
                if c_ is not c:
                    c_.a.append(0.)
            
            c.sign = ConstraintSign.eq
        
        # Сделаем правые части равенств положительными путём
        # умножения равенств на -1

        for c in task.constraints:
            if c.b < 0:
                c.mul_in_place(-1)
        
        # TODO: остальные шаги приведения в канонический вид (?)
        return task
    

    def to_supplementary(self):
        # TODO: проверить, работает ли правильно
        task = self.copy(deep=True)
        task.f = [0.0] * len(task.f) + [1.0] * len(task.constraints)

        for i in range(len(task.constraints)):
            task.constraints[i].a += [0.0] * len(task.constraints)
            task.constraints[i].a[len(self.f) + i] = 1.0

        return task


    @staticmethod
    def _pretty_summands(coefs):
        return " + ".join(
            f"{c}x{i}"
            for i, c in enumerate(coefs)
            if not isclose(c, 0.)
        )
    

    def __str__(self):
        lines = []
        fn_line = "f(x) = " + self._pretty_summands(self.f)
        lines.append(fn_line)

        for c in self.constraints:
            line = self._pretty_summands(c.a) + " " + c.sign + " " + str(c.b)
            lines.append(line)
        
        return "\n".join(lines)


"""
УДАЛЕНА
мы больше не используем ее, поскольку она ошибается
в интернете написано, что она так действительно может делать,
а подключать более стабильные версии мне было лень - там сложно
"""
def solve_scipy(task: Task, debug=False):
    task = task.copy(deep=True)
    task.to_min_in_place()
    task.remove_ge_in_place()

    a_eq = np.empty(shape=(0,len(task.f)), dtype=np.float64)
    b_eq = np.empty(shape=(0,), dtype=np.float64)
    a_leq = np.empty(shape=(0,len(task.f)), dtype=np.float64)
    b_leq = np.empty(shape=(0,), dtype=np.float64)

    if debug:
        print("Solving Numpy:")
        print(task.f)
        print(a_eq)
        print(b_eq)
        print(a_leq)
        print(b_leq)

    for c in task.constraints:
        if c.sign is ConstraintSign.eq:
            a_eq = np.append(a_eq, np.array([c.a]), axis=0)
            b_eq = np.append(b_eq, np.array([c.b]), axis=0)
        elif c.sign is ConstraintSign.le:
            a_leq = np.concatenate((a_leq, c.a))
            b_leq = np.concatenate((b_leq, c.b))
        else:
            raise ValueError(f"Unexpected constraint sign {c.sign}")
    
    if len(a_eq) == 0:
        a_eq = None
        b_eq = None

    if len(a_leq) == 0:
        a_leq = None
        b_leq = None

    if debug:
        print("Solving Numpy:")
        print(task.f)
        print(a_eq)
        print(b_eq)
        print(a_leq)
        print(b_leq)

    return scipy.optimize.linprog(task.f, A_eq=a_eq, b_eq=b_eq, A_ub=a_leq, b_ub=b_leq, bounds=(0, None))


if __name__ == "__main__":
    import sys
    task = Task.load(sys.argv[1])
    print("=== Исходная задача ===")
    print(task)
    canonical = task.to_canonical()
    print("=== Канонический вид ===")
    print(canonical)
