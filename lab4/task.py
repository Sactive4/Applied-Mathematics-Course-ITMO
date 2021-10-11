# Task solution
import json

# import numpy as np
# import scipy.optimize

from pydantic import BaseModel
from enum import Enum
from math import isclose


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
    # start: list[float]


    @staticmethod
    def load(filename):
        with open(filename) as f:
            data = json.load(f)
            return Task(**data)
    

    def to_canonical(self):
        task = self.copy()

        # Задачу максимизации приводим к задаче минимизации
        # путём умножения функции на -1

        if task.type is TaskType.max:
            task.f = [-x for x in task.f]
            task.type = TaskType.min

        # Ограничения >= приведём к ограничениям <= путём
        # умножения их на -1

        for c in task.constraints:
            if c.sign is ConstraintSign.ge:
                c.mul_in_place(-1)

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

    
    def solve_scipy(self):
        # TODO
        # self.f: минимизируемая линейная функция (столбик коэффициентов перед x)
        # self.c: 2-мерная матрица ограничений в форме <=
        # Все переменные принимаются x_j >= 0
        # return scipy.optimize.linprog(c=self.f, A_eq=self.c[:, 0:-1], b_eq=self.c[:, -1])
        
        raise NotImplementedError


if __name__ == "__main__":
    import sys
    task = Task.load(sys.argv[1])
    print("=== Исходная задача ===")
    print(task)
    canonical = task.to_canonical()
    print("=== Канонический вид ===")
    print(canonical)
