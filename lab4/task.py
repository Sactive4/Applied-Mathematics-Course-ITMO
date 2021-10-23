import json

import numpy as np

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
    start: list[float] = None
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

        return task
    

    def check_correct(self, answer):
        if answer is None:
            return False

        for c in self.constraints:
            v = np.array(c.a)[:len(answer)] @ np.array(answer)
            if c.sign == ConstraintSign.le:
                if v > c.b:
                    print(v, " ", c.sign, " ", c.b)
                    return False
            if c.sign == ConstraintSign.eq:
                if abs(v - c.b) > 0.00001:
                    print(v, " ", c.sign, " ", c.b)
                    return False
            if c.sign == ConstraintSign.ge:
                if v < c.b:
                    print(v, " ", c.sign, " ", c.b)
                    return False
        return True

    def to_supplementary(self):
        """
        Возвращает задачу, вспомогательную для исходной
        обратите внимание, что значение целевой функции
        заполняется в Table.solve(run_as_supplementary=True)
        
        Исходная задача должна быть в каноническом виде
        """

        assert self.type is TaskType.max

        task = self.copy(deep=True)
        task.f = np.array([0.0] * len(task.f) + [0.0] * len(task.constraints))

        for i in range(len(task.constraints)):
            task.constraints[i].a += [0.0] * len(task.constraints)
            task.constraints[i].a[len(self.f) + i] = 1.0
            task.f[:] += task.constraints[i].a

        task.start = [0.0] * (len(task.f) - len(task.constraints)) + [1.0] * len(task.constraints)

        for i in range(len(task.constraints)):
            task.f[-(i + 1)] = 0.0

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


if __name__ == "__main__":
    import sys
    task = Task.load(sys.argv[1])
    print("=== Исходная задача ===")
    print(task)
    canonical = task.to_canonical()
    print("=== Канонический вид ===")
    print(canonical)
