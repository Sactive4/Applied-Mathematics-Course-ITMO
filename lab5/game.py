import json

import numpy as np

from pydantic import BaseModel
from enum import Enum
from math import isclose

from lab4.table import Table
from lab4.task import Task, Constraint, ConstraintSign, TaskType

def find_min_indices(m):
    min1 = np.max(m[0])
    s = []
    for i in range(len(m)):
        t = np.max(m[i])
        if min1 >= t:
            if min1 != t:
                s = []
                min1 = t
            s.append(i)
    return min1, s

class Game(BaseModel):
    instruction: str = ""
    type: str
    matrix: list
    answer_clean: list = None
    answer_simplex: list = None

    @staticmethod
    def load(filename):
        with open(filename) as f:
            data = json.load(f)
            return Game(**data)

    def to_task(self):

        n = len(self.matrix)
        m = len(self.matrix[0])

        f = [1.0] * m
        constraints = []

        for i in range(n):
            if self.type == 'loss':
                constraints.append(Constraint.parse_obj({'a': self.matrix[i], 'sign': ConstraintSign.ge, 'b' : 1.0}))
            elif self.type == 'win':
                constraints.append(Constraint.parse_obj({'a': -1.0 * self.matrix[i], 'sign': ConstraintSign.ge, 'b' : 1.0}))
            else:
                assert False, "Unexpected game type."

        return Task.parse_obj({'type':TaskType.min, 'f' : f, 'constraints' : constraints, 'answer' : self.answer_simplex[0]})

    def get_clean_strategy(self):
        # найдем осторожную стратегию первого игрока
        #m = self.table[:-1, 1:(self.n+1)]
        m = np.array(self.matrix)
        min1, s1 = find_min_indices(m)
        min2, s2 = find_min_indices(m.T)

        r = []
        if min1 == min2:
            for i in s1:
                for j in s2:
                    if m[i, j] == min1:
                        r.append((i, j))

        return r

    def solve_clean(self):
        return self.get_clean_strategy()

    def solve_simplex(self, debug=False):
        table = Table(self.to_task())
        table.solve(debug)
        x = table.get_solution()
        y = table.get_dual_solution()
        f = table.get_solution_f()
        g = table.get_dual_solution_f()
        v = 1.0 / f
        P = v * y
        Q = v * x
        return P, Q, v

    def expectation(self, s1, s2):
        m = np.array(self.matrix)
        r = 0.0
        for i in range(len(m)):
            for j in range(len(m[0])):
                r += m[i, j] * s1[i] * s2[j]
        return r