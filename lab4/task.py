# Task solution
import json

import numpy as np
import scipy.optimize

class Task:

    f = np.array([])
    c = np.array([])

    '''
    Инициализация в канонической форме
    param f: минимизируемая линейная функция (столбик коэффициентов перед x)
    param c: 2-мерная матрица ограничений в форме <=
    Все переменные принимаются x_j >= 0
    '''
    def __init__(self, f, c):
        self.f = np.array(f)
        self.c = np.array(c)

    def __init__(self, filename):
        f = open(filename)
        d = json.load(f)
        self.f = np.array(d['f'])
        self.c = d['c']
        f.close()

    def solve_scipy(self):
        return scipy.optimize.linprog(c=self.c, A_eq=self.c[:, 0:-1], b_eq=self.c[:, -1])
