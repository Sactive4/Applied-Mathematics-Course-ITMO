import numpy as np
import scipy as sp, scipy.optimize

# см. https://habr.com/ru/post/474286/
# см. https://medium.com/@jacob.d.moore1/coding-the-simplex-algorithm-from-scratch-using-python-and-numpy-93e3813e6e70

# n - количество переменных
# m - количество уравнений/неравенств
# для всех переменных верно: x_j >= 0
# задача решается на минимум
from lab4.task import Task

n = 5
m = 3

t = np.zeros((n+1, m+1))

#        | x1 | x_2 | .. | x_(n-|  c |
#   x1   |    |    |    |    |    |
#   x2   |    |    |    |    |    |
#   ...  |    |    |    |    |    |
#   x_m  |    |    |    |    |    |
#   -------------------------------
#   K    |    |    |    |    |    |

#        | x1 | x2 | .. | x_n|  c |
#   x1   |    |    |    |    |    |
#   x2   |    |    |    |    |    |
#   ...  |    |    |    |    |    |
#   x_m  |    |    |    |    |    |
#   -------------------------------
#   K    |    |    |    |    |    |


# c = [-6, -1, -4, 5]
# r = [[3, 1, -1, 1],
#     [5, 1, 1, -1]]
# u = [4, 4]
#
# print(sp.optimize.linprog(c=c, A_eq=r, b_eq=u))

t = Task('tasks/t1.json')
print(t.solve_scipy())