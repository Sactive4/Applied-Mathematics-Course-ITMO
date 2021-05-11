

# ПУНКТ 1
# LU-композиция
# решение системы уравнений (LU + Гаусса)
# обратная матрица

def lu_decomposition(A):
    """ Найти LU-разложение матрицы A
    A - матрица, хранящаяся в разреженном виде
    :returns
    (L, U) - пара матриц L и U в разреженном виде, где A = LU
    L - нижняя треугольная матрица
    U - верхняя треугольная матрица
    :throws
    A -
    а) квадратная
    б) невырожденная <=> detA != 0
    в) миноры не вырождены <=> every detAij != 0
    """
    # пример кода
    # https://en.wikipedia.org/wiki/LU_decomposition
    # конкретные формулы и реализация
    # https://www.quantstart.com/articles/LU-Decomposition-in-Python-and-NumPy/
    return (L, U)

def trivial_system_solution(A, b, upper=True):
    """ Найти тривиальное решение уравнения Ax=B
    A - треугольная матрица, хранящаяся в разреженном виде
    b - вектор в правой части
    upper - если A верхнетреугольная True, нижнетреугольная - False
    throws
    """
    # решение методом Гаусса (тривиальное)
    return x

def system_solution(A, b):
    """ Найти решение системы Ax=b
    A - матрица, хранящаяся в разреженном виде
    b - вектор в правой части
    """
    L, U = lu_decomposition(A)
    y = trivial_system_solution(L, b, False)
    x = trivial_system_solution(U, y, True)
    return x

def system_solution_matrix(A, B):
    """ Найти решение системы AX=B
    A - матрица, хранящаяся в разреженном виде
    B - матрица, хранящаяся в разреженном виде
    returns:
    X - решение-матрица в разреженном виде
    """
    X = []
    for i in range(B.количество_столбцов):
        X[i-ый столбец] = system_solution(A, B[i-ый столбец])
    return X

def inverse_matrix(A):
    """ Найти обратную для матрицы A
    A - матрица, хранящаяся в разреженном виде
    :returns
    A_inverse - обратная матрица в разреженном виде
    """
    E = единичная матрица размера, что и A
    return system_solution_matrix(A, E)

### Обратие внимание!
# Совместность системы: https://ru.wikipedia.org/wiki/Теорема Кронекера — Капелли

# ПУНКТ 2
# Тестирование программы из п. 1

# to be continued...