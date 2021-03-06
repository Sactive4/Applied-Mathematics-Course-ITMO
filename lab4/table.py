import numpy as np
from enum import Enum

from lab4.task import Task, TaskType


class TableType(str, Enum):
    default = "default"
    solve_supplementary = "solve_supplementary"
    use_start = "use_start"


class Table:

    # n начальных переменных
    # m ограничений (среди них k неравенств, это для новых переменных нужно)
    # то

    # таблица
    #      |  b  |   1  |   2   |   3  |   4   |   5  |   6  |   7
    #   5  |  b1     3      1       1      1       1      0      0
    #   6  |  b2                                   0      1      0
    #   7  |  b3                                   0      0      1   
    #   F  |  v0    с_1    c_2     c_3    c_4      0      0      0

    # строк будет (m + 1), последняя для K
    # столбцов будет (1 + (n + k - m)), 
    # тогда адресация будет прямая, например, [0][1], для пересечения переменной 1 и 4

    # будет отдельный массив rows, который будет сопоставлять каждой строке номер переменной

    def __init__(self, task: Task, type : TableType = TableType.default):

        self.type = type
        self.n = len(task.f) # число исходных переменных

        self.tasktype_original = task.type
        self.task = task.to_canonical()
        self.nvars = len(self.task.f) # число всех переменных (свободные + базисные)
        self.nconstr = len(self.task.constraints) # число ограничений
        self.ncols = self.nvars + 1
        self.nrows = self.nconstr + 1

        self.table = np.empty(shape=(self.nrows, self.ncols))

        # Заполним таблицу ограничениями
        for i, constr in enumerate(self.task.constraints):
            self.table[i] = [constr.b] + constr.a

        # Заполним целевую функцию
        self.table[-1, 1:] = self.task.f

        # Пока проинициализируем пустые массивы для базисных переменных и вершины
        self.v = np.zeros(self.nvars)
        self.rows = np.zeros(self.nconstr)


    def prepare(self, debug=True):
        "Полагается, что базис уже задан в self.rows"

        if debug:
                print(self.table)
                print("Rows ", self.rows)

        for basis_row_i, basis_var in enumerate(self.rows):

            for row_i in range(basis_row_i, self.nconstr):
                if not np.isclose(self.table[row_i, basis_var + 1], 0):
                    break
            else:
                raise ValueError(
                    f"Could not find a row with non-zero x{basis_var}"
                )
            
            if row_i != basis_row_i:
                self.table[row_i], self.table[basis_row_i] = \
                    self.table[basis_row_i].copy(), self.table[row_i].copy()

            self.table[basis_row_i] /= self.table[basis_row_i, basis_var + 1]

            for row_i in range(self.nrows): # TODO: проверить, что nrows, а не nconstr
                if row_i == basis_row_i:
                    continue

                basis_var_coef = self.table[row_i, basis_var + 1]
                
                # Вычтем из строки домноженную на коэффициент строку базисной
                # переменной, чтобы коэффициент при переменной стал равен 0
                self.table[row_i] -= self.table[basis_row_i] * basis_var_coef

            if debug:
                print(self.table)
                print("Rows ", self.rows)
    

    def solve(self, debug=False, max_iter=100, run_as_supplementary=False):
        # None - пропущено
        # [] - решения не найдено

        # Выбираем начальную вершину
        if self.type == TableType.use_start and not(self.task.start is None):
            # если указана стартовая вершина или есть указание решать
            self.rows = self._v_to_rows(self.task.start)
            self.v = self._rows_to_v()
            assert len(self.rows) == self.nconstr, "Начальная вершина некорректная"
        elif self.type != TableType.solve_supplementary:
            # решаем вспомогательную задачу, если еще не решаем
            task_sup = self.task.to_supplementary()
            table_sup = Table(task_sup, type=TableType.use_start)
            table_sup.table[-1, 0] = np.sum(table_sup.table[:-1, 0])

            # найдем начальную вершину и выберем базис из вспомогательной задачи
            solution_sup = table_sup.solve(debug, max_iter = 5, run_as_supplementary=True)
            self.rows = np.copy(table_sup.rows)
            assert task_sup.check_correct(solution_sup), "Начальная вершина некорректная"
            self.task.start = solution_sup[:self.n]
        
        elif self.type == TableType.default:
            # В self.rows мы храним индексы, соответствующие номеру базисной переменной
            # По умолчанию, принимаем последние self.nconstr переменных
            # если мы явно не вводили новые переменные, то присвоим просто следующие виртуальные номера

            self.rows = np.arange(self.nvars - self.nconstr, self.nvars)
            for i in range(self.nconstr):
                self.rows[i] = self.nvars - self.nconstr + i

        # Если мы сейчас решаем вспомогательную задачу,
        # нужно заполнить значение целевой функции
        # как сумму базисных переменных
        if not run_as_supplementary:
            self.table[-1, 0] = self.v @ self.table[-1, 1:]

        # Приводим матрицу к базису self.rows
        if self.type != TableType.solve_supplementary:
            try:
                self.prepare(debug)
            except Exception:
                return None

        # Обновим текущую начальную вершину
        self.v = self._rows_to_v()

        for i in range(max_iter):
            x = self.next_step(debug)
            if x is None:
                continue
            else:
                if len(x) == 0:
                    break
                if debug:
                    print("Answer found.")
                return x[:(self.n)]
        if debug:
           print("No solution / Out of iterations")
        return []


    def get_solution(self):
        return np.abs(self.v[:(self.n)])

    def get_dual_solution(self):
        if self.tasktype_original == TaskType.min:
            return -1.0 * self.table[-1][self.n+1:]
        else:
            return self.table[-1][self.n+1:]

    def get_solution_f(self):
        if self.tasktype_original == TaskType.min:
            return -1.0 * (self.get_solution() @ self.task.f[:(self.n)])
        else:
            return self.get_solution() @ self.task.f[:(self.n)]

    def get_dual_solution_f(self):
        r = 0.0
        s = self.get_dual_solution()
        for i, constr in enumerate(self.task.constraints):
            r += constr.b * s[i]
        return r

    def next_step(self, debug=False):

        # Таблица вычисляется для поиска максимума
        eps = 0.000001

        if debug:
            print("==================")
            print("Next step. Table:")
            print(self.table)
            print("Point:", self.v)
            print("Rows:", self.rows)

        # проверим на оптимальность
        if np.all(self.table[-1, 1:] <= eps):
            return self.v

        # начинаем очередную оптимизацию
        # выбираем разрешающий столбец
        j = 1 + np.argmax(self.table[-1, 1:])

        # выбираем разрешающую строку
        # для тех элементов, что > 0 находим значения
        i_values = np.divide(self.table[:-1, 0], self.table[:-1, j], out=np.zeros_like(self.table[:-1, 0]) + np.inf, where=(self.table[:-1, j]>=eps))
        i = i_values.argmin()

        # проверим, если решения нет
        if self.table[i, j] == np.inf:
            return []

        if debug:
            print("Pivot Index: ", i, ", ", j)
            print("Pivot: ", self.table[i, j])

        # поделим разрешающую строку на опорный элемент
        pivot = self.table[i, j]
        self.table[i, :] /= pivot

        # по правилу прямоугольника пересчитаем таблицу
        for x in range(self.table.shape[0]): 
            if x != i:
                self.table[x, :] -= (self.table[x, j] / self.table[i, j]) * self.table[i, :] 

        # перезапишем новую базисную переменную
        self.rows[i] = j - 1

        # обновим текущую вершину
        self.v = self._rows_to_v()

        # этот сигнал, что нужно идти дальше
        return None


    def _rows_to_v(self):
        """ по выбранным базисным переменным self.rows 
            восстановить текущую вершину
        """
        v = np.zeros(shape=(self.v.shape[0]))
        for x in range(self.nconstr):
            if self.rows[x] < self.nvars:
                v[self.rows[x]] = self.table[x, 0]
        return v


    def _v_to_rows(self, v):
        """ по стартовой вершине определить базисные
            переменные
        """
        rows = []
        for y in range(len(v)):
            if v[y] != 0:
                rows.append(y)
        return np.array(rows)


def solve(fn, debug=False, use_start=False):
    task = Task.load(fn)
    if task.answer is not None:
        if not task.check_correct(task.answer):
            print("Preset answer for this task does not satisfy contstraints.")
        if task.answer == []:
            print("Preset answer says there is no solution.")
   
    if use_start:
        x = Table(task, type=TableType.use_start).solve(debug)
    else:
        x = Table(task).solve(debug)

    value1 = x @ task.f
    value2 = np.array(task.answer) @ task.f
    if not task.check_correct(x):
        print("Warning! Answer is wrong")
    return x, task.answer, value1, value2


def get_fns():
    from os import listdir
    from os.path import isfile, join
    return sorted(
        ["tasks/" + f for f in listdir("./tasks/") if isfile(join("./tasks/", f))]
    )


if __name__ == "__main__":

    debug = False
    for use_start in [True, False]:
        if use_start:
            print("Now using initial feasible solutions, if available.")
        else:
            print("Now solving with supplementary task.")

        for fn in get_fns():

            solution, answer, value1, value2 = solve(fn, debug,use_start=use_start)

            if solution is None:
                print("... SKIPPED ", fn)
            elif (answer is None) or len(answer) == 0:
                print("? ", solution, " Add answer to evaluate. ", fn)
            elif np.allclose(solution, answer) or abs(value1 - value2) < 0.00001:
                print("+ ", solution, " OK ", fn)
            else:
                print("- ", solution, " WRONG ", fn)
    