import numpy as np
from pydantic import BaseModel
from enum import Enum

from task import Task, solve_scipy

class TableType(str, Enum):
    default = "default"
    solve_supplementary = "solve_supplementary"
    use_start = "use_start"

class Table:

    # LEGACY ===== LEGACY ===== LEGACY ===== LEGACY ===== LEGACY ===== LEGACY
    # n начальных переменных
    # m ограничений (среди них k неравенств, это для новых переменных нужно)
    # то

    # таблица
    #      |  b  |   1  |   2   |   3  |   4   |   5  |   6  |   7
    #   5  |  b1     3      1       1       1      1      0      1
    #   6  |  b2                                         
    #   7  |  b3                                              
    #   K  |  0     с_1 * x_1  c_2 * x_2 ...

    # x_0 (json: start и f): x_0 * f

    # строк будет (m + 1), последняя для K
    # столбцов будет (1 + (n + k - m)), 
    # тогда адресация будет прямая, например, [0][1], для пересечения переменной 1 и 4

    # будет отдельный массив rows, columns, который будет сопоставлять каждому строке/столбцу номер переменной
    # LEGACY ===== LEGACY ===== LEGACY ===== LEGACY ===== LEGACY ===== LEGACY

    def __init__(self, task: Task, type : TableType = TableType.default):

        self.type = type
        self.n = len(task.f) # число исходных переменных

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
        self.table[-1] = [0] + self.task.f

        #self.task.start = np.array(self.task.start)

        # В self.rows мы храним индексы, соответствующие номеру базисной переменной
        # По умолчанию, принимаем последние self.nconstr переменных
        # если мы явно не вводили новые переменные, то присвоим просто следующие виртуальные номера

        self.rows = np.arange(self.nvars - self.nconstr, self.nvars)
        for i in range(self.nconstr):
            self.rows[i] = self.nvars - self.nconstr + i

        #print(self.rows)

        # В self.v храним координаты текущей вершины
        # Выбираем вершину как свободные члены при базисных переменных
        self.v = np.zeros(shape=(self.nvars))
        self.v = self._rows_to_v()
        #print(self.v)
    
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
        
        # for row_i in range(self.nconstr):
        #     if self.table[row_i, 0] < 0:
        #         coef = self.table[row_i, 0] / self.table[-1, 0]
        #         self.table[row_i] -= self.table[-1] * coef
        
        # if debug:
        #     print(self.table)
        #     print("Rows ", self.rows)

        # assert np.all(self.table[:-1, 0] >= 0), "Свободные коэффициенты стали отрицательными"

    def solve(self, debug=False, max_iter=100):
        # None - пропущено
        # [] - решения не найдено

        # print(self.rows)
        # print(self.table)

        #print(self.task.start.shape)

        #print(self.v)

        # Выбираем начальную вершину
        if self.type == TableType.use_start and not(self.task.start is None):
            self.rows = []
            for y in range(len(self.task.start)):
                if self.task.start[y] != 0:
                    self.rows.append(y)
            self.rows = np.array(self.rows)
            self.v = self._rows_to_v()
            assert len(self.rows) == self.nconstr, "Начальная вершина некорректная"
        elif self.type != TableType.solve_supplementary:
            pass
            #print(self.v)
            #self.v = Table(self.task.to_supplementary(), type=TableType.solve_supplementary).solve(debug)
            #print("my initial point is ", self.v)

        #print(self.v)

        #print("motherfucker v ", self.v)
        self.v = np.array(self.v)

        # Инициализируем таблицу для начальной вершины
        #print("Initial rows ", self.rows)

        
    
        #print(self.rows)
        # print(self.type)
        # print(self.v)

        # Приводим матрицу к базису
        if self.type != TableType.solve_supplementary:
            try:
                self.prepare(debug)
            except Exception:
                return None

        # print(self.table)

        #return

        # # Если указано, найдем начальную вершину через вспомогательную задачу
        # if self.type == TableType.solve_supplementary:
        #     self.v = Table(self.task.to_supplementary(), type=TableType.default).solve(debug)
        #     # TODO: проверить, работает ли эта вещь
        #     raise NotImplementedError("Сначала проверим =)")

        # # Если указано, воспользуемся начальной вершиной
        # if False and self.type == TableType.use_start:

        #     # # записать новые базисные переменные
        #     # self.rows = []
        #     # for y in range(self.task.start.shape[0]):
        #     #     if self.task.start[y] != 0:
        #     #         self.rows.append(y)

        #     # # обновить свободные коэффициенты
        #     # self.v = self.task.start
        #     # for y in range(self.nconstr):
        #     #     if self.v[self.rows[y]] != 0:
        #     #         self.table[y, :] /= self.table[y, 0]
        #     #         self.table[y, :] *= self.v[self.rows[y]]

        #     print(self.v)
        #     print(self.rows)

        #     self.table[-1, 0] = np.array(self.task.f).T @ np.array(self.v)
        #     print(self.table)

        #     #self.table[0, :] = np.divide(self.table[0, :], self.)
        #     # TODO: привести таблицу в вид, соответствующий стартовой вершине
        #     #raise NotImplementedError("Нужно привести таблицу в соответствии с этой вершиной")

        # #self.table[-1, 0] = np.array(self.task.f).T @ np.array(self.v)

        for i in range(max_iter):
            x = self.next_step(debug)
            if x is None:
                continue
            else:
                if len(x) == 0:
                    break
                #print("Answer found.")
                if debug:
                    print("Answer found.")
                return x[:(self.n)]
        if debug:
           print("No solution / Out of iterations")
        return []

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
        #self.v[self.rows[i]] = 0
        self.rows[i] = j - 1

        # обновим текущую вершину
        self.v = self._rows_to_v()
        #self.v[self.rows[i]] = self._rows_to_v()

        # этот сигнал, что нужно идти дальше
        return None

    def _rows_to_v(self):
        #print("blyat ", self.nvars, " ", self.nconstr, " ", self.rows)
        v = np.zeros(shape=(self.v.shape[0]))
        #print(self.nconstr)
        for x in range(self.nconstr):
            #print(self.rows[x])
            if self.rows[x] < self.nvars:
                v[self.rows[x]] = self.table[x, 0]
        return v

def solve(fn, debug=False):
    task = Task.load(fn)
    if task.answer is not None:
        if not task.check_correct(task.answer):
            print("Preset answer for this task does not satisfy contstraints.")
        if task.answer == []:
            print("Preset answer says there is no solution.")
    x = Table(task, type=TableType.use_start).solve(debug)
    if not task.check_correct(x):
        print("Warning! Answer is wrong")
    return x, task.answer
    return Table(task.to_phase1(), type=TableType.default).solve(debug), task.answer

def get_fns():
    from os import listdir
    from os.path import isfile, join
    return sorted(
        ["tasks/" + f for f in listdir("./tasks/") if isfile(join("./tasks/", f))]
    )

if __name__ == "__main__":

    #for fn in ["tasks/t6.json"]:
    debug = False
    #for fn in ['tasks/t4.json']: #, 'tasks/t6.json', 'tasks/t7.json']:
    for fn in get_fns():
        #print("======>>>>> TASK: " + fn)
        
        # TODO: Красивый вывод? (отступы, может быть)
        # TODO: Добавить во все файлы ответы, их можно посчитать через онлайн сервисы
        # или см. https://mattmolter.medium.com/creating-a-linear-program-solver-by-implementing-the-simplex-method-in-python-with-numpy-22f4cbb9b6db
        # и см. https://github.com/mmolteratx/Simplex = SinglePhase, там тупо вбить и всё будет круто
        # TODO: написать отчет, в нем отразить всю теорию, этапы работы алгоритма, ответить на вопросы в нем

        #print(">>>", fn)
        solution, answer = solve(fn, debug)
        #print(solution, answer)
        if solution is None:
            print("... SKIPPED ", fn)
        elif (answer is None) or len(answer) == 0:
            print("? ", solution, " Add answer to evaluate. ", fn)
        elif np.allclose(solution, answer):
            print("+ ", solution, " OK ", fn)
        else:
            print("- ", solution, " WRONG ", fn)
    