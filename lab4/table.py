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

        # В self.v храним координаты текущей вершины
        # Выбираем вершину как свободные члены при базисных переменных
        self.v = self._rows_to_v()
    
    def prepare(self):
        "Полагается, что базис уже задан в self.rows"

        for basis_row_i, basis_var in enumerate(self.rows):

            # Разделим строку на коэффициент при базисной переменной,
            # чтобы коэффициент стал равен 1
            # print(basis_row_i, basis_var + 1)
            # print(self.table[basis_row_i, basis_var + 1])
            # print(self.table)

            if not np.isclose(self.table[basis_row_i, basis_var + 1], 0):
                self.table[basis_row_i] /= self.table[basis_row_i, basis_var + 1]
            else:
                raise NotImplementedError()
                # for row in self.table[:-1]:
                #     if np.isclose(row[basis_var + 1], 0):
                #         continue
                #     self.table[]

            for row_i in range(self.nrows): # TODO: проверить, что nrows, а не nconstr
                if row_i == basis_row_i:
                    continue

                basis_var_coef = self.table[row_i, basis_var + 1]
                
                # Вычтем из строки домноженную на коэффициент строку базисной
                # переменной, чтобы коэффициент при переменной стал равен 0
                self.table[row_i] -= self.table[basis_row_i] * basis_var_coef




    def prepare_table(self):

        # обнаружить, сколько уже базисных переменных ес

        # construct starting tableau
        
        numVar = self.n
        numArtificial = self.nrows
        numSlacArtificial = numArtificial # не сработает с неравенствами - посмотреть осторожно
        
        t1 = np.hstack(([None], [0], [0] * numVar, [0] * numArtificial))
                    
        basis = np.array([0] * numArtificial)
        
        for i in range(0, len(basis)):
            basis[i] = numVar + i
        
        A = self.A
        
        if(not ((numSlacArtificial + numVar) == len(self.A[0]))):
            B = np.identity(numArtificial)
            A = np.hstack((self.A, B))
            
        t2 = np.hstack((np.transpose([basis]), np.transpose([self.b]), A))
        
        tableau = np.vstack((t1, t2))
        
        for i in range(1, len(tableau[0]) - numArtificial):
            for j in range(1, len(tableau)):
                if(self.minmax == "MIN"):
                    tableau[0, i] -= tableau[j, i]
                else:
                    tableau[0, i] += tableau[j, i]
        
        tableau = np.array(tableau, dtype ='float')
        
        return tableau

    def solve(self, debug=False, max_iter=100):
        # None - пропущено
        # [] - решения не найдено

        # print(self.rows)
        # print(self.table)

        #print(self.task.start.shape)
        if self.type == TableType.use_start:
            if self.task.start is None:
                pass
            else:
                self.rows = []
                for i in range(len(self.task.start)):
                    if self.task.start[i] != 0:
                        self.rows.append(i)
                self.rows = np.array(self.rows)
        
        try:
            self.prepare()
        except NotImplementedError:
            return None

        # print(self.table)

        #return

        # Если указано, найдем начальную вершину через вспомогательную задачу
        if self.type == TableType.solve_supplementary:
            self.v = Table(self.task.to_supplementary(), type=TableType.default).solve(debug)
            # TODO: проверить, работает ли эта вещь
            raise NotImplementedError("Сначала проверим =)")

        # Если указано, воспользуемся начальной вершиной
        if False and self.type == TableType.use_start:

            # # записать новые базисные переменные
            # self.rows = []
            # for y in range(self.task.start.shape[0]):
            #     if self.task.start[y] != 0:
            #         self.rows.append(y)

            # # обновить свободные коэффициенты
            # self.v = self.task.start
            # for y in range(self.nconstr):
            #     if self.v[self.rows[y]] != 0:
            #         self.table[y, :] /= self.table[y, 0]
            #         self.table[y, :] *= self.v[self.rows[y]]

            print(self.v)
            print(self.rows)

            self.table[-1, 0] = np.array(self.task.f).T @ np.array(self.v)
            print(self.table)

            #self.table[0, :] = np.divide(self.table[0, :], self.)
            # TODO: привести таблицу в вид, соответствующий стартовой вершине
            #raise NotImplementedError("Нужно привести таблицу в соответствии с этой вершиной")

        #self.table[-1, 0] = np.array(self.task.f).T @ np.array(self.v)

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
        v = np.zeros(shape=(self.n))
        for x in range(self.nconstr):
            if self.rows[x] < self.n:
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
    return ["tasks/" + f for f in listdir("./tasks/") if isfile(join("./tasks/", f))]

if __name__ == "__main__":

    #for fn in ["tasks/t6.json"]:
    debug = False
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
    