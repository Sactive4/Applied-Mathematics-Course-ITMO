import numpy as np

from task import Task, solve_scipy


class Table:
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

    def __init__(self, task: Task):
        self.n = len(task.f) # число оригинальных переменных

        task = task.to_canonical()

        nvars = len(task.f)
        nconstr = len(task.constraints)

        ncols = nvars + 1
        nrows = nconstr + 1

        self.table = np.empty(shape=(nrows, ncols))

        for i, constr in enumerate(task.constraints):
            self.table[i] = [constr.b] + constr.a

        self.table[-1] = [0] + task.f

        self.table[:, 1:] *= (-1)
        
        self.columns = np.arange(nvars)
        self.rows = np.arange(nvars - nconstr, nvars)

        if task.start:
            self.v = np.array(task.start)
        else:
            self.v = np.zeros(nvars)
            self.v[nconstr:] = 1


    # def __fake_init__(self, task: Task):
    #     self.n = 2 # количество начальных переменных
    #     self.k = 3 # количество неравенств
    #     self.m = 3 # количество ограничений
    #     self.table = np.zeros(shape=(self.m + 1, self.n + self.k - self.m)) # double
    #     self.v = np.zeros(shape=(self.n + self.k))
    #     self.rows = np.empty(shape=(self.m + 1)) # int
    #     self.columns = np.empty(shape=(self.n + self.k - self.m)) # int

    #     self.table = np.array([ [4, 2, -1, 1, 0, 0], [2, 1, -2, 0, 1, 0], [5, 1, 1, 0, 0, 1], [0, -3, 1, 0, 0, 0]], dtype=np.double)
    #     self.table[:, 1:] *= (-1)
    #     self.v[self.m:] = 1
    #     self.rows = np.array([2, 3, 4])
    #     self.columns = np.array([0, 1, 2, 3, 4])

    def swap(row, column):
        """
        Меняет местами ...
        @param row: номер строки (0..m-1)
        @param column: номер столбца (0..(n+k-m-1))
        """
        ...

    def __rectangle_value(self, a, x):
        return self.table[x[0], x[1]] - (self.table[a[0], x[1]] * self.table[x[0], a[1]]) / self.table[a[0], a[1]]

    def solve(self):
        N = 100 # max iterations
        for i in range(N):
            x = self.next_step()
            if x is None:
                continue
            else:
                return x
        return []

    def next_step(self, debug=False):

        if debug:
            print(self.table)

        # индекс разрешающего столбца
        j = 1 + np.argmin(self.table[-1, 1:])
        if self.table[-1, j] >= 0:
            # минимум найден, останавливаемся
            return self.v[:(self.n)]
        
        # индекс разрешающей строки
        i = np.argmin( self.table[:-1, 0] / self.table[:-1, j])
        #print((i,j))
        #print(self.table[i, j])
        if self.table[i, j] >= 0:
            # решения нет
            return []

        new_table = np.array(self.table, copy=True)
        for x in range(self.table.shape[0]):
            for y in range(self.table.shape[1]):
                if x == i:
                    new_table[x, y] /= self.table[i, j]
                    if y != j:
                        new_table[x, y] *= (-1)
                else:
                    new_table[x, y] = self.__rectangle_value((i, j), (x, y))
        
        self.table[:] = new_table # np.array(new_table, copy=True)
        self.rows[i] = self.columns[j - 1]
        self.v = np.zeros(shape=(self.v.shape[0]))
        for x in range(len(self.rows)):
            self.v[self.rows[x]] = self.table[x, 0]

        #print(self.table)
        #print(self.table[:-1, :(self.m+1)])
        #print(np.linalg.solve(self.table[:-1, :(self.m+1)], self.table[:, 0]))
        #print(self.table)

        return None


if __name__ == "__main__":
    for fn in ["tasks/example.json", "tasks/example2.json"]:
        print("NOW: " + fn)
        task = Task.load(fn)
        t = Table(task)
        print(t.solve())
        print(solve_scipy(task).x)
    