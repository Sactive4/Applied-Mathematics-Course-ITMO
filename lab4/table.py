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

    def __init__(self, task: Task, Supplementary=False):
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

        #self.table[:, 1:] *= (-1)
        #self.table[-1, :] *= (-1)
        
        self.columns = np.arange(nvars)
        self.rows = np.arange(nvars - nconstr, nvars)

        # debug params
        self.supplementary = Supplementary
        self.nvars = len(task.f)
        self.nconstr = len(task.constraints)
        self.ncols = self.nvars + 1
        self.nrows = self.nconstr + 1


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

    # def get_initial_point(self):
        

    # def swap(row, column):
    #     """
    #     Меняет местами ...
    #     @param row: номер строки (0..m-1)
    #     @param column: номер столбца (0..(n+k-m-1))
    #     """
    #     ...

    def __rectangle_value(self, a, x):
        return self.table[x[0], x[1]] - (self.table[a[0], x[1]] * self.table[x[0], a[1]]) / self.table[a[0], a[1]]

    def solve(self, debug=False):

        if self.supplementary or True: # TODO: NOW WE CHOOSE LAST VARIABLES AS BASIS
            self.v = np.zeros(self.nvars)
            self.v[-self.nconstr:] = 1
            #self.v = self.table[]
            if debug:
                print("Solving supplementary task.")
        else:
            if debug:
                print("Solving main task.")
            if task.start:
                self.v = np.array(task.start)
            else:
                self.v = Table(task.to_supplementary(), Supplementary=True).solve(debug=True)
        
        if debug:
            print("Initial point is: ")
            print(self.v)


        N = 6 # max iterations
        for i in range(N):
            x = self.next_step(debug)
            if x is None:
                continue
            else:
                if len(x) == 0:
                    print("no solution =(")
                return x
        print("no solution =(")
        return []

    def next_step(self, debug=False):

        eps = 0.0000001

        if debug:
            print("Next step. Table:")
            print(self.table)
            print("Point:")
            print(self.v)
            print("Rows:")
            print(self.rows)

        # индекс разрешающего столбца

        #bbb = (np.divide(self.table[:-1, 0], self.table[:-1, j], out=np.zeros_like(self.table[:-1, 0]) - 777, where=(self.table[:-1, j]>=eps)))
        #valid_idx = np.where(self.table[-1, 1:] in )[0]
        #j = valid_idx[ccc[valid_idx].argmin()]


        # TODO: здесь надо сделать поиск разрешающего столбца
        # ТОЛЬКО ПО СВОБОДНЫМ ПЕРЕМЕННЫМ!!!
        print([i not in self.rows for i in range(len(self.table.shape[1]))])
        self.table[-1, [i not in self.rows for i in range(len(self.table.shape[1]))]]

        j = 1 + np.argmin(self.table[-1, 1:])
        #j = 1 + np.argmin(self.table[-1, 1:])
        if self.table[-1, j] >= -eps:
            if debug:
                print("Result found.")
                print(self.v)
            # минимум найден, останавливаемся
            return self.v[:(self.n)]
        
        ccc = (np.divide(self.table[:-1, 0], self.table[:-1, j], out=np.zeros_like(self.table[:-1, 0]) - 777, where=(self.table[:-1, j]>=eps)))

        if debug:
            print("ccc: ", ccc)
        # индекс разрешающей строки
        #i = np.argmin( self.table[:-1, 0] / self.table[:-1, j])
        #i = np.argmin(ccc)
        #print((i,j))
        #print(self.table[i, j])
        
        #print (-0.0 / 1.0 > eps)

        #valid_idx = np.where(ccc == ccc)[0]
        #print(ccc)
        #print(np.where(ccc > eps)[0])
        valid_idx = np.where(ccc > -eps)[0]
        if (len(valid_idx) == 0):
            if debug:
                print("Result found.")
                print(self.v)
            # минимум найден, останавливаемся
            return self.v[:(self.n)]

        i = valid_idx[ccc[valid_idx].argmin()]

        
        
        #if self.table[i, j] <= 0:
        #    # решения нет
        #    return []

        if debug:
            print("Разрешающий элемент: ", i, ", ", j)
        new_table = np.array(self.table, copy=True)
        for x in range(self.table.shape[0]):
            for y in range(self.table.shape[1]):
                if x == i:
                    new_table[x, y] /= self.table[i, j]
                    #if y != j:
                    #    new_table[x, y] *= (-1)
                else:
                    new_table[x, y] = self.__rectangle_value((i, j), (x, y))
        
        self.table[:] = new_table # np.array(new_table, copy=True)
        self.rows[i] = self.columns[j - 1]

        #print(self.v)

        self.v = np.zeros(shape=(self.v.shape[0]))
        for x in range(len(self.rows)):
            self.v[self.rows[x]] = self.table[x, 0]

        #print(self.table)
        #print(self.table[:-1, :(self.m+1)])
        #print(np.linalg.solve(self.table[:-1, :(self.m+1)], self.table[:, 0]))
        #print(self.table)

        return None

def solve(fn):
    print("NOW: " + fn)
    task = Task.load(fn)
    t = Table(task)
    print(t.solve())
    print(solve_scipy(task).x)

if __name__ == "__main__":

    for fn in ["tasks/t1.json"]:
    #for fn in ["tasks/example.json", "tasks/example2.json", "tasks/t1.json", "tasks/t2.json", "tasks/t3.json", "tasks/t4.json", "tasks/t5.json", "tasks/t6.json", "tasks/t7.json"]:
        print("NOW: " + fn)
        task = Task.load(fn)
        t = Table(task)

        s = t.solve(True)
        nps = solve_scipy(task, False).x

        print("***** SOLUTIONS *****")
        print("OURS: ", s)
        print("NUMPY:", nps)
        print(np.allclose(s, nps))
    