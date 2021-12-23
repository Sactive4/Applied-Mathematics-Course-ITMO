

import numpy as np

lmbd = 4.46
mu = 2.7
r = 1
m = 2

# S0 ------> S1 ------> S2 ------> S3 ------> .... -----> S_{m}   (всего m + 1 состояний)
#     lmbds[0]    lmbds[1]    lmbds[2]  ....        lmbds[m - 1]
#       lmbd        lmbd        lmbd                  lmbd
# S0 <------ S1 <------ S2 <------ S3 <------ .... <----- S_{m}   (всего m + 1 состояний)
#      mus[0]       mus[1]      mus[2]  ....       mus[m - 1]
#        mu         2 mu       3 mu     ....      r * mu    r * mu

lmbds = []
mus = []

for i in range(m + 1):
    if i != m:
        lmbds.append(lmbd)
    else:
        lmbds.append(0.0)
    if i != 0:
        mus.append(min(r * mu, i * mu))
    else:
        mus.append(0.0)

print(lmbds)
print(mus)

A = np.zeros(shape=(m+1, m+1))

for i in range(m + 1):
    #for j in range(m + 1):
    if (i > 0):
        A[i, i - 1] = lmbds[i - 1]
    A[i, i] = -(lmbds[i] + mus[i])
    if (i < m):
        A[i, i + 1] = mus[i + 1]

print(A)

# TODO: Теперь надо решить систему уравнений A = 0
# при этом p0 + p1 + ... + p_{m} = 1

# решим
As = [lmbds[0] / mus[1]]
for i in range(m - 1):
    As.append(As[i] * (lmbds[i + 1] / mus[i + 2]))
As = [1.0] + As

# Пусть у нас есть ответы (стационарная точка)
p = np.zeros(shape=(m + 1))

p_sum = np.sum(As, axis=0)
for i in range(len(p)):
    p[i] = As[i] / p_sum



from scipy.integrate import solve_ivp
def rhs(s, v): 
    equations = []
    for i in range(m+1):
        eq = 0.0
        for j in range(m+1):
            eq += A[i, j] * v[j]
        equations.append(eq)
    return equations

res = solve_ivp(rhs, (0, 5), [1.0] + [0.0] * (m))

import matplotlib.pyplot as plt

# нестационарное решение ( функции )
p_diff = np.zeros(shape=(m+1))
for i in range(len(res.y)):
    plt.plot(res.t, res.y[i], label="S" + str(i))
    p_diff[i] = res.y[i, -1]


print(p)
print(p_diff)

plt.legend()
plt.show()

# TODO: см. методичку страницы 7
# найти число машин, занятых рубкой, вероятности
# среднее число занятых машин
# коэффициент загрузки
# число свободных машин
# коэфф. простоя


