import numpy as np

lmbd = 4.46 # Интенсивность поступления
mu = 2.7    # Интенсивность рубки
r = 1       # Число машин
m = 2       # Максимальная длина очереди

# S0 ------> S1 ------> S2 ------> S3 ------> .... -----> S_{m}   (всего m + r + 1 состояний)
#     lmbds[0]    lmbds[1]    lmbds[2]  ....        lmbds[m - 1]
#       lmbd        lmbd        lmbd                  lmbd
# S0 <------ S1 <------ S2 <------ S3 <------ .... <----- S_{m}   (всего m + r + 1 состояний)
#      mus[0]       mus[1]      mus[2]  ....       mus[m - 1]
#        mu         2 mu       3 mu     ....      r * mu    r * mu

states_count = m + r + 1

lmbds = [lmbd] * (states_count - 1) + [0.0]
mus = [min(r, i) * mu for i in range(states_count)]

print("Интенсивности перехода из i-го состояния в (i+1)-ое:", lmbds, sep="\n")
print("Интенсивности перехода из i-го состояния в (i-1)-ое:", mus, sep="\n")

A = np.zeros(shape=(states_count, states_count))

for i in range(states_count):
    if (i > 0):
        A[i, i - 1] = lmbds[i - 1]
    A[i, i] = -(lmbds[i] + mus[i])
    if (i < states_count - 1):
        A[i, i + 1] = mus[i + 1]

print("Система уравнений:", A, sep="\n")

# Теперь надо решить систему уравнений A = 0
# при этом p0 + p1 + ... + p_{m} = 1

# решим
As = [lmbds[0] / mus[1]]
for i in range(states_count - 2):
    As.append(As[i] * (lmbds[i + 1] / mus[i + 2]))
As = [1.0] + As

# Пусть у нас есть ответы (стационарная точка)
p = np.zeros(shape=states_count)

p_sum = np.sum(As, axis=0)
for i in range(len(p)):
    p[i] = As[i] / p_sum

print("Стационарное решение:", p)

print()

# Рассчитаем стационарные характеристики

p = list(p)

busy_machines_count_prob = p[:r] + [sum(p[r:])]
average_busy_count = sum(
    count * prob for count, prob in enumerate(busy_machines_count_prob)
)
busy_coef = average_busy_count / r

free_machines_count_prob = reversed(busy_machines_count_prob)
average_free_count = sum(
    count * prob for count, prob in enumerate(free_machines_count_prob)
)
free_coef = average_free_count / r

queue_length_prob = [sum(p[:r + 1])] + p[r + 1:]
average_queue_length = sum(
    count * prob for count, prob in enumerate(queue_length_prob)
)

print("Вероятность, что k машин заняты рубкой:")
for i, prob in enumerate(busy_machines_count_prob):
    print(f"{i} — {prob:.4f}")

print("Вероятность, что все машины заняты и l брёвен находится в очереди:")
for i, prob in enumerate(queue_length_prob):
    print(f"{i} — {prob:.4f}")

print(f"Среднее число занятых рубкой машин: {average_busy_count:.4f}")
print(f"Среднее число свободных от рубки машин: {average_free_count:.4f}")
print(f"Коэффициент загрузки машин: {busy_coef:.4f}")
print(f"Коэффициент простоя машин: {free_coef:.4f}")
print(f"Среднее число брёвен в очереди: {average_queue_length:.4f}")

print()

# Перейдём к нестационарной системе

import matplotlib.pyplot as plt
fig, (states_ax, coefs_ax) = plt.subplots(2)
fig.tight_layout(h_pad=2)
states_ax.set_title("Вероятности состояний")
states_ax.set_xlabel("Время")
states_ax.set_ylabel("Вероятность")
coefs_ax.set_title("Коэффициенты")
coefs_ax.set_xlabel("Время")
coefs_ax.set_ylabel("Коэффициент")

from scipy.integrate import solve_ivp
def rhs(s, v): 
    equations = []
    for i in range(states_count):
        eq = 0.0
        for j in range(states_count):
            eq += A[i, j] * v[j]
        equations.append(eq)
    return equations

res = solve_ivp(rhs, (0, 5), [1.0] + [0.0] * (states_count - 1))

# нестационарное решение ( функции )
p_diff = np.zeros(shape=states_count)
for i in range(len(res.y)):
    states_ax.plot(res.t, res.y[i], label="S" + str(i))
    p_diff[i] = res.y[i, -1]

print("Нестационарное решение:", p_diff)

busy_coef_dyn = (
    sum(k * res.y[k] for k in range(r))
    + r * sum(res.y[k] for k in range(r, states_count))
) / r

free_coef_dyn = sum(k * res.y[r - k] for k in range(1, r + 1)) / r

coefs_ax.plot(res.t, busy_coef_dyn, label="Загрузки")
coefs_ax.plot(res.t, free_coef_dyn, label="Простоя")

states_ax.legend()
coefs_ax.legend()
plt.show()
