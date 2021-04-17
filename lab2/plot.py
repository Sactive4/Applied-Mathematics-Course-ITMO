import matplotlib.pyplot as plt
import numpy as np


def plot_3d_with_trajectory(
    f, trajectory, x_left, x_right, y_left, y_right, resol=10, title="F(X, Y)"
):
    """Отобразить линии уровня функции двух переменных и траекторию работы алгоритма
    f - отрисовываемая функция
    trajectory - последовательность точек траектории
    x_left - левая граница оси X
    x_right - правая граница оси X
    y_left - левая граница оси Y
    y_right - правая граница оси Y
    resol - разрешение, число отрисовываемых точек на единицу длины
    title - заголовок графика
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)

    x = np.linspace(x_left, x_right, (x_right - x_left) * resol)
    y = np.linspace(y_left, y_right, (y_right - y_left) * resol)
    x, y = np.meshgrid(x, y)
    z = f(x, y)

    ax.contour3D(x, y, z)

    points_x = [point[0] for point in trajectory]
    points_y = [point[1] for point in trajectory]
    points_z = [f(*point) for point in trajectory]

    ax.scatter(points_x, points_y, points_z, c="red")

    plt.show()
