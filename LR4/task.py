import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import time
import os

from tdma import tdma

L_x = np.sqrt(2)
L_y = 1
h_x = 0.01
h_y = 0.01
eps_ = 1e-6

lambda_min = (np.pi / L_x) ** 2 + (np.pi / L_y) ** 2
lambda_max = 4 / h_x ** 2 + 4 / h_y ** 2
tau_ = 1 / np.sqrt(lambda_min * lambda_max)
print(f"eps: {eps_}. tau: {tau_}.")

path = "D:\\мифи\\7_семестр\\численные_методы\\лабораторные\\LR4\\data"
steps_per_save = 10


def phi1(y):
    #return 15*y*np.cos(2*y) + 3*y**2
    #return 5 - 4 * y
    #return np.sqrt(2) * y ** 2 - y ** 3
    return y**3 + np.cos(3*y)



def phi2(y):
    #return 5 * (3*y - 2) * np.cos(3 + 2*y) + 3*y**2
    #return 5 - 4 * y
    # return -np.cos(np.sqrt(2) * np.pi * y) + 1
    return -2 + y**3 + np.cos(8*np.sqrt(2) + 3*y)


def psi1(x):
    #return -10*x*np.cos(3*x)
    #return 5
    # return np.sqrt(2) * x ** 2 - x ** 3
    return -x**2 + np.cos(8*x)


def psi2(x):
    #return 5 * (6-2*x)*np.cos(3*x + 4) + 12
    #return 1
    #return -np.cos(np.sqrt(2) * np.pi * x) + 1
    return -x**2 + 1 + np.cos(8*x + 3)


def f(x, y):
    #return (-65*np.cos(3*x+2*y)*(-2*x+3*y)+6)
    #return 0
    # return x * y
    return -2 - 73 * np.cos(8 * x + 3 * y) + 6 * y


def solve(Lx=L_x, Ly=L_y, hx=h_x, hy=h_y, tau=tau_, eps=eps_):
    Nx = np.ceil((Lx / hx)).astype(int)
    Ny = np.ceil((Ly / hy)).astype(int)
    mesh = np.zeros(shape=(Ny, Nx))
    mesh_prev = np.copy(mesh)
    xaxis = np.linspace(0.0, Lx, Nx)
    yaxis = np.linspace(0.0, Ly, Ny)
    mesh[:, 0] = phi1(yaxis)
    mesh[:, -1] = phi2(yaxis)
    mesh[0, :] = psi1(xaxis)
    mesh[-1, :] = psi2(xaxis)
    mesh_prev[1:-1, 1:-1] = 0

    right1 = np.zeros(Nx - 2)
    left1 = np.zeros((Nx - 4) * 3 + 4)
    right2 = np.zeros(Ny - 2)
    left2 = np.zeros((Ny - 4) * 3 + 4)

    m = 0
    for _f in os.listdir(path):
        os.remove(os.path.join(path, _f))
    np.savetxt(f"{path}//{m}.csv", mesh, delimiter=",")
    print("start")
    t = time.time()
    curr_eps = 1
    while curr_eps > eps:
        mesh_prev = np.copy(mesh)
        m += 1
        for i in range(1, Ny - 1):
            left1[0] = 1 / tau + 2 / hx ** 2
            left1[1] = -1 / hx ** 2
            right1[0] = mesh[i, 0] / hx ** 2 + mesh[i, 1] / tau + (mesh[i, 2] - 2 * mesh[i, 1] + mesh[i, 0]) / hx ** 2 - f(xaxis[1], yaxis[i])
            left1[-1] = left1[0]
            left1[-2] = left1[1]
            right1[-1] = mesh[i, -1] / hx ** 2 + mesh[i, -2] / tau + (mesh[i, -3] - 2 * mesh[i, -2] + mesh[i, -1]) / hx ** 2 - f(xaxis[-2], yaxis[i])
            j = 2
            while j < (Nx - 4) * 3 + 2:
                left1[j:j + 3] = [-1 / hx ** 2, 1 / tau + 2 / hx ** 2, -1 / hx ** 2]
                j += 3
            for j in range(1, Nx - 3):
                right1[j] = mesh[i, j + 1] / tau + (mesh[i, j + 2] - 2 * mesh[i, j + 1] + mesh[i, j]) / hx ** 2 - f(xaxis[j + 1], yaxis[i])
            mesh[i, 1:-1] = tdma(left1, right1)

        for i in range(1, Nx - 1):
            left2[0] = 1 / tau + 2 / hy ** 2
            left2[1] = -1 / hy ** 2
            right2[0] = mesh[0, i] / hy ** 2 + mesh[1, i] / tau + (mesh[2, i] - 2 * mesh[1, i] + mesh[0, i]) / hy ** 2 - f(xaxis[i], yaxis[1])
            left2[-1] = left1[0]
            left2[-2] = left1[1]
            right2[-1] = mesh[-1, i] / hy ** 2 + mesh[-2, i] / tau + (mesh[-3, i] - 2 * mesh[-2, i] + mesh[-1, i]) / hy ** 2 - f(xaxis[i], yaxis[-2])
            j = 2
            while j < (Ny - 4) * 3 + 2:
                left2[j:j + 3] = [-1 / hy ** 2, 1 / tau + 2 / hy ** 2, -1 / hy ** 2]
                j += 3
            for j in range(1, Ny - 3):
                right2[j] = mesh[j + 1, i] / tau + (mesh[j + 2, i] - 2 * mesh[j + 1, i] + mesh[j, i]) / hy ** 2 - f(xaxis[i], yaxis[j + 1])
            mesh[1:-1, i] = tdma(left2, right2)

        curr_eps = np.max(np.abs(mesh - mesh_prev))
        if m % steps_per_save == 0:
            print(f"step: {m}, curr_eps = {curr_eps}.")
            np.savetxt(f"{path}//{m}.csv", mesh, delimiter=",")

    t = time.time() - t
    np.savetxt(f"{path}//{m}.csv", mesh, delimiter=",")
    print(f"end\nsteps: {m}. time: {t}. curr_eps = {curr_eps}.")


def plot_step(s, cmap, file, show=False):
    files = [int(os.path.splitext(f_)[0]) for f_ in os.listdir(path)]
    files.sort()
    if s == "result":
        s = files[-1]

    grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
    fig, (ax, cax) = plt.subplots(1, 2, gridspec_kw=grid_kws, figsize=(6, 6))
    x_tick_n = 5
    y_tick_n = 5
    Nx = np.ceil((L_x / h_x)).astype(int)
    Ny = np.ceil((L_y / h_y)).astype(int)

    sns.heatmap(np.genfromtxt(os.path.join(path, str(s) + ".csv"), delimiter=","), ax=ax, cbar=True, cbar_ax=cax, cmap=cmap)  # , vmin=0.0, vmax=2.0)
    ax.set_xticks(np.linspace(0, Nx, x_tick_n))
    ax.set_xticklabels(f'{c:.1f}' for c in np.linspace(0, L_x, x_tick_n))
    ax.xaxis.tick_top()
    ax.set_yticks(np.linspace(0, Ny, y_tick_n))
    ax.set_yticklabels(f'{c:.1f}' for c in np.linspace(0, L_y, y_tick_n))
    ax.set_title(f"Step: {s}")
    if show:
        plt.show()
    if not os.path.exists('images'):
        os.mkdir('images')
    fig.savefig(f'images/{file}')


def animation(param):
    files = [int(os.path.splitext(f_)[0]) for f_ in os.listdir(path)]
    files.sort()
    files = [str(f) + '.csv' for f in files]

    def init():
        pass

    def update(frame):
        # print(frame)
        ax.axis('off')
        sns.heatmap(np.genfromtxt(os.path.join(path, files[frame]), delimiter=","), ax=ax, cbar=True, cbar_ax=cax, cmap='jet')  # , vmin=0.0, vmax=2.0)
        ax.set_title(f"Step: {files[frame].split('.')[0]}")

    grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
    fig, (ax, cax) = plt.subplots(1, 2, gridspec_kw=grid_kws, figsize=(6, 6))

    ani = FuncAnimation(fig=fig, init_func=init, func=update, frames=len(files), interval=1, repeat=False)

    if param == "show":
        plt.show()
    else:
        Writer = writers['ffmpeg']
        writer = Writer(fps=25)
        if not os.path.exists('animations'):
            os.mkdir('animations')
        ani.save(f'animations/task.mp4', writer)


def plot_exact(cmap, savepath):
    def exact(x, y):
        return -x**2 + y**3 + np.cos(-8*x - 3*y)
        #return 5*(-2*x+3*y)*np.cos(3*x+2*y)+3*y**2

    acc = 1000
    x = np.linspace(0, L_x, acc)
    y = np.linspace(0, L_y, acc)
    xv, yv = np.meshgrid(x, y)
    grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
    fig, (ax, cax) = plt.subplots(1, 2, gridspec_kw=grid_kws, figsize=(6, 6))
    x_tick_n = 5
    y_tick_n = 5
    sns.heatmap(exact(xv, yv), ax=ax, cbar=True, cbar_ax=cax, cmap=cmap)  # , vmin=0.0, vmax=2.0)
    ax.set_xticks(np.linspace(0, acc, x_tick_n))
    ax.set_xticklabels(f'{c:.1f}' for c in np.linspace(0, L_x, x_tick_n))
    ax.xaxis.tick_top()
    ax.set_yticks(np.linspace(0, acc, y_tick_n))
    ax.set_yticklabels(f'{c:.1f}' for c in np.linspace(0, L_y, y_tick_n))
    ax.set_title("Exact solution.")
    if not os.path.exists('images'):
        os.mkdir('images')
    fig.savefig(f'images/{savepath}')
