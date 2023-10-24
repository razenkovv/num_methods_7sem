import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tdma import tdma

L_x = np.sqrt(2)
L_y = 1
h_x = 0.05
h_y = 0.05
eps_ = 1e-6

lambda_min = (np.pi / L_x) ** 2 + (np.pi / L_y) ** 2
lambda_max = 4 / h_x ** 2 + 4 / h_y ** 2
tau_ = 1 / np.sqrt(lambda_min * lambda_max)
print("tau: ", tau_)


def phi1(y):
    return 1
    return np.sqrt(2) * y ** 2 - y ** 3


def phi2(y):
    return 1
    return -np.cos(np.sqrt(2) * np.pi * y) + 1


def psi1(x):
    return 5
    return np.sqrt(2) * x ** 2 - x ** 3


def psi2(x):
    return 1
    return -np.cos(np.sqrt(2) * np.pi * x) + 1


def f(x, y):
    return 0
    return x * y


def solve(Lx=L_x, Ly=L_y, hx=h_x, hy=h_y, tau=tau_, eps=eps_):
    Nx = np.ceil((Lx / hx)).astype(int)
    Ny = np.ceil((Ly / hy)).astype(int)
    mesh = np.ones(shape=(Ny, Nx))
    xaxis = np.linspace(0.0, Lx, Nx)
    yaxis = np.linspace(0.0, Ly, Ny)
    mesh[:, 0] = phi1(xaxis)
    mesh[:, -1] = phi2(yaxis)
    mesh[0, :] = psi1(xaxis)
    mesh[-1, :] = psi2(xaxis)
    mesh_prev = np.copy(mesh)
    mesh_prev[1:-1, 1:-1] = 0

    right1 = np.zeros(Nx - 2); left1 = np.zeros((Nx - 4) * 3 + 4)
    right2 = np.zeros(Ny - 2); left2 = np.zeros((Ny - 4) * 3 + 4)

    grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
    fig, (ax, cax) = plt.subplots(1, 2, gridspec_kw=grid_kws, figsize=(6, 6))
    plt.ion()
    # for m in range(100):
    # print(m)
    m = 0
    while np.max(np.abs(mesh - mesh_prev)) > eps:
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

        if m % 1 == 0:
            print(m)
            sns.heatmap(mesh, ax=ax, cbar=True, cbar_ax=cax, cmap='jet', xticklabels=[0, Lx])#, vmin=0.0, vmax=1.0)
            ax.set_xticks(np.linspace(0, Nx, 5))
            ax.set_xticklabels(f'{c:.1f}' for c in np.linspace(0, Lx, 5))
            ax.xaxis.tick_top()
            ax.set_yticks(np.linspace(0, Ny, 5))
            ax.set_yticklabels(f'{c:.1f}' for c in np.linspace(0, Ly, 5))
            plt.show()
            plt.pause(0.01)
            plt.cla()
    print("m: ", m)
    print("end")
    plt.ioff()
    plt.close()

    grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
    fig, (ax, cax) = plt.subplots(1, 2, gridspec_kw=grid_kws, figsize=(6, 6))
    sns.heatmap(mesh, ax=ax, cbar=True, cbar_ax=cax, cmap='jet')#, vmin=0.0, vmax=2.0)
    ax.set_xticks(np.linspace(0, Nx, 5))
    ax.set_xticklabels(f'{c:.1f}' for c in np.linspace(0, Lx, 5))
    ax.xaxis.tick_top()
    ax.set_yticks(np.linspace(0, Ny, 5))
    ax.set_yticklabels(f'{c:.1f}' for c in np.linspace(0, Ly, 5))
    plt.show()
