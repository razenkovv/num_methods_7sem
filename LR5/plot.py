import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import seaborn as sns
import os

import task as t

def plot_exact(cmap, savepath, delta):
    # acc = 1000
    acc = t.Nx
    x = np.linspace(t.l, t.r, acc)
    y = np.linspace(t.l, t.r, acc)
    xv, yv = np.meshgrid(x, y)
    grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
    fig, (ax, cax) = plt.subplots(1, 2, gridspec_kw=grid_kws, figsize=(6, 6))
    x_tick_n = 5
    y_tick_n = 5
    sns.heatmap(t.exact(xv, yv, delta), ax=ax, cbar=True, cbar_ax=cax, cmap=cmap)  # , vmin=0.0, vmax=2.0)
    ax.set_xticks(np.linspace(0, acc, x_tick_n))
    ax.set_xticklabels(f'{c:.1f}' for c in np.linspace(t.l, t.r, x_tick_n))
    ax.xaxis.tick_top()
    ax.set_yticks(np.linspace(0, acc, y_tick_n))
    ax.set_yticklabels(f'{c:.1f}' for c in np.linspace(t.l, t.r, y_tick_n))
    ax.set_title("Exact solution.")
    if not os.path.exists('images'):
        os.mkdir('images')
    fig.savefig(f'images/{savepath}')


def plot_step(s, cmap, file, show=False):
    if not os.path.exists('data'):
        os.mkdir('data')
    files = [int(os.path.splitext(f_)[0]) for f_ in os.listdir("data")]
    files.sort()
    if s == "result":
        s = files[-1]

    grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
    fig, (ax, cax) = plt.subplots(1, 2, gridspec_kw=grid_kws, figsize=(6, 6))
    x_tick_n = 5
    y_tick_n = 5

    sns.heatmap(np.genfromtxt(os.path.join("data", str(s) + ".csv"), delimiter=","), ax=ax, cbar=True, cbar_ax=cax, cmap=cmap)  # , vmin=0.0, vmax=2.0)
    ax.set_xticks(np.linspace(0, t.Nx, x_tick_n))
    ax.set_xticklabels(f'{c:.1f}' for c in np.linspace(0, t.r - t.l, x_tick_n))
    ax.xaxis.tick_top()
    ax.set_yticks(np.linspace(0, t.Ny, y_tick_n))
    ax.set_yticklabels(f'{c:.1f}' for c in np.linspace(0, t.r - t.l, y_tick_n))
    ax.set_title(f"Step: {s}")
    if show:
        plt.show()
    if not os.path.exists('images'):
        os.mkdir('images')
    fig.savefig(f'images/{file}')


def animation(param="save"):
    files = [int(os.path.splitext(f_)[0]) for f_ in os.listdir("data")]
    files.sort()
    files = [str(f) + '.csv' for f in files]

    def init():
        pass

    def update(frame):
        print(frame)
        ax.axis('off')
        sns.heatmap(np.genfromtxt(os.path.join("data", files[frame]), delimiter=","), ax=ax, cbar=True, cbar_ax=cax, cmap='jet')  # , vmin=0.0, vmax=2.0)
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


def plot_relative_error(cmap, savepath, delta, calc_sol, title):
    xv, yv = np.meshgrid(t.xaxis, t.yaxis)
    grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
    fig, (ax, cax) = plt.subplots(1, 2, gridspec_kw=grid_kws, figsize=(6, 6))
    x_tick_n = 5
    y_tick_n = 5
    exact = t.exact(xv, yv, delta)
    sns.heatmap(np.abs((exact - calc_sol) / exact), ax=ax, cbar=True, cbar_ax=cax, cmap=cmap)  # , vmin=2.5, vmax=3.5)
    ax.set_xticks(np.linspace(0, t.Nx, x_tick_n))
    ax.set_xticklabels(f'{c:.1f}' for c in np.linspace(t.l, t.r, x_tick_n))
    ax.xaxis.tick_top()
    ax.set_yticks(np.linspace(0, t.Ny, y_tick_n))
    ax.set_yticklabels(f'{c:.1f}' for c in np.linspace(t.l, t.r, y_tick_n))
    ax.set_title(title)
    if not os.path.exists('images'):
        os.mkdir('images')
    fig.savefig(f'images/{savepath}')


def plot_relative_diff_my_scipy(cmap, savepath, u, u_sc):
    xv, yv = np.meshgrid(t.xaxis, t.yaxis)
    grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
    fig, (ax, cax) = plt.subplots(1, 2, gridspec_kw=grid_kws, figsize=(6, 6))
    x_tick_n = 5
    y_tick_n = 5
    sns.heatmap(np.abs((u - u_sc) / u_sc), ax=ax, cbar=True, cbar_ax=cax, cmap=cmap)  # , vmin=0.0, vmax=2.0)
    ax.set_xticks(np.linspace(0, t.Nx, x_tick_n))
    ax.set_xticklabels(f'{c:.1f}' for c in np.linspace(t.l, t.r, x_tick_n))
    ax.xaxis.tick_top()
    ax.set_yticks(np.linspace(0, t.Ny, y_tick_n))
    ax.set_yticklabels(f'{c:.1f}' for c in np.linspace(t.l, t.r, y_tick_n))
    ax.set_title("rel_diff_my_scipy.")
    if not os.path.exists('images'):
        os.mkdir('images')
    fig.savefig(f'images/{savepath}')

def plot_smth(cmap, savepath, u, title):
    xv, yv = np.meshgrid(t.xaxis, t.yaxis)
    grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
    fig, (ax, cax) = plt.subplots(1, 2, gridspec_kw=grid_kws, figsize=(6, 6))
    x_tick_n = 5
    y_tick_n = 5
    sns.heatmap(u, ax=ax, cbar=True, cbar_ax=cax, cmap=cmap)  # , vmin=0.0, vmax=2.0)
    ax.set_xticks(np.linspace(0, t.Nx, x_tick_n))
    ax.set_xticklabels(f'{c:.1f}' for c in np.linspace(t.l, t.r, x_tick_n))
    ax.xaxis.tick_top()
    ax.set_yticks(np.linspace(0, t.Ny, y_tick_n))
    ax.set_yticklabels(f'{c:.1f}' for c in np.linspace(t.l, t.r, y_tick_n))
    ax.set_title(title)
    if not os.path.exists('images'):
        os.mkdir('images')
    fig.savefig(f'images/{savepath}')


def plot_error(cmap, savepath, delta, calc_sol, title):
    xv, yv = np.meshgrid(t.xaxis, t.yaxis)
    grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
    fig, (ax, cax) = plt.subplots(1, 2, gridspec_kw=grid_kws, figsize=(6, 6))
    x_tick_n = 5
    y_tick_n = 5
    exact = t.exact(xv, yv, delta)
    sns.heatmap(np.abs(exact - calc_sol), ax=ax, cbar=True, cbar_ax=cax, cmap=cmap)  # , vmin=2.5, vmax=3.5)
    ax.set_xticks(np.linspace(0, t.Nx, x_tick_n))
    ax.set_xticklabels(f'{c:.1f}' for c in np.linspace(t.l, t.r, x_tick_n))
    ax.xaxis.tick_top()
    ax.set_yticks(np.linspace(0, t.Ny, y_tick_n))
    ax.set_yticklabels(f'{c:.1f}' for c in np.linspace(t.l, t.r, y_tick_n))
    ax.set_title(title)
    if not os.path.exists('images'):
        os.mkdir('images')
    fig.savefig(f'images/{savepath}')

def plot_diff_my_scipy(cmap, savepath, u, u_sc):
        xv, yv = np.meshgrid(t.xaxis, t.yaxis)
        grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
        fig, (ax, cax) = plt.subplots(1, 2, gridspec_kw=grid_kws, figsize=(6, 6))
        x_tick_n = 5
        y_tick_n = 5
        sns.heatmap(np.abs(u - u_sc), ax=ax, cbar=True, cbar_ax=cax, cmap=cmap)  # , vmin=0.0, vmax=2.0)
        ax.set_xticks(np.linspace(0, t.Nx, x_tick_n))
        ax.set_xticklabels(f'{c:.1f}' for c in np.linspace(t.l, t.r, x_tick_n))
        ax.xaxis.tick_top()
        ax.set_yticks(np.linspace(0, t.Ny, y_tick_n))
        ax.set_yticklabels(f'{c:.1f}' for c in np.linspace(t.l, t.r, y_tick_n))
        ax.set_title("diff_my_scipy.")
        if not os.path.exists('images'):
            os.mkdir('images')
        fig.savefig(f'images/{savepath}')
