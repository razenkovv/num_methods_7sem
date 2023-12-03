import numpy as np
import matplotlib.pyplot as plt

import task as t

def plot(res, name, plot_exact=False, path="./images"):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(t.xaxis, res, color='blue', zorder=2, s=6, marker='o', label='calc')
    if plot_exact:
        xs = np.linspace(t.l, t.r, 1000)
        ax.plot(xs, t.exact(xs), color='red', zorder=1, label='exact')
    ax.set_title("Result")
    plt.legend()
    fig.savefig(path+"/"+name)

def plot_error(res, name, path="./images"):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    errors = np.abs(res - t.exact(t.xaxis))
    print("max error: ", np.max(errors))
    ax.plot(t.xaxis, errors, color='blue')
    ax.set_title("Error")
    fig.savefig(path+"/"+name)

def plot_f(name, path="./images"):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(t.xaxis, t.f(t.xaxis), color='blue')
    ax.set_title("f")
    fig.savefig(path+"/"+name)

