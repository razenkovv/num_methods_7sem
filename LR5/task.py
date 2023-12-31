import numpy as np
from scipy.sparse import csr_array, eye
from scipy.sparse.linalg import spsolve

# -1 * laplasian(u) = f

l = 0.0
r = 1.0

Nx = int(100)
Ny = int(100)

hx = (r - l) / Nx
hy = (r - l) / Ny

xaxis = np.linspace(l+hx/2, r-hx/2, Nx)
yaxis = np.linspace(l+hy/2, r-hy/2, Ny)

steps_per_save = 1000

def f(x, y):
    #return 0*x + 0*y
    #return -4
    #return -8*x
    #return 2 - 6*y + 73 * np.cos(-8*x - 3*y)
    return 58.0*np.sin(7.0*x - 3.0*y) + 80.0*np.cos(2.0*x + 6.0*y)

def phi_L(y):
    #return -1
    #return 0
    #return -y**2
    #return -8 * np.sin(-3*y)
    return -7.0 * np.cos(7.0 * l - 3.0 * y) + 4.0 * np.sin(2.0 * l + 6.0 * y)

def phi_R(y):
    #return 1
    #return 2
    #return 3 + y**2
    #return -2 + 8*np.sin(-8 - 3*y)
    return +7.0 * np.cos(7.0 * r - 3.0 * y) - 4.0 * np.sin(2.0 * r + 6.0 * y)

def phi_B(x):
    #return -1
    #return 0
    #return -x**3
    #return -3 * np.sin(-8*x)
    return +3.0 * np.cos(7.0 * x - 3.0 * l) + 12.0 * np.sin(2.0 * x + 6.0 * l)

def phi_T(x):
    #return 1
    #return 2
    #return x**3 + 2*x
    #return 3 + 3 * np.sin(-8*x - 3)
    return -3.0 * np.cos(7.0 * x - 3.0 * r) - 12.0 * np.sin(2.0 * x + 6.0 * r)

def exact(x, y, delta):
    #return x + y
    #return x**2 + y**2 - delta
    #return x**3 + x * y**2 - delta
    #return -x**2 + y**3 + np.cos(-8*x - 3*y) - delta
    return np.sin(7.0 * x - 3.0 * y) + 2.0 * np.cos(2.0 * x + 6.0 * y) - delta


def to2D(n):
    return int(n / Nx), int(n % Nx)

def to1D(i, j):
    return int(Nx * i + j)

def solve_scipy():
    print("matrix creation start")
    matrix, right_part = create_matrix()
    print("matrix creation end")
    u = spsolve(matrix, right_part)
    np.savetxt(f"scipy_data//1.csv", u.reshape(Ny, Nx), delimiter=",")
    return u.reshape(Ny, Nx)

def solve(eps=1e-3):
    print("matrix creation start")
    matrix, right_part = create_matrix()
    print("SUM: ", np.sum(right_part))
    right_part -= np.mean(right_part)
    print("SUM after: ", np.sum(right_part))
    print("matrix creation end")

    u = np.ones(Nx * Ny)
    curr_eps = 1
    m = int(0)
    while curr_eps > eps:
        rk = matrix @ u - right_part
        curr_eps = np.sqrt(rk @ rk)
        if m % steps_per_save == 0:
            print(f"step: {m}, curr_eps = {curr_eps}.")
            np.savetxt(f"data//{m}.csv", u.reshape(Ny, Nx), delimiter=",")
        tmp = matrix @ rk
        tau = tmp @ rk / (tmp @ tmp)
        u = u - tau * rk
        m += 1

    print(f"step: {m-1}, curr_eps = {curr_eps}.")
    np.savetxt(f"data//{m-1}.csv", u.reshape(Ny, Nx), delimiter=",")

    return u.reshape(Ny, Nx)


def create_matrix():
    data = np.zeros(5 * (Nx - 2) * (Ny - 2) + 4 * (2 * (Nx - 2) + 2 * (Ny - 2)) + 3 * 4)
    i_ind = np.zeros_like(data)
    j_ind = np.zeros_like(data)

    right_part = np.zeros(Nx * Ny)

    counter = 0

    n = to1D(0, 0)
    # print("n: ", n)
    i_ind[counter:counter+3] = np.repeat(n, 3)
    j_ind[counter:counter+3] = np.array([n, to1D(0, 1), to1D(1, 0)])
    data[counter:counter+3] = np.array([hy/hx+hx/hy, -hy/hx, -hx/hy])
    # data[counter:counter+3] = np.array([4, -1, -1])  # change
    right_part[0] = f(xaxis[0], yaxis[0])*hx*hy + phi_L(yaxis[0]) * hy + phi_B(xaxis[0]) * hx
    counter += 3

    for i in range(1, Nx - 1):
        n = to1D(0, i)
        # print("n: ", n)
        i_ind[counter:counter+4] = np.repeat(n, 4)
        j_ind[counter:counter+4] = np.array([to1D(0, i - 1), n, to1D(0, i + 1), to1D(1, i)])
        data[counter:counter+4] = np.array([-hy/hx, 2*hy/hx+hx/hy, -hy/hx, -hx/hy])
        # data[counter:counter+4] = np.array([-1, 4, -1, -1])  # change
        right_part[i] = f(xaxis[i], yaxis[0])*hx*hy + phi_B(xaxis[i]) * hx
        counter += 4

    n = to1D(0, Nx - 1)
    # print("n: ", n)
    i_ind[counter:counter+3] = np.repeat(n, 3)
    j_ind[counter:counter+3] = np.array([to1D(0, Nx-2), n, to1D(1, Nx-1)])
    data[counter:counter+3] = np.array([-hy/hx, hy/hx+hx/hy, -hx/hy])
    # data[counter:counter+3] = np.array([-1, 4, -1])  # change
    right_part[Nx-1] = f(xaxis[Nx-1], yaxis[0])*hx*hy + phi_B(xaxis[Nx-1]) * hx + phi_R(yaxis[0]) * hy
    counter += 3

    counter_right_part = Nx

    for i in range(1, Ny - 1):
        for j in range(0, Nx):
            n = to1D(i, j)
            #print("n: ", n)
            if j == 0:
                i_ind[counter:counter+4] = np.repeat(n, 4)
                j_ind[counter:counter+4] = np.array([to1D(i-1, j), n, to1D(i, j+1), to1D(i+1, j)])
                data[counter:counter+4] = np.array([-hx/hy, hy/hx+2*hx/hy, -hy/hx, -hx/hy])
                # data[counter:counter+4] = np.array([-1, 4, -1, -1]) # change
                right_part[counter_right_part] = f(xaxis[j], yaxis[i])*hx*hy + phi_L(yaxis[i]) * hy
                counter_right_part += 1
                counter += 4
            elif j == Nx - 1:
                i_ind[counter:counter+4] = np.repeat(n, 4)
                j_ind[counter:counter+4] = np.array([to1D(i-1, j), to1D(i, j-1), n, to1D(i+1, j)])
                data[counter:counter+4] = np.array([-hx/hy, -hy/hx, hy/hx+2*hx/hy, -hx/hy])
                # data[counter:counter+4] = np.array([-1, -1, 4, -1]) # change
                right_part[counter_right_part] = f(xaxis[j], yaxis[i])*hx*hy + phi_R(yaxis[i]) * hy
                counter_right_part += 1
                counter += 4
            else:
                i_ind[counter:counter+5] = np.repeat(n, 5)
                j_ind[counter:counter+5] = np.array([to1D(i-1, j), to1D(i, j-1), n, to1D(i, j+1), to1D(i+1, j)])
                data[counter:counter+5] = np.array([-hx/hy, -hy/hx, 2*hy/hx+2*hx/hy, -hy/hx, -hx/hy])
                # data[counter:counter+5] = np.array([-1, -1, 4, -1, -1]) # change
                right_part[counter_right_part] = f(xaxis[j], yaxis[i]) * hx * hy
                counter_right_part += 1
                counter += 5

    n = to1D(Ny-1, 0)
    #print("n: ", n)
    i_ind[counter:counter + 3] = np.repeat(n, 3)
    j_ind[counter:counter + 3] = np.array([to1D(Ny-2, 0), n, to1D(Ny-1, 1)])
    data[counter:counter+3] = np.array([-hx/hy, hy/hx+hx/hy, -hy/hx])
    # data[counter:counter+3] = np.array([-1, 4, -1])  # change
    right_part[counter_right_part] = f(xaxis[0], yaxis[Ny-1])*hy*hx + phi_L(yaxis[Ny-1]) * hy + phi_T(xaxis[0]) * hx
    counter_right_part += 1
    counter += 3

    for i in range(1, Nx - 1):
        n = to1D(Ny-1, i)
        #print("n: ", n)
        i_ind[counter:counter + 4] = np.repeat(n, 4)
        j_ind[counter:counter + 4] = np.array([to1D(Ny-2, i), to1D(Ny-1, i-1), n, to1D(Ny-1, i+1)])
        data[counter:counter+4] = np.array([-hx/hy, -hy/hx, 2*hy/hx+hx/hy, -hy/hx])
        # data[counter:counter+4] = np.array([-1, -1, 4, -1])  # change
        right_part[counter_right_part] = f(xaxis[i], yaxis[Ny-1])*hy*hx + phi_T(xaxis[i]) * hx
        counter_right_part += 1
        counter += 4

    n = to1D(Ny - 1, Nx - 1)
    # print("N_end: ", n)
    i_ind[counter:counter + 3] = np.repeat(n, 3)
    j_ind[counter:counter + 3] = np.array([to1D(Ny-2, Nx-1), to1D(Ny-1, Nx-2), n])
    data[counter:counter+3] = np.array([-hx/hy, -hy/hx, hy/hx+hx/hy])
    # data[counter:counter + 3] = np.array([-1, -1, 4])  # change
    right_part[counter_right_part] = f(xaxis[Nx-1], yaxis[Ny-1])*hx*hy + phi_T(xaxis[Nx-1]) * hx + phi_R(yaxis[Ny-1]) * hy
    counter_right_part += 1
    counter += 3

    return csr_array((data, (i_ind, j_ind)), shape=(Nx*Ny, Nx*Ny)), right_part

