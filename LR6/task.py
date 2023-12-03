import numpy as np

from tdma import tdma

l = 0.0
r = 1.0
h = 0.01
left_value = 0.0
right_value = 0.0

hs = np.ones(np.ceil((r - l) / h).astype(int)) * h
N = len(hs)
xaxis = np.array([np.sum(hs[:i]) for i in range(N+1)])

# u'' = f(x)

def f(x):
    #return -np.cos(x) - np.sin(x)
    #return x**5 + 3*x**4 - 55*x**8

    # if x < (l+r)/3 - h or x > (l+r)/3 + h:
    #     return 10
    # else:
    #     return 10 - 150

    lft = 0.355
    rgt = 0.365
    return 10 - 300*np.heaviside(x-lft, 1) * np.heaviside(rgt-x, 1)


def exact(x):
    #return -0.381773*x + np.sin(x) + np.cos(x) - 1
    #return -0.611111*x*(x**9 - 0.038961*x**6 - 0.163636*x**5 - 0.797403)
    return False

def create_matrix():
    lft = np.zeros((N-3)*3 + 4)
    rgt = np.zeros(N-1)
    lft[0] = 1 / hs[0] + 1 / hs[1]
    lft[1] = -1 / hs[1]
    lft[-1] = 1 / hs[-1] + 1 / hs[-2]
    lft[-2] = -1 / hs[-1]
    rgt[0] = -1*(f((xaxis[0]+xaxis[1])/2) * ((xaxis[0]+xaxis[1])/2-xaxis[0])/(xaxis[1]-xaxis[0]) * hs[0] +
             f((xaxis[1]+xaxis[2])/2) * (xaxis[2]-(xaxis[1]+xaxis[2])/2) / (xaxis[2] - xaxis[1]) * hs[1] + 1/hs[0]*left_value)
    rgt[-1] = -1*(f((xaxis[-2]+xaxis[-3])/2) * ((xaxis[-2]+xaxis[-3])/2-xaxis[-3])/(xaxis[-2]-xaxis[-3]) * hs[-2] +
              f((xaxis[-2]+xaxis[-1])/2) * (xaxis[-1]-(xaxis[-1]+xaxis[-2])/2) / (xaxis[-1] - xaxis[-2]) * hs[-1] + 1/hs[-1]*right_value)

    j = int(1)
    i = int(2)
    while i < (N-3)*3 + 2:
        lft[i:i+3] = [-1/hs[j], 1/hs[j]+1/hs[j+1], -1/hs[j+1]]
        i += 3
        j += 1

    i = int(1)
    while i < N-2:
        rgt[i] = -1*(f((xaxis[i]+xaxis[i+1])/2) * ((xaxis[i]+xaxis[i+1])/2-xaxis[i])/(xaxis[i+1]-xaxis[i]) * hs[i] +
                       f((xaxis[i+1]+xaxis[i+2])/2) * (xaxis[i+2]-(xaxis[i+1]+xaxis[i+2])/2) / (xaxis[i+2] - xaxis[i+1]) * hs[i+1])
        i += 1

    return lft, rgt

def solve():
    lft, rgt = create_matrix()
    res = np.zeros(N+1)
    res[1:-1] = tdma(lft, rgt)
    res[0] = left_value
    res[-1] = right_value
    return res
