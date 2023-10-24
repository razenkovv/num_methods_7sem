import numpy as np

def tdma(left, right):
    """
    TriDiagonal Matrix Algorithm\n
    Решение СЛАУ с трехдиагональной матрицей. Важно диагональное преобладание.\n
    :param left: массив значений в левой части матрицы (все ненулевые элементы подряд построчно)
    :param right: правая часть матрицы
    :return: массив решений
    """
    n = 3
    if len(left) != (len(right) - 2) * n + 4:
        raise Exception("tdma method: something wrong with input\n")
    a_coef = np.zeros(len(right))
    b_coef = np.zeros(len(right))
    a_coef[0] = right[0] / left[0]
    b_coef[0] = -1 * left[1] / left[0]
    ind = n
    for i in range(1, len(right) - 1):
        a_coef[i] = (right[i] - left[ind - 1] * a_coef[i - 1]) / (left[ind] + left[ind - 1] * b_coef[i - 1])
        b_coef[i] = -1 * left[ind + 1] / (left[ind] + left[ind - 1] * b_coef[i - 1])
        ind += n
    a_coef[-1] = (right[-1] - left[ind - 1] * a_coef[-2]) / (left[ind] + left[ind - 1] * b_coef[-2])
    b_coef[-1] = 0.0

    res = np.zeros(len(right))
    res[-1] = a_coef[-1]
    for i in range(len(res) - 2, -1, -1):
        res[i] = a_coef[i] + b_coef[i] * res[i + 1]
    return res
