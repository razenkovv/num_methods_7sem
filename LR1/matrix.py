import numpy as np


class Matrix:
    def __init__(self, data):
        """
        :param data: left part row by row or right part as a row
        """
        self._matrix = data

    def print(self):
        print(self._matrix)

    def __data__(self):
        return self._matrix

    def dimension(self):
        return self._matrix.shape

    def mult(self, rgt):
        return Matrix(self.__data__() @ rgt.__data__())

    def __sub__(self, rgt):
        return Matrix(self._matrix - rgt.__data__())

    def __getitem__(self, item):
        return self._matrix[item]

    def __setitem__(self, key, value):
        self._matrix[key] = value

    def norm(self):
        return np.linalg.norm(self._matrix)

    def gauss(self, col):
        _lft = Matrix(np.copy(self.__data__()))
        _rgt = Matrix(np.copy(col.__data__()))
        lft = _lft.__data__()
        rgt = _rgt.__data__()

        if _lft.dimension()[0] != _lft.dimension()[1] or _lft.dimension()[0] != _rgt.dimension()[0]:
            raise Exception('Something wrong with dimensions')
        for i in range(self.dimension()[0]):
            column_argmax = np.argmax(np.abs(lft[i:, i]))
            column_max = lft[i:, i][column_argmax]
            if np.isclose(column_max, 0.0):
                raise Exception('System can\'t be solved with this method. Matrix must be non-singular.')

            lft[[i, column_argmax + i]] = lft[[column_argmax + i, i]]
            rgt[[i, column_argmax + i]] = rgt[[column_argmax + i, i]]
            k = lft[i, i]
            rgt[i] /= k
            lft[i] /= k

            for j in range(i + 1, _lft.dimension()[0]):
                rgt[j] -= rgt[i] * lft[j][i]
                lft[j][i:] -= lft[i][i:] * lft[j][i]

        res = np.zeros(_lft.dimension()[0])
        for i in range(_lft.dimension()[0] - 1, -1, -1):
            s = 0.0
            for j in range(_lft.dimension()[0] - 1, i - 1, -1):
                s += lft[i, j] * res[j]
            res[i] = rgt[i] - s

        return res

    def jacobi(self, col, residual=1e-8, eps=1e-15):
        _lft = Matrix(np.copy(self.__data__()))
        _rgt = Matrix(np.copy(col.__data__()))
        lft = _lft.__data__()
        rgt = _rgt.__data__()

        if _lft.dimension()[0] != _lft.dimension()[1] or _lft.dimension()[0] != _rgt.dimension()[0]:
            raise Exception('Something wrong with dimensions')

        diag = np.diag(_lft.__data__())
        res = rgt / diag
        delta1 = eps + 1
        delta2 = residual + 1

        counter = 0
        while delta2 > residual:
            # print(f'{delta2}')
            res_new = (rgt - np.sum(lft * res, axis=1) + diag * res) / diag
            # delta1 = Matrix(res_new - res).norm()  # change between current and previous iteration
            delta2 = (_rgt - _lft.mult(Matrix(res_new))).norm()  # residual
            res = res_new
            counter += 1

        return res, counter

    def zeidel(self, col, residual=1e-8, eps=1e-15):
        _lft = Matrix(np.copy(self.__data__()))
        _rgt = Matrix(np.copy(col.__data__()))
        lft = _lft.__data__()
        rgt = _rgt.__data__()

        if _lft.dimension()[0] != _lft.dimension()[1] or _lft.dimension()[0] != _rgt.dimension()[0]:
            raise Exception('Something wrong with dimensions')

        diag = np.diag(_lft.__data__())
        res = rgt / diag
        res_new = np.zeros_like(res)
        delta1 = eps + 1
        delta2 = residual + 1

        counter = 0
        while delta2 > residual:
            # print(f'{delta2}')
            for i in range(0, len(res)):
                res_new[i] = (rgt[i] - np.sum(lft[i, :i] * res_new[:i]) - np.sum(lft[i, i+1:] * res[i+1:])) / lft[i, i]
            # delta1 = Matrix(res_new - res).norm()  # change between current and previous iteration
            delta2 = (_rgt - _lft.mult(Matrix(res_new))).norm()  # residual
            res = res_new
            counter += 1

        return res, counter
