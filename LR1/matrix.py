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

    def gauss(self, col):
        _lft = Matrix(np.copy(self.__data__()))
        _rgt = Matrix(np.copy(col.__data__()))

        if _lft.dimension()[0] != _lft.dimension()[1] or _lft.dimension()[0] != _rgt.dimension()[0]:
            raise Exception('Something wrong with dimensions')
        for i in range(self.dimension()[0]):
            column_argmax = np.argmax(np.abs(_lft.__data__()[i:, i]))
            column_max = _lft.__data__()[i:, i][column_argmax]
            if np.isclose(column_max, 0.0):
                raise Exception('System can\'t be solved with this method. Matrix must be non-singular.')

            _lft.__data__()[[i, column_argmax + i]] = _lft.__data__()[[column_argmax + i, i]]
            _rgt.__data__()[[i, column_argmax + i]] = _rgt.__data__()[[column_argmax + i, i]]
            k = _lft.__data__()[i, i]
            _rgt.__data__()[i] /= k
            _lft.__data__()[i] /= k

            for j in range(i+1, _lft.dimension()[0]):
                _rgt.__data__()[j] -= _rgt.__data__()[i] * _lft.__data__()[j][i]
                _lft.__data__()[j][i:] -= _lft.__data__()[i][i:] * _lft.__data__()[j][i]

        res = np.zeros(_lft.dimension()[0])
        for i in range(_lft.dimension()[0] - 1, -1, -1):
            s = 0.0
            for j in range(_lft.dimension()[0] - 1, i - 1, -1):
                s += _lft.__data__()[i, j] * res[j]
            res[i] = _rgt.__data__()[i] - s

        return res
