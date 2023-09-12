import numpy as np


class Matrix:
    def __init__(self, data):
        """
        :param data: left part row by row or right part as a column
        """
        self._matrix = np.transpose(data)

    def print(self):
        print(np.transpose(self._matrix))

    def data(self):
        return np.transpose(self._matrix)

    def __data__(self):
        return self._matrix

    def dimension(self):
        return np.transpose(self._matrix).shape

    def gauss(self, rgt):
        if self.dimension()[0] != self.dimension()[1] or self.dimension()[0] != rgt.dimension()[0]:
            raise Exception('Something wrong with dimensions')
        for i in range(self.dimension()[0]):
            column_argmax = np.argmax(self._matrix[i][i:])
            column_max = self._matrix[i][i:][column_argmax]
            if np.isclose(column_max, 0.0):
                raise Exception('System can\'t be solved with this method. Matrix must be non-singular.')
            self._matrix[:, [i, column_argmax]] = self._matrix[:, [column_argmax, i]]
            rgt.__data__()[:, [i, column_argmax]] = rgt.__data__()[:, [column_argmax, i]]
            k = self._matrix[i, i]
            rgt.__data__()[:, i] /= k
            self._matrix[:, i] /= k
            for j in range(i+1, self.dimension()[0]):
                self._matrix[:]

            self.print()
            rgt.print()
            break
