from scipy import linalg

from timer import Timer
from matrix import Matrix


class Solver:
    def __init__(self):
        self.scipy_solver_result = None
        self.gauss_result = None
        self.jacobi_result = None
        self.zeidel_result = None

    def scipy_routine(self, lft, rgt):
        _lft = lft.__data__()
        _rgt = rgt.__data__()
        scipy_t = Timer()
        self.scipy_solver_result = linalg.solve(_lft, _rgt)
        scipy_t.stop()
        # print('\nScipy_solver result: ', self.scipy_solver_result)
        residual = Matrix(_rgt - _lft @ self.scipy_solver_result).norm()
        print(f'\nResidual of Scipy_solver: {residual}')
        print(f'Scipy_solver time: {scipy_t.get()}')
        return self.scipy_solver_result, residual

    def gauss_routine(self, lft, rgt):
        gauss_t = Timer()
        self.gauss_result = lft.gauss(rgt)
        gauss_t.stop()
        # print('\nGauss result       : ', self.gauss_result)
        print(f'\nGauss error: {Matrix(self.gauss_result - self.scipy_solver_result).norm()}')
        residual = (rgt - lft.mult(Matrix(self.gauss_result))).norm()
        print(f'Residual of Gauss: {residual}')
        print(f'Gauss time: {gauss_t.get()}')
        return self.gauss_result, residual

    def jacobi_routine(self, lft, rgt):
        jacobi_t = Timer()
        self.jacobi_result, jacobi_iter = lft.jacobi(rgt, residual=1e-8, eps=1e-15)
        jacobi_t.stop()
        # print('\nJacobi result       : ', self.jacobi_result)
        print(f'\nJacobi error: {Matrix(self.jacobi_result - self.scipy_solver_result).norm()}')
        residual = (rgt - lft.mult(Matrix(self.jacobi_result))).norm()
        print(f'Residual of Jacobi: {residual}')
        print(f'Jacobi time: {jacobi_t.get()}')
        print(f'Jacobi number of iterations: {jacobi_iter}')
        return self.jacobi_result, residual, jacobi_iter

    def zeidel_routine(self, lft, rgt):
        zeidel_t = Timer()
        self.zeidel_result, zeidel_iter = lft.zeidel(rgt, residual=1e-8, eps=1e-15)
        zeidel_t.stop()
        # print('\nZeidel result       : ', self.zeidel_result)
        print(f'\nZeidel error: {Matrix(self.zeidel_result - self.scipy_solver_result).norm()}')
        residual = (rgt - lft.mult(Matrix(self.zeidel_result))).norm()
        print(f'Residual of Zeidel: {residual}')
        print(f'Zeidel time: {zeidel_t.get()}')
        print(f'Zeidel number of iterations: {zeidel_iter}')
        return self.zeidel_result, residual, zeidel_iter
