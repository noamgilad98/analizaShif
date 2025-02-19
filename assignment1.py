"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random


class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
        """

        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time. 
        The assignment will be tested on variety of different functions with 
        large n values. 
        
        Interpolation error will be measured as the average absolute error at 
        2*n random points between a and b. See test_with_poly() below. 

        Note: It is forbidden to call f more than n times. 

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.** 
        
        Note: sometimes you can get very accurate solutions with only few points, 
        significantly less than n. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """

        def get_cubic(a, b, c, d):
            return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t,
                                                                                                              2) * c + np.power(
                t, 3) * d

        def T_solv(a, b, c, d):
            n = len(d)
            x = np.zeros((n, 2))
            bc = np.zeros(n)
            dc = np.zeros((n, 2))
            bc[0] = b[0]
            dc[0] = d[0]
            for i in range(1, n):
                bc[i] = b[i] - (c[i - 1] * (a[i] / bc[i - 1]))
                dc[i] = d[i] - (dc[i - 1] * (a[i] / bc[i - 1]))
            x[n - 1] = (dc[n - 1] / bc[n - 1])
            for i in range(n - 2, -1, -1):
                x[i] = (dc[i] - c[i] * x[i + 1]) / bc[i]
            return x

        def get_bezier_coef(points):
            n = len(points) - 1
            a = np.array([1 for i in range(n)])
            a[0], a[-1] = 0, 2
            b = np.array([4 for i in range(n)])
            b[0], b[-1] = 2, 7
            c = np.array([1 for i in range(n)])
            c[-1] = 0
            P = np.array([2 * (2 * points[i] + points[i + 1]) for i in range(n)])
            P[0] = points[0] + 2 * points[1]
            P[n - 1] = 8 * points[n - 1] + points[n]
            A = T_solv(a, b, c, P)
            B = [0] * n
            for i in range(n - 1):
                B[i] = 2 * points[i + 1] - A[i + 1]
            B[n - 1] = (A[n - 1] + points[n]) / 2
            return A, B

        def get_bezier_cubic(points):
            A, B = get_bezier_coef(points)
            return np.array([get_cubic(points[i], A[i], B[i], points[i + 1]) for i in range(len(points) - 1)])

        points = np.array([[i, f(i)] for i in np.linspace(a, b, n)])

        def g_func(points):
            n = len(points) - 1
            a, b = points[0][0], points[-1][0]
            curves = get_bezier_cubic(points)

            def find_index(x):
                i = int((x - a) / (b - a) * n)
                if i == n:
                    i -= 1
                return curves[i](abs((x - points[i][0]) / ((b - a) / n)))[1]

            return find_index

        g = g_func(points)
        return g



##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 3
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, 1, 3, 10)

            xs = np.random.random(200)
            print(xs)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)

if __name__ == "__main__":
    unittest.main()
