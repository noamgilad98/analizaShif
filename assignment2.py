"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.
        
        This function may not work correctly if there is infinite number of
        intersection points. 


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        def derivative(f, x):
            h = 1e-6
            return (f(x + h) - f(x)) / h

        def solver(f, a, step, maxerr, max_iter):
            xn = a
            for n in range(0, max_iter):
                y = f(xn)
                if abs(y) < maxerr:
                    return xn
                slope = derivative(f, xn)
                if (slope == 0):
                    return None
                xn = xn - y / slope
                if not a < xn < step:
                    return None
            return None

        def loop(f, a, b, step, maxerr, max_iter):
            solutions = np.array([])
            x = np.arange(a, b, step)
            if x[-1] < b:
                x = np.append(x, b)
            for i in range(len(x) - 1):
                root = solver(f, x[i], x[i + 1], maxerr, max_iter)
                if root is not None:
                    solutions = np.append(solutions, root)
            return solutions

        def find_roots(f, a, b, maxerr):
            roots = np.array([])
            n = abs(b - a)
            if n <= 2:
                roots = loop(f, a, b, 0.01, maxerr, 5)
                return roots
            elif n <= 10:
                range_split = np.linspace(a, b, 6)
                for i in range(5):
                    s = range_split[i]
                    x = np.array([abs(f(s + j * 0.15)) for j in range(4)])
                    t = x.max() - x.min()
                    if t > 1.5 and abs(x.mean()) <= 2:
                        ans = loop(f, range_split[i], range_split[i + 1], 0.02, maxerr, 10)
                        roots = np.append(roots, ans)
                    else:
                        ans = loop(f, range_split[i], range_split[i + 1], 0.1, maxerr, 10)
                        roots = np.append(roots, ans)
                return roots
            elif n <= 50:
                range_split = np.linspace(a, b, 15)
                for i in range(14):
                    s = range_split[i]
                    x = np.array([abs(f(s + j * 0.001)) for j in range(4)])
                    t = x.max() - x.min()
                    if t > 0.5 and abs(x.mean()) <= 1.5:
                        ans = loop(f, range_split[i], range_split[i + 1], 0.02, maxerr, 10)
                        roots = np.append(roots, ans)
                    elif (t < 0.5 and derivative(f, s) > 0 and f(s) > 10) or (
                            t < 0.5 and derivative(f, s) < 0 and f(s) < 10):
                        ans = loop(f, range_split[i], range_split[i + 1], 1, maxerr, 20)
                        roots = np.append(roots, ans)
                    else:
                        ans = loop(f, range_split[i], range_split[i + 1], 0.2, maxerr, 20)
                        roots = np.append(roots, ans)
                return roots
            else:
                range_split = np.linspace(a, b, 25)
                for i in range(24):
                    s = range_split[i]
                    x = np.array([abs(f(s + j * 0.001)) for j in range(4)])
                    t = x.max() - x.min()
                    if t > 1 and abs(x.mean()) <= 1.5:
                        ans = loop(f, range_split[i], range_split[i + 1], 0.02, maxerr, 10)
                        roots = np.append(roots, ans)
                    elif (t < 0.5 and derivative(f, s) > 0 and f(s) > 15) or (
                            t < 0.5 and derivative(f, s) < 0 and f(s) < 15):
                        ans = loop(f, range_split[i], range_split[i + 1], 3, maxerr, 50)
                        roots = np.append(roots, ans)
                    else:
                        ans = loop(f, range_split[i], range_split[i + 1], 1, maxerr, 50)
                        roots = np.append(roots, ans)
                return roots

        return find_roots(lambda t: f1(t) - f2(t), a, b, maxerr)


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        # f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f10, f3_nr, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()
