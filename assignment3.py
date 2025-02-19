"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""

import numpy as np
import time
import random


class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n
        points. Your main objective is minimizing the integration error.
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions.

        Integration error will be measured compared to the actual value of the
        definite integral.

        Note: It is forbidden to call f more than n times.

        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """
        if n == 1:
            return np.float32(f((b - a) / 2) * (b - a))
        if n == 2:
            return np.float32((f((b)+f(a))/ 2) * (b - a))
        elif (n < 7) or ((b - a) < 5):
            if n%2 == 0:
                n -= 1
            Y = np.array([f(i) for i in np.linspace(a, b, n)])
            h = (b - a) / (n-1)
            F0 = Y[0]
            F1 = 0
            F2 = Y[-1]
            for i in range(1, n - 1):
                if i % 2 == 0:
                    F0 += Y[i]
                    F2 += Y[i]
                else:
                    F1 += Y[i]
            return np.float32(h / 3 * (F0 + 4 * F1 + F2))
        else:
            if n%2 == 0:
                n -= 1
            t = 5
            if n > 20:
                t1 = int((1 / ((b - a) / n))*1.5)
                t = max(t, t1)
            points = np.array([[i, None] for i in np.linspace(a, b, n)])
            f1, i1 = f(points[t][0]), points[t][0]
            points[t][1] = f1
            f2, i2 = f(points[int((n - t) / 2)][0]), points[int((n - t) / 2)][0]
            points[int((n - t) / 2)][1] = f2
            f3, i3 = f(points[-1][0]), points[-1][0]
            points[-1][1] = f3
            count = 3
            if np.abs((f2 - f1) / (i2 - i1)) <= 0.25 and np.abs((f3 - f2) / (i3 - i2)) <= 0.15:
                ans = ((i3 - i2) * ((f3 + f2) / 2)) + ((i2 - i1) * ((f2 + f1) / 2))
                f4, i4 = f(points[2][0]), points[2][0]
                count += 1
                while (n - count) >= 2:
                    ans += ((i1 - i4) * ((f1 + f4) / 2))
                    f1, i1 = f4, i4
                    f4, i4 = f((i1 + a) / 2), (i1 + a) / 2
                    count += 1
                ans += ((i1 - i4) * ((f1 + f4) / 2))
                f1, i1 = f4, i4
                f4, i4 = f(a), a
                ans += ((i1 - i4) * ((f1 + f4) / 2))
                return np.float32(ans)
            else:
                for point in points:
                    if point[1] == None:
                        point[1] = f(point[0])
                Y = np.transpose(points)[1]
                h = (b - a) / (n-1)
                F0 = Y[0]
                F1 = 0
                F2 = Y[-1]
                for i in range(1, n - 1):
                    if i % 2 == 0:
                        F0 += Y[i]
                        F2 += Y[i]
                    else:
                        F1 += Y[i]
                ans = (h / 3 * (F0 + 4 * F1 + F2))
                return np.float32(ans)

    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds
        all intersection points between the two functions to work correctly.

        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area.

        In order to find the enclosed area the given functions must intersect
        in at least two points. If the functions do not intersect or intersect
        in less than two points this function returns NaN.
        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """

        def derivative(f, x):
            h = 1e-6
            return (f(x + h) - f(x)) / h

        def newton(f, a, step, maxerr=0.0001, max_iter=15):
            xn = a
            for n in range(0, max_iter):
                y = f(xn)
                if abs(y) < maxerr:
                    return xn
                slope = derivative(f, xn)
                if (slope == 0):
                    return None
                xn = xn - y / slope
            return None

        def my_bisection(f, a, b, tol=0.0001):
            m = (a + b) / 2
            if np.abs(f(m)) < tol:
                return m
            elif np.sign(f(a)) == np.sign(f(m)):
                return my_bisection(f, m, b, tol)
            elif np.sign(f(b)) == np.sign(f(m)):
                return my_bisection(f, a, m, tol)

        def find_root(f, a, b):
            root = newton(f, a, b)
            if root == None:
                root = my_bisection(f, a, b)
            return root

        f = lambda x: f1(x) - f2(x)
        left_root = 0
        right_root = 0
        root_find = 0
        split = np.linspace(1, 100, 150)
        y = np.array([f(x) for x in split])
        if y[0] != 0:
            for i in range(99):
                # a = split[i]
                # b = split[i + 1]
                if y[i] * y[i + 1] < 0:
                    a = split[i]
                    b = split[i + 1]
                    left_root = find_root(f, a, b)
                    root_find += 1
                    if y[-1] != 0:
                        for j in range(149, -i, -1):
                            # a = split[j]
                            # b = split[j - 1]
                            if y[j] * y[j - 1] < 0:
                                a = split[j]
                                b = split[j - 1]
                                right_root = find_root(f, b, a)
                                root_find += 1
                                break
                    else:
                        right_root = split[-1]
                        root_find += 1
                    break
        else:
            left_root = split[0]
            root_find += 1
            if y[-1] != 0:
                for j in range(149, -1, -1):
                    if y[j] * y[j - 1] < 0:
                        a = split[j]
                        b = split[j - 1]
                        right_root = find_root(f, b, a)
                        root_find += 1
                        break
                    else:
                        right_root = split[-1]
                        root_find += 1
        if root_find <= 1:
            return np.float32(None)
        elif np.abs(right_root - left_root) <= 0.002:
            return np.float32(None)
        else:
            f = lambda x: np.abs(f1(x) - f2(x))
            a, b = left_root, right_root
            n = int((b - a) * 40)
            if n % 2 == 0:
                n += 1
            h = (b - a) / (n - 1)
            X = np.linspace(a, b, n)
            y = np.array([f(x) for x in X])
            F0 = y[0]
            F1 = 0
            F2 = y[-1]
            for i in range(1, n - 1):
                if i % 2 == 0:
                    F0 += y[i]
                    F2 += y[i]
                else:
                    F1 += y[i]
            return np.float32(h / 3 * (F0 + 4 * F1 + F2))


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        r = ass3.integrate(f1, -1, 10, 10)

        self.assertEquals(r.dtype, np.float32)

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 20)
        true_result = -7.78662 * 10 ** 33
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))


if __name__ == "__main__":
    unittest.main()
