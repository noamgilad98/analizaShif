"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment. 

"""

import numpy as np
import time
import random


class Assignment4:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def fit(self, f: callable, a: float, b: float, d:int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        f : callable. 
            A function which returns an approximate (noisy) Y value given X. 
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int 
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        a function:float->float that fits f between a and b
        """

        def derivative(f, x):
            h = 0.0001
            return (f(x + h) - f(x)) / h

        def solver(f, a, step, maxerr, max_iter):
            xn = 0
            for n in range(0, max_iter):
                if not a <= xn < a + step:
                    return None
                y = f(xn)
                if abs(y) < maxerr:
                    return xn
                slope = derivative(f, xn)
                if (slope == 0):
                    return None
                xn = xn - y / slope
            return None

        def bezier_fit_range(f, a, b, tt, C_arg):
            range_f = np.abs(b - a)
            P = [[a + x * range_f, f(a + x * range_f)] for x in tt]
            # M = np.array([[-1, +3, -3, +1], [+3, -6, +3, 0], [-3, +3, 0, 0], [+1, 0, 0, 0]])
            # tt = np.linspace(0, 1, n)
            # T = np.array([[t ** 3, t ** 2, t ** 1, t ** 0] for t in tt])
            # t_t_inv = np.linalg.inv(np.dot(T.transpose(), T))
            # m_inv = np.linalg.inv(M)
            C = np.dot(C_arg, P)
            x0, x1, x2, x3 = C[0][0], C[1][0], C[2][0], C[3][0]
            y0, y1, y2, y3 = C[0][1], C[1][1], C[2][1], C[3][1]
            ay = -y0 + 3 * y1 - 3 * y2 + y3
            by = 3 * y0 - 6 * y1 + 3 * y2
            cy = -3 * y0 + 3 * y1
            dy = y0
            B_t = np.poly1d([ay, by, cy, dy])

            def bezier(t):
                if t is None:
                    t = 0.5
                return B_t(t)

            def find_t(x):
                a = -x0 + 3 * x1 - 3 * x2 + x3
                b = 3 * x0 - 6 * x1 + 3 * x2
                c = -3 * x0 + 3 * x1
                d = x0 - x
                f = np.poly1d([a, b, c, d])
                return bezier(solver(f, -5, 10, 0.000001, 20))

            return lambda x: find_t(x)

        T = time.time()
        f(a)
        T = time.time() - T
        if T <= 0.001:
            T = 0.001
        num = int((maxtime - 0.4) / T) - 3
        s = d
        n = int(num / d)
        if s <= 0:
            s = 1
        if np.abs(b-a) <= 0.0001:
            y = np.mean([f(a) for _ in range(num)])
            return lambda x: y
        # if d <= 0:
        #     d = 5
        # num = int((maxtime - 1.2) / T)
        # if num <= 20:
        #     s = 1
        #     n = num
        # elif num <= 30:
        #     s = 2
        #     n = int(num / 2)
        # elif (num / d) >= 10:
        #     s = d
        #     n = int(num / d)
        # else:
        #     s = 3
        #     n = int(num / 3)
        # if s == 0:
        #     s += 1
        # s = int(maxtime/2)
        # n = 10
        bezier_list = [0] * s
        M = np.array([[-1, +3, -3, +1], [+3, -6, +3, 0], [-3, +3, 0, 0], [+1, 0, 0, 0]])
        tt = np.linspace(0, 1, n)
        T = np.array([[t ** 3, t ** 2, t ** 1, t ** 0] for t in tt])
        t_t_inv = np.linalg.inv(np.dot(T.transpose(), T))
        m_inv = np.linalg.inv(M)
        C_arg = np.linalg.multi_dot([m_inv, t_t_inv, T.transpose()])
        r = np.linspace(a, b, s + 1)
        for i in range(s):
            bezier_list[i] = bezier_fit_range(f, r[i], r[i + 1], tt, C_arg)

        def find_index(x):
            max_index = len(bezier_list)
            i = int((x - a) / (b - a) * (s))
            if i == max_index:
                i -= 1
            return bezier_list[i](x)

        return find_index


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    # def test_return(self):
    #     f = NOISY(0.01)(poly(1,1,1))
    #     ass4 = Assignment4()
    #     T = time.time()
    #     shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
    #     T = time.time() - T
    #     self.assertLessEqual(T, 5)

    # def test_delay(self):
    #     # f = DELAYED(7)(NOISY(0.01)(poly(1,1,1)))
    #
    #     ass4 = Assignment4()
    #     T = time.time()
    #     shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
    #     T = time.time() - T
    #     self.assertGreaterEqual(T, 5)

    def test_err(self):
        # f = poly(1,1,1)
        f = lambda x: 3*(x**6)-4*(x**5)+1*(x**4)+7*(x**3)+2*(x**2)+1*x+10
        nf = DELAYED(0.01)(NOISY(1)(f))
        # x = 15
        # y , yy = f(x) , np.mean([nf(x) for j in range(200)])
        # print(y ,yy)
        ass4 = Assignment4()
        T = time.time()
        ff = ass4.fit(f=nf, a=-1, b=-0.9999999, d=10, maxtime=5)
        T = time.time() - T
        mse=0
        for x in np.linspace(-1,-0.9999999,1000):
            self.assertNotEqual(f(x), nf(x))
            mse+= (f(x)-ff(x))**2
        mse = mse/1000
        print(mse , T)


if __name__ == "__main__":
    unittest.main()
