"""Generic class for QuasiNewton"""
from logging import exception
import matplotlib.pyplot as plt
from numpy import *
class QuasiNewton:

    def __init__(self,problem):
        self.epsilon = 0.00001  # step size
        self.f = problem.function# object function
        self.gradd = problem.gradient
        self.n = 2  # the dimension of the domain, R^n
        self.alpha = 1
        self.TOL = 1e-8
        self.values = array([])

    def gradient(self, x):
     """
     Computes the gradient of f at the given point x by forward-difference (p.195 Nocedal, Wright)
     :return: (array)
    """
     g = empty((self.n, 1))
     for i in range(self.n):  # for every coordinate of x
            e = zeros((self.n, 1))  # unit vectors in the domain
            e[i] = self.epsilon  # we want to take a step in the i:th direction
            g[i] = (self.f(x + e) - self.f(x - e)) / (2*self.epsilon)  # (8.1) at p.195

     return g

    def hessian(self, x):
        """
        Computes the inverse hessian of f at the given point x by forward-difference (p.201 Nocedal, Wright)
        :return: nxn matrix
        """
        G = empty((self.n,self.n))
        for i in range(self.n):
            for j in range(self.n):
                direction1 = zeros((self.n,1))
                direction2 = zeros((self.n,1))
                direction1[i] = self.epsilon
                direction2[j] = self.epsilon
                G[i,j] = (self.f(x + direction1 + direction2) - self.f(x+direction2) - self.f(x+direction1) + self.f(x))/(self.epsilon**2)
        G = (G + G.transpose())/2
        return G

    def newton_direction(self,x):
        """INTE ANVÄND ÄN. VET EJ OM VI KOMMER BEHÖVA DENNA SOM METOD"""
        inverse_hessian = linalg.inv(self.hessian(x))

        g = self.gradient(x)

        newton_direction = inverse_hessian.dot(g)
        return newton_direction

    def newstep(self,x):
        """
        Computes coordinates for the next step in accordance with the Quasi Newton procedure.
        :return: (array)
        """

        inverse_hessian = linalg.inv(self.hessian(x))

        g = self.gradient(x)

        newton_direction = inverse_hessian.dot(g)# The Newton direction determines step direction.


        new_coordinates = x-self.alpha*newton_direction
        return new_coordinates

    def newHessian():
        """Returns Hessian. Overridden in 9 special methods"""
        return

    def exactlinesearch(self, x):
        """
        Exact line search method as described in (3.3) p.31 Nocedal, Wright
        :param x:
        :return:
        """
        direct = self.newton_direction(x)

        alphas = linspace(2, 0, 100000)
        for alpha in alphas:
            graddarray = self.gradient(x-alpha*direct)
            if all(graddarray < 0.01):
                return alpha
        raise exception("NO ALPHA FOUND")


    def inexactlinesearch():
        """Defines inexact linesearch"""
        return 1

    def termination_criterion(self,x):
        """
        Asserts that the criterions for optimum are fulfilled. The criterions are:
        1.Hessian is symmetric and positive definite.
        2. The gradient is zero.
        :return: boolean that is true if criteria are fulfilled
        """
        Hess = self.hessian(x)
        if not all(self.gradient(x) < self.TOL ):
            return False
        """else:
            try:
                print(Hess)
                linalg.cholesky(Hess)
            except:
                return False #If the Choleskymethod gives error False is returned."""

        print("SOLVED!")
        return True

    def solve(self):
        x = ones((self.n,1))*2
        #self.alpha = self.exactlinesearch(x)
        solved = self.termination_criterion(x)
        value = x
        self.values = value


        print(shape(self.values))
        while solved is False:
            #self.alpha = self.exactlinesearch(value)
            newvalue = self.newstep(value)
            value = newvalue
            solved = self.termination_criterion(value)
            self.values = hstack((self.values,value))

        #plt.scatter(self.values[0], self.values[1])

        return [value, self.f(value)]





