# This is a sample Python script.
from numpy import *
# Press Skift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from numpy import*
from OptimizationProblem import*
from QuasiNewton import*

def testfunction(x):
    value = sin(x[0]) + 3*sin(x[1])
    return value
def testgradient(x):
    grad = array([cos(x[0]),cos(x[1])])
    return grad

def rosenbrock(x):
    if len(x) != 2:
        raise ValueError("Rosenbrock takes arguments from R^2, not R^{}".format(len(x)))
        return
    x_1 = x[0]
    x_2 = x[1]
    return 100*(x_2 - x_1**2)**2 + (1 - x_1)**2


problem = OptimizationProblem(rosenbrock,testgradient)
solution = QuasiNewton(problem)
solution.solve()
values = solution.values

a = solution.solve()
print(solution.values)






