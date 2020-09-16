# This is a sample Python script.
from numpy import *
# Press Skift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from numpy import*
from OptimizationProblem import*
from QuasiNewton import*

def testfunction(x):
    value = sin(x[0]) + 4*sin(x[1])
    return value
def testgradient(x):
    grad = cos(x[0])+cos(x[1])
    return grad


problem = OptimizationProblem(testfunction,testgradient)
solution = QuasiNewton(problem)
a = solution.solve()
print(a)





