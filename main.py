from OptimizationProblem import *
import matplotlib.pyplot as plt
from QuasiNewton import *
from linesearchmethods import inexact_linesearch
from chebyquad_problem import *
import scipy.optimize as opt

"""
Main file for manual testing of the quasi Newton solvers
"""


def contour_rosenbrock(levels=100, optipoints=array([])):
    """
    Plots the contours of the rosenbrock function, alternatively together with the optimization points
    :param levels: (int) number of level curves we want to display
    :param optipoints: (array)
    :return: None
    """

    # Verifying that 'optipoints' has the correct shape
    if optipoints.shape == (0,):
        pass
    elif optipoints.ndim == 2 and len(optipoints) == 2:  # optipoints is an ndarray with 2 rows
        pass
    else:
        raise ValueError("\'optipoints\' must have exactly 2 rows")

    size = 1000
    x = linspace(-0.5, 2, size)
    y = linspace(-1.5, 4, size)
    X, Y = meshgrid(x, y)
    input = array([X, Y])
    Z = rosenbrock(input)

    if len(optipoints) != 0:  # plot optimization points (either optipoints is empty or has 2 columns)
        plt.scatter(optipoints[0, :], optipoints[1, :])

    plt.contour(X, Y, Z, levels)
    plt.title("Contour plot of Rosenbrock's function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def task7():
    x = array([[0],
               [3]])
    rho = 0.1
    sigma = 0.7
    tau = 0.1
    chi = 9
    problem = OptimizationProblem(rosenbrock)
    method = QuasiNewton(problem)
    s = method.gradient(x)
    res = inexact_linesearch(rosenbrock, x, s, rho, sigma, tau, chi)
    print(res)


def newton_methods_test(problem=None):
    """
    Test Quasi Newton methods on the (optionally specified) 'problem'
    :param problem:
    :return:
    """

    # Let the user choose function if the problem wasn't already specified
    if not problem:
        function_name = input("Choose function to optimize:\n\t\'sinus\', \'paraboloid\' or \'rosebrock\' (dangerous)"
                         "\nFunction: ")
        valid_functions = {"sinus": sin_function, "paraboloid": paraboloid_function, "rosenbrock": rosenbrock}
        while function_name not in valid_functions:
            print("\nFunction \'{}\' does not exist, choose one of the following:\n\'sinus\', \'paraboloid\' or "
                  "\'rosenbrock\'".format(function_name))
            method = input("Function: ")

        function = valid_functions[function_name]

    # Let the user choose method
    method = input("\nTesting Newton methods, choose one of the following:\n\t\'newton\', \'goodBroyden\', "
                   "\'badBroyden\', \'symmetricBroyden\', \'DFP\', \'BFGS\'\nMethod: ")

    # Check that the chosen method is valid
    valid_methods = {"newton": Newton, "goodBroyden": GoodBroyden, "badBroyden": BadBroyden
                        , "symmetricBroyden": SymmetricBroyden, "DFP": DFP, "BFGS": BFGS}
    while method not in valid_methods:
        print("\nMethod \'{}\' does not exist, choose one of the following:\n\t\'newton\', \'goodBroyden\', "
              "\'badBroyden\', \'symmetricBroyden\', \'DFP\', \'BFGS\'".format(method))
        method = input("Method: ")

    if not problem:  # if we didn't specify a problem we take the 'simple_function'
        print("Using default problem: sinus function")
        problem = OptimizationProblem(function)

    # Choose line search method
    lsm = input("\nChoose line search method:\t\'exact\' or \'inexact\'\nLSM: ")

    # Create solver-instance
    try:
        solver = valid_methods[method](problem, lsm)
    except ValueError:
        lsm = input("\nLSM \'{}\' does not exist, choose between \'exact\' and \'inexact\'\nLSM: ".format(lsm))
        solver = valid_methods[method](problem, lsm)

    print("\nOptimizing {} with {} method with {} line search".format(function_name, method, lsm))
    min_point, min_value = solver.solve()
    # optipoints = solver.values
    print("\nOptimal point:\n", min_point)
    print("\nMinimum value:\n", min_value)
    # contour_rosenbrock(optipoints=optipoints)  # uncomment to plot rosenbrock_contour and optimization points



def chebyquadtest():

    degree = input("Testing mimimization of the Chebyquad function of degree n. Choose degree n: ")
    degree = int(degree)

    method = input(
        "Testing Newton methods, choose one of the following:\n\t\'newton\', \'goodBroyden\', \'badBroyden\', \'symmetricBroyden\', \'DFP\', \'BFGS\'\nMethod: ")

    # Check that the chosen method is valid
    valid_methods = {"newton": Newton, "goodBroyden": GoodBroyden, "badBroyden": BadBroyden
        , "symmetricBroyden": SymmetricBroyden, "DFP": DFP, "BFGS": BFGS}
    while method not in valid_methods:
        print(
            "\nMethod \'{}\' does not exist, choose one of the following:\n\t\'newton\', \'goodBroyden\', \'badBroyden\', \'symmetricBroyden\', \'DFP\', \'BFGS\'".format(
                method))
        method = input("Method: ")

    problem = OptimizationProblem(chebyquad,dimension =degree)
    solver = valid_methods[method](problem, lsm = "inexact")
    a = solver.solve()

    print()
    print("Optimizing the Chebyquad function of degree n = " + str(degree) + " with a " + str(method) + "-method: ")
    print("Function value: " + str(a[1]))
    print("The minimum was found in: " + str(transpose(a[0])))
    print()
    print("Optimizing the same function with numpy.optimize.fmin: ")
    min2 = opt.fmin_bfgs(chebyquad, ones((degree, 1)) * 1, disp=False, full_output=True)
    print("Function value: " + str(min2[1]))
    print("The minimum was found in: " + str(min2[0]))
    print()

    if isclose(min2[0],transpose(a[0])).all() and isclose(min2[1],a[1]):
        print("The two methods yield the same result.")
    else:
        print("The two methods do NOT yield the same result.")

def HessianQualityControl():

    problem = OptimizationProblem()
    solver = BFGS(problem, hessians= "on")
    solution = solver.solve()
    inverse_hessians = solution[0]

    def true_hessian_inverse(x):
        hessian = [1/(-3*sin(x[0])), 0, 0, 1/-sin(x[1])]
        hessian = reshape(hessian,[-1])
        return hessian

    nmbr = size(solution[1])
    inverse_hessians = reshape(solution[0],[-1])
    allhess = empty(0)
    differences = empty(0)
    k = 1


    #print(inverse_hessians)

    for i in range(0,nmbr,2):
        coordinates = solution[1][i:i+2]
        truehess = true_hessian_inverse(coordinates)
        #print(truehess)
        allhess = append(allhess,truehess)

    while inverse_hessians.size > 1:
        testvalues = inverse_hessians[0:4]
        #print(testvalues)
        #print()
        truevalues = allhess[0:4]
        #print(truevalues)
        #print()

        difference = testvalues-truevalues
        print("k = " + str(k))
        k +=1
        print(difference)
        differences = append(differences, mean(difference))

        inverse_hessians = inverse_hessians[4:-1]
        allhess = allhess[4:-1]




    x = linspace(1,k-1,k-1)
    plt.plot(x,differences)
    plt.xlabel("k")
    plt.ylabel("Mean difference between calculated and exact inverse Hessian")
    plt.show()
    #print(differences)
    #print()

def main():
    #newton_methods_test()
    #chebyquadtest()
    HessianQualityControl()


main()