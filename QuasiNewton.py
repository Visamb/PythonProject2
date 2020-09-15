"""Generic class for QuasiNewton"""

class QuasiNewton:
"""
Takes a problem as input, where problem is  an instance of Optimizationproblem. Formulate problem independent
of solutionmethod. ONLY things that have to do with the methods.
"""
    def __init__(self,problem):
            self.alpha = 0.0001  # step size
            self.f = problem.function# object function
            self.gradient = problem.gradient#object gradient function
            self.n = 2  # the dimension of the domain, R^n

        def gradient(self, x):
            """
            Computes the gradient of f at the given point x by forward-difference (p.195 Nocedal, Wright)
            :return: (array)
            """
            g = empty((self.n, 1))
            for i in range(self.n):  # for every coordinate of x
                e = zeros((self.n, 1))  # unit vectors in the domain
                e[i] = self.alpha  # we want to take a step in the i:th direction
                g[i] = (self.f(x + self.alpha * e) - self.f(x)) / self.alpha  # (8.1) at p.195

            return g

        def hessian(self):
            """Computes hessian of f at the given point x """
            return H

        

    def newstep(self,Hessian,x, alpha):
        """Takes a step"""
        return x-alpha*Hessian*self.gradient

    def newHessian():
        """Returns Hessian. Overridden in 9 special methods"""
        return

    def exactlinesearch():
        """Defines exact line search"""
        return alpha

    def inexactlinesearch():
        """Defines inexact linesearch"""
        return alpha

    def solve(exact = true):
    """Solves the problem, (Whilesatserna osv som kollar right och left conditions). Kallar upprepade gånger på Newstep
    och newHessian parameter som från början är exact (eller inexact) linesearch. Hade kanske kunnat läggas i en call"""
    return [optimal x, maxvalue]




