"""Generic class for QuasiNewton"""

class QuasiNewton:
"""
Takes a problem as input, where problem is  an instance of Optimizationproblem. Formulate problem independent
of solutionmethod. ONLY things that have to do with the methods.
"""
    def __init__(self,problem):
        self.problem = problem
        self.gradient = """DEFINE GRADIENT"""
        self.Hessian = """Define HESSIAN"""
        self.alpha = 1
    def newstep(self,Hessian,x, alpha):
        """Takes a step"""
        return x-alpha*Hessian*self.gradient

    def newHessian():
        """Returns Hessian. Overridden in 9 special methods"""
        return self.Hessian

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




