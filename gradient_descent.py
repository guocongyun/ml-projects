import numpy as np
from sympy import diff, symbols
class GradientDescent:

    def __init__(self):
        self.iteration = 0
        self.acceptable_error = 10**-14

    def error_function(self, u, v):
        return (u*np.math.e**v-2*v*np.math.e**(-u))**2

    def error_function_derivative(self, x = symbols("x"), y = symbols("y")):
        u, v = symbols("u v")
        derivative = lambda x, y: np.array([float(diff(self.error_function(u,v),u).subs({u:x, v:y})),
                                   float(diff(self.error_function(u,v),v).subs({u:x, v:y}))])
        return derivative(x, y)
    
    def gradient_descent(self, highest_error = 10**-14):
        u = v = float(1.0)
        iteration = 0
        while(self.error_function(u,v) >= highest_error):
            err = self.error_function_derivative(u, v)
            u -= 0.1*err[0]
            v -= 0.1*err[1]
            iteration +=1

        print(iteration)
        print(u,v)

    def gradient_descent_(self, iteration_num = 15):
        u = v = float(1.0)
        err = self.error_function_derivative(u, v)
        for _ in range(iteration_num):
            err = self.error_function_derivative(u, v)
            u -= 0.1*err[0]
            err = self.error_function_derivative(u, v)
            v -= 0.1*err[1]

        print(u,v)
        print(self.error_function(u,v))

if __name__ == "__main__":
    gradient_descent = GradientDescent()
    gradient_descent.gradient_descent_()
