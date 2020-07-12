import numpy as np
from sympy import diff, symbols
from matplotlib import pyplot as plt
import random

RUNS = 100
TRAIN_DATA_NUM = 100
TEST_DATA_NUM = 1000

class Dataset:

    def __init__(self):
        self.X = None
        self.Y = None
        self.slope = None
        self.constant = None

    def generate_data(self, classify=True, n=10):

        self.X, self.Y  = [], []

        for _ in range(n): # don't use np append in a loop as it's slow

            x1 , x2 = self.rand_point(2)
            x = np.array([1, x1, x2])
            y = self.evaluate(x, True)
            self.X.append(x)
            self.Y.append(y)

        self.X, self.Y = np.array(self.X), np.array(self.Y)
        return self.X, self.Y
    

    def create_target_function(self):
        x0, y0, x1, y1 = self.rand_point(4)
        self.slope = (y1-y0)/(x1-x0)
        self.constant = y0 - self.slope * x0

    def evaluate(self, x, classify=True):
        if classify: y = np.sign(self.constant*x[0]+self.slope*x[1]-x[2])
        else: y = self.constant*x[0]+self.slope*x[1]-x[2]
        return y

    def rand_point(self,n):
        return np.random.uniform(-1,1,size=n)

class GradientDescent:

    def __init__(self):
        self.learning_rate = 0.01

    def err_func(self, u, v):
        return (u*np.math.e**v-2*v*np.math.e**(-u))**2

    def err_func_(self, gradient):
        return (np.transpose(gradient) @ gradient)**(1/2) * self.learning_rate

    def err_func_deriv(self, x = symbols("x"), y = symbols("y")):
        u, v = symbols("u v")
        derivative = lambda x, y: np.array([float(diff(self.err_func(u,v),u).subs({u:x, v:y})),
                                   float(diff(self.err_func(u,v),v).subs({u:x, v:y}))])
        return derivative(x, y)

    def gradient_descent(self, weights, dataset ,highest_error = 10**-14):

        finished = False
        total_epoch = 0
        while (finished == False):

            old_weights = weights
            indices = list(range(len(dataset.X)))
            random.shuffle(indices)

            for index in indices:
                
                wt = np.transpose(weights)
                xn = dataset.X[index]
                yn = dataset.Y[index]
                component_gradient = -yn * xn / (1+np.math.exp(yn* (wt @ xn)))
                weights = weights - component_gradient * self.learning_rate
                # IMPORTANT weight-=1 modify the data in place, instad of creating a new object
                # hence if old_weights = weights, weights-=1 => old_weights-=1 
            
            total_epoch+=1
            
            if (np.linalg.norm(weights - old_weights) < 0.01): finished = True

        # print(np.linalg.norm(self.learning_rate*component_gradient)) # == ||self.learning_rate*component_gradient|| == (np.transpose(gradient) @ gradient)**(1/2) * self.learning_rate
        return weights,total_epoch

        # u = v = float(1.0)
        # iteration = 0
        # while(self.err_func(u,v) >= highest_error):
        #     err = self.err_func_deriv(u, v)
        #     u -= self.learning_rate*err[0]
        #     v -= self.learning_rate*err[1]
        #     iteration +=1

    def gradient_descent_(self, iteration_num = 15):
        u = v = float(1.0)
        err = self.err_func_deriv(u, v)
        for _ in range(iteration_num):
            err = self.err_func_deriv(u, v)
            u -= 0.1*err[0]
            err = self.err_func_deriv(u, v)
            v -= 0.1*err[1]


class LogisticRegression():
    
    def __init__(self):
        self.weight = None
    
    def training(self, dataset): # minimising Error in, Ein
        dataset.create_target_function()
        dataset.generate_data(True, TRAIN_DATA_NUM)
        gradient_descent = GradientDescent()
        self.weight,total_epoch = gradient_descent.gradient_descent([0,0,0], dataset, 0.01)
        return total_epoch

    def testing(self, error_total, dataset, insample = True): # Calculating Error in, Ein and Error out, Eo

        # Generate new data if testing outsample error
        if (not insample): dataset.generate_data(True, TEST_DATA_NUM)

        # testing using square error
        # square_error = np.dot(np.transpose(dataset.X @ self.weight - dataset.Y), (dataset.X @ self.weight - dataset.Y)) # not really square error
        # square_error = 1/TRAIN_DATA_NUM * square_error # @ means matrix multiplication
        # error_total += square_error

        # testing using classification error
        # prediction = np.sign(dataset.X @ self.weight) # the position in dot or @ product matters
        # actual_value = np.array(dataset.Y)
        # if (not insample): classification_error = sum(prediction != actual_value)/TEST_DATA_NUM
        # elif(insample): classification_error = sum(prediction != actual_value)/TRAIN_DATA_NUM
        # error_total += classification_error

        # testing using cross entropy error
        error_total = 0
        if (not insample):
            for _ in range(TEST_DATA_NUM):
                error_total += np.math.log(1 + np.math.exp(-dataset.Y[_] * np.dot(dataset.X[_], self.weight)))
        elif (insample):
            for _ in range(TRAIN_DATA_NUM):
                error_total += np.math.log(1 + np.math.exp(-dataset.Y[_] * (np.transpose(self.weight) @ dataset.X[_])))
        return error_total / TEST_DATA_NUM

    def main(self):
        total_epoch = 0
        error_insample_total = 0
        error_outsample_total = 0
        dataset_ = Dataset()

        for _ in range(RUNS):
            total_epoch += self.training(dataset_)
            # error_insample_total = self.testing(error_insample_total, dataset_, True)
            error_outsample_total += self.testing(error_insample_total, dataset_, False)

        print(error_outsample_total/RUNS)
        print(total_epoch/RUNS)
        

        # return error_insample_total,error_outsample_total, self.weight

if __name__ == "__main__":
    logistic_regression = LogisticRegression()
    logistic_regression.main()