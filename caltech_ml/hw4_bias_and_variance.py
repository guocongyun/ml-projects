import numpy as np
from scipy import integrate

RUNS = 1000
TRAIN_DATA_NUM = 2


class Dataset:

    def __init__(self):
        self.X = None
        self.Y = None

    def generate_data(self, classify=True, n=10):

        self.X, self.Y  = [], []
        
        for _ in range(n): # don't use np append in a loop as it's slow

            x1 = self.rand_point(1)
            # x = np.array(x1) # 0.23, 0.24 1) y=ax
            # x = np.array([1]) # 0.51, 0.26 2) y=b
            x = np.array([1, x1], dtype=float) # 0.50 1.75  3) y=ax+b
            # x = np.array(x1) # 0.504 33.5 4) y=ax^2
            # x = np.array([1, x1], dtype=float) # 0.594 326.5 4) y=ax^2 + b
            
            # IMPORTANT, 1 have default int value
            # IMPORTANT, if only have one parameter, don't put the [] for np.array as that cause trouble with dot product
            y = self.target_function(x, classify)

            # x = self.transform(x)

            self.X.append(x)
            self.Y.append(y)


        self.X, self.Y = np.array(self.X), np.array(self.Y)
        return self.X, self.Y

    def transform(self, x):
        return np.array([1,x[1]**2])

    def target_function(self, x, classify=True):
        print(x)
        # fx = np.sin(x*np.pi)
        # fx = np.sin(self.rand_point(1)*np.pi)
        fx = np.sin(x[1]*np.pi)
        # fx = np.sin(x*np.pi)
        if(classify): return np.sign(fx)
        else: return fx

    def rand_point(self,n):
        return np.random.uniform(-1,1,size=n)

class VarianceAndBias():

    def __init__(self):
        self.g_ = None
        self.weight = None
        self.hypothesis = lambda x, coefficient: coefficient[0] + coefficient[1]*x

    def calc_average_hypothesis(self, dataset, size = RUNS): # average hypothesis is in the form ax+b, and it's not a value
        total_coefficient = [0,0]
        average_coefficient = [0,0]
        for _ in range(RUNS):
            hypothesis = self.training(dataset)
            total_coefficient[0] += hypothesis[0]
            # total_coefficient[1] += hypothesis[1]
        average_coefficient[0] = total_coefficient[0]/RUNS
        # average_coefficient[1] = total_coefficient[1]/RUNS
        return average_coefficient

    def training(self, dataset):
        dataset.generate_data(False, TRAIN_DATA_NUM)
        transpose_X = np.transpose(dataset.X)
        inverse_XTX = np.linalg.pinv(transpose_X @ dataset.X)
        peusdo_inverse = inverse_XTX @ transpose_X
        self.weight = peusdo_inverse @ dataset.Y
        return self.weight

    def calc_bias(self, average_coefficient, dataset): # bias = Ex[(g_ - f)**2]
        upper_bound = 1
        lower_bound = -1
        fx = lambda x: 1/(upper_bound - lower_bound)*((self.hypothesis(x, average_coefficient) - dataset.target_function([1,x],False))**2)
        bias = integrate.quad(fx,lower_bound,upper_bound)[0]
        # total_bias = 0
        # iteration = 1000
        # for _ in range(int(lower_bound*(iteration)),int(upper_bound*(iteration)),(upper_bound-lower_bound)):
        #     total_bias += fx(_/iteration)
        # bias = total_bias/iteration*(upper_bound-lower_bound)
        return bias

    def calc_variance(self, average_coefficient, dataset): # variance = Ex[(g - g_)**2]
        upper_bound = 1
        lower_bound = -1
        fx = lambda x: 1/(upper_bound - lower_bound)*((self.hypothesis(x, self.training(dataset)) - self.hypothesis(x, average_coefficient))**2)
        # variance = integrate.quad(fx,lower_bound,upper_bound) # This function is not sutible for numerical integration
        total_variance = 0
        iteration = 1000
        for _ in range(int(lower_bound*(iteration)),int(upper_bound*(iteration)),(upper_bound-lower_bound)):
            total_variance += fx(_/iteration)
            variance = total_variance/iteration*(upper_bound-lower_bound)
        return variance

    def main(self):
        dataset_ = Dataset() 
        average_coefficient = self.calc_average_hypothesis(dataset_)
        bias = self.calc_bias(average_coefficient,dataset_)
        variance = self.calc_variance(average_coefficient,dataset_)
        print(average_coefficient)
        print(bias)
        print(variance)


if __name__ == "__main__":
    var_and_bias = VarianceAndBias()
    var_and_bias.main()
