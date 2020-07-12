import numpy as np
from matplotlib import pyplot as plt

RUNS = 1000
TRAIN_DATA_NUM = 10
TEST_DATA_NUM = 1000

class Dataset:

    def __init__(self):
        self.X = None
        self.Y = None

    def generate_data(self, classify=True, n=TRAIN_DATA_NUM):

        self.X, self.Y  = [], []
        
        for _ in range(n): # don't use np.array.append in a loop as it's slow
            x1 , x2 = self.rand_point(2)
            x = np.array([1, x1, x2])
            y = self.target_function(x, classify)

            if (np.random.randint(0,10) == 0): y = -y # add noise to y
            x = self.transformation(x) # transform x

            self.X.append(x)
            self.Y.append(y)

        self.X, self.Y = np.array(self.X), np.array(self.Y)
        return self.X, self.Y

    def transformation(self, x):
        return np.array([1, x[1], x[2], x[1]*x[2], x[1]**2, x[2]**2])

    def target_function(self, x, classify=True):
        fx = x[1] ** 2 + x[2] ** 2 - 0.6
        if(classify): return np.sign(fx)
        else: return fx

    def rand_point(self,n):
        return np.random.uniform(-1,1,size=n)

class LinearRegression():
    
    def __init__(self):
        self.weight = None
        self.temp_error = 1
        self.temp_weight = 1
    
    def training(self, dataset): 
        dataset.generate_data(True, TRAIN_DATA_NUM)
        transpose_X = np.transpose(dataset.X)
        inverse_XTX = np.linalg.pinv(transpose_X @ dataset.X)
        peusdo_inverse = inverse_XTX @ transpose_X
        self.weight = peusdo_inverse @ dataset.Y
        return self.weight

    def testing(self, error, dataset, insample = True): # Calculating Errorin and Errorout

        # Generate new data if testing outsample error
        if (TEST_DATA_NUM != 0 and not insample): dataset.generate_data(True, TEST_DATA_NUM)

        # testing using square error
        # square_error = np.dot(np.transpose(dataset.X @ self.weight - dataset.Y), (dataset.X @ self.weight - dataset.Y)) # not really square error
        # square_error = 1/TRAIN_DATA_NUM * square_error # @ means matrix multiplication
        # error += square_error

        prediction = np.sign(dataset.X @ self.weight) # the position in dot or @ product matters
        actual_value = np.array(dataset.Y)

        if (not insample): classification_error = sum(prediction != actual_value)/TEST_DATA_NUM
        elif(insample): classification_error = sum(prediction != actual_value)/TRAIN_DATA_NUM
        
        if (classification_error < self.temp_error ): self.temp_error = classification_error; self.temp_weight = self.weight

        error += classification_error
        return error

    def main(self):
        error_insample_total = 0
        error_outsample_total = 0
        dataset_ = Dataset() 

        for _ in range(RUNS):
            _ = self.training(dataset_)
            error_insample_total = self.testing(error_insample_total, dataset_, True)
            error_outsample_total = self.testing(error_insample_total, dataset_, False)
        print(self.temp_weight)

        return error_insample_total,error_outsample_total, self.weight

if __name__ == "__main__":
    linear_regression = LinearRegression()
    error_insample, error_outsample, weights = linear_regression.main()

    error_insample_avg = error_insample / RUNS
    error_outsample_avg = error_outsample / RUNS


    print("\nAverage in sample classification error is:")
    print("Ein(w) = ", error_insample_avg)
    print("Eout(w) = ", error_outsample_avg)