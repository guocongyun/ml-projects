import requests
import numpy as np
import random
from sklearn.linear_model import LinearRegression

#%%
class Dataset:

    def __init__(self):
        self.X = None
        self.X_validation = None
        self.Y = None
        self.line_num = None

    def read_data(self, location):

        with open(location) as data_in:
            line_num = 0
            self.X, self.Y = [], []
            for line in data_in:
                line = list(map(float, line.strip().split()))
                # default value for spliting is any white space
                # rstrip strip right space, lstrip strip left space, strip does both
                x = np.array([1, line[0], line[1]])

                # x = self.transformation()
                y = np.array(line[2])
                
                self.X.append(x)
                self.Y.append(y)
                
                line_num += 1

            self.line_num = line_num
            self.X, self.Y = np.array(self.X), np.array(self.Y)
            return self.X, self.Y

    def split_data(self, num):
        self.X_validation = self.X[num:]
        self.X = self.X[:num]

        self.Y_validation = self.Y[num:]
        self.Y = self.Y[:num]

    def transformation(self):
        transformations = [
            lambda x: x[:,0], # np.ones([len(x),len(x[0])])
            lambda x: x[:,1], # x[:,1] transforms the entire array
            lambda x: x[:,2],
            lambda x: x[:,1]**2,
            lambda x: x[:,2]**2,
            lambda x: x[:,1]*x[:,2],
            lambda x: np.abs(x[:,1] - x[:,2]),
            lambda x: np.abs(x[:,1] + x[:,2])
        ]
        # return lambda k: transformations[k](x)
        phi = lambda k: lambda x: np.column_stack([transform(x) for transform in transformations[:k+1]])
        # IMPORTANT 
        # column stack [1,2],[3,4]
        # => [1,3]
        # => [2,4]
        # row stack [1,2], [3,4]
        # => [1,2]
        # => [3,4]
        return phi


#%%

class LinearRegression_():
    
    def __init__(self):
        self.linear_reg = None

    def training(self, dataset,k): # minimising Error in, Ein
        dataset.read_data("in.dta")
        dataset.split_data(25)
        dataset.X, dataset.X_validation = dataset.X_validation, dataset.X 
        dataset.Y, dataset.Y_validation = dataset.Y_validation, dataset.Y
        # IMPORTANT, this can swap the two variable
        phi = dataset.transformation()
        self.linear_reg = LinearRegression(fit_intercept=False).fit(phi(k)(dataset.X),dataset.Y)

    def testing(self, error_total, dataset, k, insample = True): 

        # if (insample): dataset.read_data("in.dta")
        if (not insample): dataset.read_data("out.dta")

        if (insample): 
            prediction = np.sign(self.linear_reg.predict(dataset.transformation()(k)(dataset.X_validation)))
            actual_value = np.array(dataset.Y_validation)

        if (not insample): 
            prediction = np.sign(self.linear_reg.predict(dataset.transformation()(k)(dataset.X)))
            actual_value = np.array(dataset.Y)

        classification_error = np.mean(prediction != actual_value)
        error_total += classification_error
        return error_total

    def main(self):
        dataset_ = Dataset()
        k_range = range(0,8)
        # print(k_range)
        for k in k_range:
            error_insample_total = 0
            error_outsample_total = 0
            self.training(dataset_,k)
            error_insample_total = self.testing(error_insample_total, dataset_, k, True)
            error_outsample_total = self.testing(error_outsample_total, dataset_, k, False)
            print(k)
            print(error_insample_total)
            print(error_outsample_total)

        return error_insample_total, error_outsample_total

#%%
if __name__ == "__main__":
    linear_reg = LinearRegression_()
    linear_reg.main()
