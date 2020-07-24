# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import requests
import numpy as np
import random

try:
    with open("in.dta", "x") as f_in:
        request_in = requests.get("http://work.caltech.edu/data/in.dta")
        f_in.write(request_in.text)
except FileExistsError as e:
    print("Training data already downloaded")

try:
    with open("out.dta","x") as f_out:
        request_out = requests.get("http://work.caltech.edu/data/out.dta")
        f_out.write(request_out.text)
except FileExistsError as e:
    print("Test data already downloaded")


# %%
class Dataset:

   def __init__(self):
       self.X = None
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
               x = self.transformation(x)
               y = np.array(line[2])
               
               self.X.append(x)
               self.Y.append(y)
               
               line_num += 1

           self.line_num = line_num
           self.X, self.Y = np.array(self.X), np.array(self.Y)
           return self.X, self.Y
   
   def transformation(self, x):
       return np.array([x[0], x[1], x[2], x[1]**2, x[2]**2, x[1]*x[2], abs(x[1]-x[2]), abs(x[1]+x[2])])


# %%
class LinearRegression():
    
    def __init__(self):
        self.weight = None
    
    def training(self, dataset,k): # minimising Error in, Ein
        dataset.read_data("in.dta")
        transpose_X = np.transpose(dataset.X)
        inverse_XTX = np.linalg.pinv(transpose_X @ dataset.X + (10**k) * np.identity(8)) # IMPORTANT, the weight decay depend on value for k
        peusdo_inverse = inverse_XTX @ transpose_X
        self.weight = peusdo_inverse @ dataset.Y

        return self.weight

    def testing(self, error_total, dataset, insample = True): 

        # Generate new data if testing outsample error
        if (insample): dataset.read_data("in.dta")
        if (not insample): dataset.read_data("out.dta")

        # testing using square error
        # square_error = np.transpose(dataset.X @ self.weight - dataset.Y) @ (dataset.X @ self.weight - dataset.Y) # not really square error
        # print(square_error)
        # square_error = 1/dataset.line_num * square_error # @ means matrix multiplication
        # error_total += square_error

        # testing using classification error
        prediction = np.sign(dataset.X @ self.weight) # the position in dot or @ product matters
        actual_value = np.array(dataset.Y)
        if (not insample): classification_error = sum(prediction != actual_value)/dataset.line_num
        elif(insample): classification_error = (sum(prediction != actual_value))/dataset.line_num
        error_total += classification_error
        return error_total

    def main(self):
        dataset_ = Dataset()
        k_range = range(-5,5)
        print(k_range)
        for k in k_range:
            error_insample_total = 0
            error_outsample_total = 0
            _ = self.training(dataset_,k)
            error_insample_total = self.testing(error_insample_total, dataset_, True)
            error_outsample_total = self.testing(error_outsample_total, dataset_, False)
            print(k)
            print(error_insample_total)
            print(error_outsample_total)

        return error_insample_total, error_outsample_total, self.weight


# %%
linear_reg = LinearRegression(
)
# print("waht")
linear_reg.main()

