import numpy as np
from matplotlib import pyplot as plt

RUNS = 1000
TRAIN_DATA_NUM = 10
TEST_DATA_NUM = 1000

# IMPORTANT target_function = [constant, slope, -1]
# X = [1, x1, x2]
# weight = [a0,a1,a2]

# the 2d graph is ploted in x,y, we are not trying to determine y, but using x, y to determine if it's above constant*1 + slope*x1 -y
# x2 == y !!!!
# and the actual output i.e. classification or value in the case of regression is some other unknown, z for exmaple
# however, since we a linear target function, we are only using two variabls z === 0 
# hence constant*1 + slope*x1 -1*x2 = 0 => constant*1 + slope*x1 = y

# hence w0+w1x+w2y = 0
# hence x = 0 => y = -w0/w2
# hence y = 0 => x = -w0/w1
# hence slope = -(w0/w2)/(w0/w1)
# intercept =-w0/w2

# dataset = dataset() will cause referenced before assignment error

# return 1 if x[0] * slope + constant > x[1] else -1 target function is a line, and you classify the points based on if they are on the line or not 

# matmul or @ doesn't support sclar multiplication,i.e. 1*matrix
# classification * x doesn't work unless both are array or list
# prediction = np.sign(dataset.X @ self.weight) # the position in dot or @ product matters
# matmul automaticlly traspose the multiplication, i.e. X.T @ Y == np.malmut(X,Y)

# plt.gca().set_aspect(1) # this makes the plot more symmetric
# plt.plot((-1,1),(actual_y_left, actual_y_right), color="green")  -1->actual_y_left and 1 -> actual_y_right

class Dataset:

    def __init__(self):
        self.X = None
        self.Y = None
        self.target_function = None

    def generate_data(self, target_function, classify=True, n=10):

        self.X, self.Y  = [], []
        
        for _ in range(n): # don't use np append in a loop as it's slow

            x1 , x2 = self.rand_point(2)
            x = np.array([1, x1, x2])
            y = self.classification(x, target_function, classify)
            self.X.append(x)
            self.Y.append(y)

        self.X, self.Y = np.array(self.X), np.array(self.Y)
        return self.X, self.Y

    def classification(self, x, target_function, classify):
        if classify: return int(np.sign(np.transpose(x) @ target_function)) 
        else: return int(np.matmul(np.transpose(x) @ target_function))

    def create_target_function(self):
        x0, y0, x1, y1 = self.rand_point(4)
        slope = (y1-y0)/(x1-x0)
        constant = y0 - slope * x0
        self.target_function = [constant,slope, -1]
        return np.array(self.target_function)

    def rand_point(self,n):
        return np.random.uniform(-1,1,size=n)


class Perceptron:

    def __init__(self, weights = None):
        self.weights = None

    def create_weights(self):
        self.weights = np.zeros(3) 
        # np.zeros(3): default type int
        # np.array([3]): type int
        # np.array([3.]): type float

    def predict(self, point):
        return np.sign(np.transpose(self.weights) @ point)
        
    def training(self, iteration_total, dataset): 
        if (dataset.target_function == None): dataset.create_target_function()
        X, Y = dataset.generate_data(dataset.target_function, TRAIN_DATA_NUM)
        
        # self.create_weights()
        # feeding weights from linear regression into preceptron
        
        linear_regression = LinearRegression()
        self.weights = linear_regression.training(dataset)
        while True:
            # self.plot(dataset)
            misclassified = []
            for x, classification in zip(X , Y):
                prediction = self.predict(x)
                if classification != prediction:
                    misclassified.append((x,classification))
            
            if misclassified:
                x, classification = misclassified[np.random.choice(len(misclassified))] # random.choice doesn't pop the item
                self.weights = self.weights + classification * x 
                iteration_total += 1 # iteration num means the num of changes

            else: break # the perceptron converges when miclassified == 0 which breaks the loop
    
        return iteration_total

    def testing(self, error_total, dataset): # test the perceptron
        X, Y = dataset.generate_data(dataset.target_function, TEST_DATA_NUM)
        y_values = Y

        y_hypothesis = np.sign(self.weights @ np.transpose(X))
        ratio_mismatch = ((y_values != y_hypothesis).sum())/TEST_DATA_NUM
        error_total += ratio_mismatch

        return error_total

    def plot(self, dataset):
        cs = ["red" if y > 0 else "blue" for y in dataset.Y]
        plt.scatter([x[1] for x in dataset.X], [x[2] for x in dataset.X], c=cs)
        y_left = (self.weights[1] - self.weights[0]) / self.weights[2]
        y_right = (-self.weights[1] - self.weights[0]) / self.weights[2]
        plt.plot((-1,1),(y_left, y_right), color="red")

        actual_y_left = + dataset.target_function[0] - dataset.target_function[1]
        actual_y_right = dataset.target_function[0] + dataset.target_function[1]
        plt.plot((-1,1),(actual_y_left, actual_y_right), color="green")


        plt.gca().set_aspect(1) # this makes the plot more symmetric
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.title("Candidate function found with PLA")
        plt.show()

    def main(self): 
        iteration_total = 0
        error_total = 0

        for _ in range(RUNS):
            dataset_ = Dataset() 
            iteration_total = self.training(iteration_total, dataset_)
            error_total = self.testing(error_total, dataset_)
        return iteration_total, error_total

class LinearRegression():
    
    def __init__(self):
        self.weight = None
    
    def training(self, dataset): # minimising Error in, Ein
        if (dataset.target_function == None): dataset.create_target_function()
        dataset.generate_data(dataset.target_function, True, TRAIN_DATA_NUM)
        transpose_X = np.transpose(dataset.X)
        inverse_XTX = np.linalg.pinv(transpose_X @ dataset.X)
        peusdo_inverse = inverse_XTX @ transpose_X
        self.weight = peusdo_inverse @ dataset.Y

        return self.weight

    def testing(self, error_total, dataset, insample = True): # Calculating Error in, Ein and Error out, Eo

        # Generate new data if testing outsample error
        if (not insample): dataset.generate_data(dataset.target_function, True, TEST_DATA_NUM)

        # testing using square error
        # square_error = np.dot(np.transpose(dataset.X @ self.weight - dataset.Y), (dataset.X @ self.weight - dataset.Y)) # not really square error
        # square_error = 1/TRAIN_DATA_NUM * square_error # @ means matrix multiplication
        # error_total += square_error

        # testing using classification error
        prediction = np.sign(dataset.X @ self.weight) 
        actual_value = np.array(dataset.Y)
        if (not insample): classification_error = sum(prediction != actual_value)/TEST_DATA_NUM
        elif(insample): classification_error = sum(prediction != actual_value)/TRAIN_DATA_NUM
        error_total += classification_error
        return error_total

    def main(self):
        error_insample_total = 0
        error_outsample_total = 0
        dataset_ = Dataset()

        for _ in range(RUNS):
            _ = self.training(dataset_)
            error_insample_total = self.testing(error_insample_total, dataset_, True)
            error_outsample_total = self.testing(error_outsample_total, dataset_, False)

        return error_insample_total,error_outsample_total, self.weight


if __name__ == "__main__":
    ##  This is the main for perceptron  ##
    perceptron = Perceptron()
    iteration_total, error_total = perceptron.main()
    iterations_avg = iteration_total / RUNS
    print("\nAverage number of PLA iterations over", RUNS, "runs: t_avg = ", iterations_avg)
    error_avg = error_total / RUNS
    print("\nAverage ratio for the mismatch between f(x) and h(x) outside of the training data:")
    print("P(f(x)!=h(x)) = ", error_avg)

    ##  This is the main for linear_regression  ##
    # linear_regression = LinearRegression()
    # error_insample_total, error_outsample_total, weights = linear_regression.main()
    # error_insample_avg = error_insample_total / RUNS
    # error_outsample_avg = error_outsample_total / RUNS
    # print("\nAverage in sample classification error is:")
    # print("Ein(w) = ", error_insample_avg)
    # print("Eout(w) = ", error_outsample_avg)