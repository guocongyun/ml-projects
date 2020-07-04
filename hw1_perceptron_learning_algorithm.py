import numpy as np
from matplotlib import pyplot as plt

RUNS = 100
TRAIN_DATA_NUM = 10
TEST_DATA_NUM = 10

# IMPORTANT target_function = [constant, slope, -1]
# X = [1, x1, x2]
# weight = [a0,a1,a2]

# the 2d graph is ploted in x,y, we are not trying to determine y, but using x, y to determine if it's above constant*1 + slope*x1 -y
# x2 == y !!!!
# hence constant*1 + slope*x1 -1*x2 = 0 => constant*1 + slope*x1 = y

# hence w0+w1x+w2y = 0
# hence x = 0 => y = -w0/w2
# hence y = 0 => x = -w0/w1
# hence slope = -(w0/w2)/(w0/w1)
# intercept =-w0/w2

class dataset:

    def __init__(self):
        self.X, self.Y = None, None

    def generate_train_data(self, target_function, n=10):

        self.X, self.Y  = [], []
        
        for _ in range(n):

            x1 , x2 = self.rand_point(2)
            x = np.array([1, x1, x2])
            y = self.classification(x, target_function)
            self.X.append(x)
            self.Y.append(y)

        return self.X, self.Y

    def classification(self, x, target_function):
        return np.sign(np.dot(x,target_function))

        # return 1 if x[0] * slope + constant > x[1] else -1 
        # IMPORTANT target function is a line, and you classify the points based on if they are on the line or not 

    def create_target_function(self):
        x0, y0, x1, y1 = self.rand_point(4)
        slope = (y1-y0)/(x1-x0)
        constant = y0 - slope * x0
        return [constant,slope, -1]

    def rand_point(self,n):
        return np.random.uniform(-1,1,size=n)

class perceptron:

    def __init__(self):
        self.weights = None
        self.target_function = None

    def create_weights(self):
        self.weights = np.zeros(3) # default type int

    def predict(self, point):
        return int(np.sign(np.dot(self.weights, np.array(point))))
        
    def training(self, iteration_total, dataset): # train the perceptron
        self.target_function = dataset.create_target_function()
        X, Y = dataset.generate_train_data(self.target_function, TRAIN_DATA_NUM)
        self.create_weights()

        while True:
            self.plot(dataset)
            misclassified = []
            for x, classification in zip(X , Y):
                prediction = self.predict(x)
                if classification != prediction:
                    misclassified.append((x,classification))
            
            if misclassified:
                x, classification = misclassified[np.random.choice(len(misclassified))] # random.choice doesn't pop the item
                self.weights = self.weights + x * classification               
                iteration_total += 1 # iteration num means the num of changes


            else: break # the perceptron converges when miclassified == 0 which breaks the loop
        
        return iteration_total

    def testing(self, error_total, dataset): # test the perceptron
        X, Y = dataset.generate_train_data(self.target_function, TEST_DATA_NUM)
        y_values = Y
        y_hypothesis = np.sign(np.dot(X, self.weights))
        ratio_mismatch = ((y_values != y_hypothesis).sum())/TEST_DATA_NUM
        error_total += ratio_mismatch

        return error_total

    def plot(self, dataset):
        print(self.target_function)
        print(self.weights)
        cs = ["red" if y > 0 else "blue" for y in dataset.Y]
        plt.scatter([x[1] for x in dataset.X], [x[2] for x in dataset.X], c=cs)
        y_left = (self.weights[1] - self.weights[0]) / self.weights[2]
        y_right = (-self.weights[1] - self.weights[0]) / self.weights[2]
        plt.plot((-1,1),(y_left, y_right), color="red")

        actual_y_left = + self.target_function[0] - self.target_function[1]
        actual_y_right = self.target_function[0] + self.target_function[1]
        print(actual_y_left,actual_y_right)
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
            dataset_ = dataset() # IMPORTANT dataset = dataset() will cause referenced before assignment error
            iteration_total = self.training(iteration_total, dataset_)
            error_total = self.testing(error_total, dataset_)
        return iteration_total, error_total

if __name__ == "__main__":
    perceptron = perceptron()
    iteration_total, error_total = perceptron.main()

    iterations_avg = iteration_total / RUNS
    print("\nAverage number of PLA iterations over", RUNS, "runs: t_avg = ", iterations_avg)

    error_avg = error_total / RUNS
    print("\nAverage ratio for the mismatch between f(x) and h(x) outside of the training data:")
    print("P(f(x)!=h(x)) = ", error_avg)