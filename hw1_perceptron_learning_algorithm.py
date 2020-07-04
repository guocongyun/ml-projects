import numpy as np
from matplotlib import pyplot as plt

RUNS = 1000

# ISSUE: why is y determined that way

class dataset:

    def generate_train_data(self, slope, constant, n=10):

        X = []
        Y = []
        
        for _ in range(n):
            x1 , x2 = self.rand_point()
            x = np.array([x1,x2,1])
            y = self.classification(x, slope, constant)
            X.append(x)
            Y.append(y)
        return X, Y

    def classification(self, x, slope, constant):
        return 1 if x[0] * slope + constant > x[1] else -1 # IMPORTANT target function is a line, and you classify the points based on if they are on the line or not 

    def create_target_function(self):
        x0, y0 = self.rand_point()
        x1, y1 = self.rand_point()
        slope = (y1-y0)/(x1-x0)
        constant = y0 - slope * x0
        return slope, constant

    def rand_point(self):
        return np.random.uniform(-1,1), np.random.uniform(-1,1)

class perceptron:

    def create_weights(self):
        weights = np.zeros(3) # default type int
        return weights

    def predict(self, point, weights):
        return int(np.sign(np.dot(weights, np.array(point))))
        
    def main(self): 
        iteration_total = 0
        ratio_mismatch_total = 0
        for _ in range(RUNS):

            # train the perceptron
            train_data = dataset()
            slope, constant = train_data.create_target_function()
            X, Y = train_data.generate_train_data(slope, constant, 100)
            weights = self.create_weights()

            while True:
                misclassified = []
                for x, classification in zip(X , Y):
                    prediction = self.predict(x, weights)
                    if classification != prediction:
                        misclassified.append((x,classification))
                
                if misclassified:
                    
                    x, classification = misclassified[np.random.choice(len(misclassified))] # choice also pops the item
                    weights = weights + x * classification
                    
                    iteration_total += 1 # iteration num means the num of changes

                else: break # the perceptron converges when miclassified == 0

            # test the perceptron
            train_data = dataset()
            X, Y = train_data.generate_train_data(slope, constant, 1000)

            y_target = Y
            y_hypothesis = np.sign(np.dot(X, weights))
            ratio_mismatch = ((y_target != y_hypothesis).sum())/1000
            ratio_mismatch_total += ratio_mismatch
            

        return iteration_total, ratio_mismatch_total


if __name__ == "__main__":
    perceptron = perceptron()
    iteration_total, ratio_mismatch_total = perceptron.main()

    iterations_avg = iteration_total / RUNS
    print("\nAverage number of PLA iterations over", RUNS, "runs: t_avg = ", iterations_avg)

    ratio_mismatch_avg = ratio_mismatch_total / RUNS
    print("\nAverage ratio for the mismatch between f(x) and h(x) outside of the training data:")
    print("P(f(x)!=h(x)) = ", ratio_mismatch_avg)