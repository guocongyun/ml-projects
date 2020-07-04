import random
import matplotlib.pyplot as plt
import numpy as np

def random_point():
    x0, y0 = random.uniform(-1, 1), random.uniform(-1, 1)
    return (x0, y0)

class Dataset:
    def target_func(self, p):
        if self.target_a*p[0] + self.target_b > p[1]:
            return -1
        else:
            return 1
        
    def __init__(self, num_points):
        p0 = random_point()
        p1 = random_point()
        self.target_a = (p1[1] - p0[1]) / (p1[0] - p0[0])
        self.target_b = p0[1] - self.target_a * p0[0]
        
        self.xs = []
        self.ys = []
        for i in range(num_points):
            xn = random_point()
            self.xs.append(xn)
            self.ys.append(self.target_func(xn))
            
    def plot(self):
        cs = ["red" if y > 0 else "blue" for y in self.ys]
        plt.scatter([x[0] for x in self.xs], [x[1] for x in self.xs], c=cs)
        plt.plot((-1, 1), 
                 (-self.target_a+self.target_b, self.target_a+self.target_b))
        plt.gca().set_aspect(1)
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.title("Target function")
        plt.show()
        
class PLA:
    def candidate_func(self, p):
        return int(np.sign(self.w[0]*1 + self.w[1]*p[0] + self.w[2]*p[1]))
        
    def __init__(self, dataset):
        self.w = np.array([0, 0, 0])
        self.dataset = dataset
        
    def fit(self, plot_iters=False):
        self.w = np.array([0, 0, 0])
        num_iters = 0
        
        while True:
            misclassified_points = []
            for (x, y) in zip(self.dataset.xs, self.dataset.ys):
                if self.candidate_func(x) != y:
                    misclassified_points.append((np.array([1, x[0], x[1]]), y))
            if len(misclassified_points) > 0:
                print(len(misclassified_points))
                num_iters += 1
                x, y = random.choice(misclassified_points)
                self.w = self.w + y*x
                if plot_iters:
                    self.plot()
            else:
                return num_iters
        
    def plot(self):
        cs = ["red" if y > 0 else "blue" for y in self.dataset.ys]
        plt.scatter([x[0] for x in self.dataset.xs], [x[1] for x in self.dataset.xs], c=cs)
        y_left = (self.w[1] - self.w[0]) / self.w[2]
        y_right = (-self.w[1] - self.w[0]) / self.w[2]
        plt.plot((-1,1), (y_left, y_right))
        plt.gca().set_aspect(1)
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.title("Candidate function found with PLA")
        plt.show()
def main():
    ds = Dataset(num_points=10)


    pla = PLA(dataset=ds)
    pla.fit(plot_iters=False)
    print(    pla.fit(plot_iters=False))
    ds.plot()   

if __name__ == "__main__":
    main()
