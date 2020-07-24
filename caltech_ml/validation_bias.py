import numpy as np
import matplotlib.pyplot as plt
import sympy

es = np.random.rand(100000, 2)
e = np.min(es, axis=1) # axis=1 means compare horizontally
plt.hist(e)

print(sympy.solve("1/3 * (1 + 4/(1-x)**2 + 4/(1+x)**2 ) - 1/2","x"))

# IMPORTANT, random.rand draw from unifrom distribution between 0,1
# random.normal draw from normal distribution of mean=0, var=1