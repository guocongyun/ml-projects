import numpy as np
import matplotlib.pyplot as plt

es = np.random.rand(100000, 2)
e = np.min(es, axis=1) # axis=1 means compare horizontally
plt.hist(e)

# IMPORTANT, random.rand draw from unifrom distribution between 0,1
# random.normal draw from normal distribution of mean=0, var=1