from scipy import optimize
import math
import matplotlib.pyplot as plt

def growth_function(N, d):
    if N >= d:
        return N**d
    elif N < d:
        return 2**N

def vc_bound (N, d=50, delta=0.05):
    return math.sqrt( 8/N * math.log(4*growth_function(2*N, d) / delta))

def radermacher_penalty_bound(N, d=50, delta=0.05):
    return math.sqrt(2 * math.log(2*N*growth_function(N, d))/N) + math.sqrt(2/N * math.log(1/delta)) + 1/N

def parrondo_and_van_den_broek(N, d=50, delta=0.05):
    fx = lambda eps : math.sqrt((2*eps + math.log(6*growth_function(2*N, d)/delta))/N) - eps
    # return optimize.brentq(fx,0,1)
    return optimize.brentq(fx,0,5)

def devroye(N, d=50, delta=0.05):
    # fx = lambda eps : math.sqrt((4*eps*(1+eps) + math.log(4/delta) + 2*d*math.log(N))/(2*N)) - eps
    fx = lambda eps : math.sqrt((4*eps*(1+eps) + math.log(4/delta) + (N**2)*math.log(2))/(2*N)) - eps
    # return optimize.brentq(fx,0,1)
    return optimize.brentq(fx,0,5)

xs = range(3, 10, 1)
ys_vs = [vc_bound(N) for N in xs]
ys_radermacher = [radermacher_penalty_bound(N) for N in xs]
ys_parrondo = [parrondo_and_van_den_broek(N) for N in xs]
ys_devroye = [devroye(N) for N in xs]

figl = plt.figure(2, dpi=80)
plt.plot(xs, ys_vs, 'ro-', label = 'VC')
plt.plot(xs, ys_radermacher, 'go-', label = 'radermacher')
plt.plot(xs, ys_parrondo, 'bo-', label = 'parrondo')
plt.plot(xs, ys_devroye, 'co-', label = 'devroye')

plt.xlabel(r'sample size $N$') # IMPORTANT r refix means to treat the string as raw literal, ignoring all \n
plt.ylabel(r'$\epsilon$ bound')
plt.legend()
plt.show()


