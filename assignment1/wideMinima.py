import numpy as np
import matplotlib.pyplot as plt


np.seterr(invalid='ignore', over='ignore')  # suppress warning caused by division by inf

def f(x):
    return 1/(1 + np.exp(3*(x-3))) * 10 * x**2  + 1 / (1 + np.exp(-3*(x-3))) * (0.5*(x-10)**2 + 50)

def fprime(x):
    return 1 / (1 + np.exp((-3)*(x-3))) * (x-10) + 1/(1 + np.exp(3*(x-3))) * 20 * x + (3* np.exp(9))/(np.exp(9-1.5*x) + np.exp(1.5*x))**2 * ((0.5*(x-10)**2 + 50) - 10 * x**2) 

x = np.linspace(-5,20,100)
plt.plot(x,f(x), 'k')

## initial point
x = np.random.uniform(-5,20)
# x = -1      # specify if you want
print('initial point :', f'{x:.4f}')

## iteration
max_iter = 400
alpha = 0.01          # or 0.3 and 4
for i in range(max_iter):
    plt.scatter(x, f(x), alpha=0.4, c='b')
    x += -alpha*fprime(x)

plt.show()