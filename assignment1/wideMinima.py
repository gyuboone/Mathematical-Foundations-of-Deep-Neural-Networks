import numpy as np
import matplotlib.pyplot as plt


np.seterr(invalid='ignore', over='ignore')  # suppress warning caused by division by inf

def f(x):
    return 1/(1 + np.exp(3*(x-3))) * 10 * x**2  + 1 / (1 + np.exp(-3*(x-3))) * (0.5*(x-10)**2 + 50)

def fprime(x):
    return 1 / (1 + np.exp((-3)*(x-3))) * (x-10) + 1/(1 + np.exp(3*(x-3))) * 20 * x + (3* np.exp(9))/(np.exp(9-1.5*x) + np.exp(1.5*x))**2 * ((0.5*(x-10)**2 + 50) - 10 * x**2) 

x = np.linspace(-5,20,100)

## plot all
fig, axs = plt.subplots(1, 4, figsize = (16,4))
for ax in axs:
    ax.plot(x,f(x), 'k')

max_iter = 400
alpha = [0.01 ,0.01, 0.3, 4]
color = ['g', 'r', 'g', 'g']

## fixing seed
np.random.seed(13)

for i in range(4):
    x = np.random.uniform(-5,20)
    x=1
    axs[i].set_title(f'alpha = {alpha[i]}')
    axs[i].scatter(x,f(x), alpha=1, c=color[i], label = f'initial_x= {x:.1f}')
    ## iteration (GD)
    for _ in range(max_iter):
        x += -alpha[i]*fprime(x)
        axs[i].scatter(x,f(x), alpha=0.4, c='b')
    axs[i].legend()

plt.savefig('GD.png')
plt.show()