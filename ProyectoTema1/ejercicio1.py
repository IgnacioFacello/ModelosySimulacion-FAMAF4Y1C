import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import random as rnd

REAL = 0.0013

f = lambda y : (2*scp.pi)**(-1/2) * scp.exp(-(((1/y) + 2)**2)/2) * y**(-2)

def ejercicio1a(n):
    ''' Metodo de montecarlo en [0,1]
    '''
    acc = 0
    for _ in range(n):
        acc += f(rnd.random())
    return acc/n

equid = np.arange(0,500_001,10_000)[1:]
acc = {}
for i in equid:
    acc[i] = ejercicio1a(i)
    print(f"n: {i:7} val: {acc[i]:.2}")

plt.plot([0,500_000],[REAL,REAL])
plt.plot(acc.keys, acc.values)
plt.show()