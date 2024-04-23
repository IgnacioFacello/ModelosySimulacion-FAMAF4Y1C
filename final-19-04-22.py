import numpy as np
import random as rnd

def montecarlo_inf(fun, nsim):
    ''' Funcion para el metodo de montecarlo en el intervalo [0, inf]
    '''
    integral=0
    for _ in range(nsim):
        u=rnd.random()
        integral+= fun(1/u-1)/(u**2)
    return integral/nsim

def montecarlo_01_2(fun, nsim):
    ''' integral Monte Carlo en el intervalo (0,1)x(0,1)
    '''
    integral = 0
    for _ in range(nsim):
        integral += fun(rnd.random(), rnd.random())
    return integral/nsim

f = lambda x,y: 1-np.exp(-(x+y))

print('Integral a')
for e in [3,4,5,6]:
    mu = montecarlo_01_2(f, 10**e)
    print(f'{10**e:7} | {mu}')
print('Valor exacto: e**(−2)⋅(2e−1) ~= 0.6\n')

g = lambda x: ((x+1)**2)*np.exp(-(x+1)**2)

print('Integral b')
for e in [3,4,5,6]:
    mu = montecarlo_inf(g, 10**e)
    print(f'{10**e:7} | {mu}')
print(f'Valor exacto: ~0.2536')
