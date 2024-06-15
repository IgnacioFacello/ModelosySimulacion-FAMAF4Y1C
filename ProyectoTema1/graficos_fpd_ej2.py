from scipy.stats import norm, uniform, expon, gamma
import numpy as np
import matplotlib.pyplot as plt
import pylab 

NORMAL_MU, NORMAL_SIGMA = 10, 1
EXP_LAMBDA = 1/11
GAMMA_ALFA, GAMMA_BETA = 10, 1

# Variables aleatorias
class Normal():
    def __init__(self, mu, sigma) -> None:
        self.mu = mu
        self.sigma = sigma
    
    def generador(self, size, ):
        return norm.rvs(size=size, loc=self.mu, scale=self.sigma)

    def funcion_densidad_probabilidad(self, y):            # mu media, sigma desv estandar
        return (2 * np.pi * (self.sigma ** 2)) ** (-1/2) * np.exp(- ((y - self.mu) ** 2) / 2 * (self.sigma ** 2))

class Exponential():
    def __init__(self, lamda=EXP_LAMBDA) -> None:
        self.lamda = lamda

    def generador(self, size):
        return expon.rvs(size=size, scale=1/self.lamda)

    def funcion_densidad_probabilidad(self, y):          
        return self.lamda * np.exp(-self.lamda * y)

def fact(n):
    return n * fact(n-1) if n >= 1 else 1

class Gamma():
    def __init__(self, alfa=GAMMA_ALFA, beta=GAMMA_BETA) -> None:
        self.alfa = alfa
        self.beta = beta
        pass
    
    def generador(self, size):
        return gamma.rvs(size=size, a=self.alfa, scale=self.beta)

    def funcion_densidad_probabilidad(self, y):           # alfa debe ser natural
        return 1/fact(self.alfa-1) * (self.beta ** (-self.alfa)) * (y ** (self.alfa - 1)) * np.exp(-y/self.beta)
    
def varianza_muestral(xs, media): 
    varianza = 0
    for x in xs:
        varianza += (x - media) ** 2
    return varianza/(len(xs) - 1)  

NORMAL_EST = Normal(0, 1)
NORMAL = Normal(NORMAL_MU, NORMAL_SIGMA)
EXPONENTIAL = Exponential(EXP_LAMBDA)
GAMMA = Gamma(GAMMA_ALFA, GAMMA_BETA)

xs_gt3 = np.arange(2,10,0.1)
real = [ norm.pdf(x) for x in xs_gt3 ]

fig, ax = plt.subplots(2,2, sharex='all', sharey='all')

ax[0][0].set_title('Control')
ax[0][0].plot(xs_gt3, [ uniform.pdf(1/x) for x in xs_gt3 ])
ax[0][0].plot(xs_gt3, real)

ax[0][1].set_title(f'Normal({NORMAL_MU}, {NORMAL_SIGMA})')
ax[0][1].plot(xs_gt3, [ NORMAL.funcion_densidad_probabilidad(x) for x in xs_gt3 ])
ax[0][1].plot(xs_gt3, real)

ax[1][0].set_title(f'Gamma({GAMMA_ALFA}, {GAMMA_BETA})')
ax[1][0].plot(xs_gt3, [ EXPONENTIAL.funcion_densidad_probabilidad(x) for x in xs_gt3 ])
ax[1][0].plot(xs_gt3, real)

ax[1][1].set_title(f'Exponencial({EXP_LAMBDA})')
ax[1][1].plot(xs_gt3, [ GAMMA.funcion_densidad_probabilidad(x) for x in xs_gt3 ])
ax[1][1].plot(xs_gt3, real)

plt.show()
