# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, expon, gamma
from random import random, seed

EQUID = np.arange(10_000, 500_001, 10_000) # Lista de 50 numeros equidistantes entre 0 y 500.000
REAL = 1 - norm.cdf(3) # 1 - P(X <= 3)

# Densidad de X Normal estandar
norm_dens_std = lambda x : (2*np.pi)**(-1/2) * np.exp(-((x)**2)/2)

# Densidad de X Normal estandar (Modificada para el metodo de montecarlo)
norm_dens_std_mod = lambda x : (2*np.pi)**(-1/2) * np.exp(-(((1/x) + 2)**2)/2) * x**(-2)

def ejercicio1a(n):
    ''' Aproximacion usando el metodo de montecarlo en [0,1]
    '''
    acc = 0
    ys = uniform.rvs(size=n)
    for y in ys:
        acc += norm_dens_std_mod(y)
    return acc/n

# ============= ======   =========  =======       ==========  ==========    ====  ======= ==    ======  ====================
def indicadora(y):
    return 1 if y >= 3 else 0

def ejercicio1b(n, fun_imp, fun_gen):
    ''' Metodo sampling de importancia para calcular P(X > 3)
    '''
    acc = 0
    ys = fun_gen(n)
    for y in ys:
        acc += norm_dens_std(y) * indicadora(y) / fun_imp(y) 
    return acc/n

def generarNormal_rechazo(): # Genera una Normal Estandar N(0,1)
    while (True):
        Y1 = - np.log(1 - random()) # Exponencial Lambda 1
        Y2 = - np.log(1 - random()) # Exponencial Lambda 1
        Y = Y2 - (((Y1 - 1) ** 2) / 2)
        if (Y > 0):
            if (random() <= 1/2):
                return Y1
            else:
                return -Y1

# Generadores de Y

def normal_gen(mu, sigma, size):
    # return [ sigma * n + mu for n in norm.rvs(size=size) ]
    return [ generarNormal_rechazo() * sigma + mu for _ in range(size) ]

def exponencial(lamda):
    U = 1 - random()
    return -np.log(U) / lamda
    
def exp_gen(lam, size):
    # return [ n for n in expon.rvs(size=size, scale=1/lam) ]
    return [ exponencial(lam) for _ in range(size) ]

def gen_gamma(k, u): # k natural y u real > 0 
    U = 1
    for _ in range(k):
        U *= 1 - random()
    return - np.log(U) * u

def gamma_gen(alfa, beta, size):
    # return [ n for n in gamma.rvs(size=size, a=alfa, scale=beta)]
    return [ gen_gamma(alfa, beta) for _ in range(size) ]

# Densidad de distribuciones
def normal_dens(mu, sigma, y): #mu media, sigma desv estandar
    return (2 * np.pi * (sigma ** 2)) ** (-1/2) * np.exp(-((y - mu) ** 2) / 2 * (sigma ** 2))

def exp_dens(lamda, y): #1/mu media
    return lamda * np.exp(- lamda * y)

def fact(n):
    return n * fact(n-1) if n >= 1 else 1

def gamma_dens(alfa, beta, y): # alfa debe ser natural
    return 1/fact(alfa-1) * (beta ** (-alfa)) * (y ** (alfa - 1)) * np.exp(-y/beta)

# Funcion de densidad fx/gy simplificada particulares
opt_I = lambda y : np.exp(-4*y + 8)                             # norm_4_1
opt_II = lambda y : 4 * np.exp((- 2 * (y ** 2) + y) / 4)        # exp_025
opt_III = lambda y : (32 / (y ** 2)) * np.exp((y - y ** 2) / 2) # gamma_3_2

def muestras(dens, gen): 
    acc = {}
    for i in EQUID:
        aux = ejercicio1b(n=i, 
                        fun_imp=dens, 
                        fun_gen=gen
        )
        acc[i] = round(aux,7)
    return acc

# %%
# Muestras generadas usando el metodo de montecarlo
acc_mtc = {}
for i in EQUID:
    aux = ejercicio1a(i)
    acc_mtc[i] = round(aux,6)

# %%
# Muestras con el metodo de Sampling de importancia y la funcion de importancia I
acc_impI = muestras(
    lambda y : normal_dens(4, 1, y), 
    lambda n : normal_gen(4, 1, n)
    )

# %%
acc_impII = muestras(
    lambda y : exp_dens(1/4, y),
    lambda n : exp_gen(1/4, n)
    )

# %%
acc_impIII = muestras(
    lambda y : gamma_dens(3, 2, y), 
    lambda n : gamma_gen(3, 2, n)
    )

# %%
# Graficas
def plot_aproximation(acc, name):
    plt.plot(acc.keys(), acc.values(), label=name)

plt.figure(figsize=(20,8))
plt.xticks(EQUID, rotation=45)
plt.plot([1000,500_000],[REAL,REAL], label='Real')

plot_aproximation(acc_mtc, 'Control')
plot_aproximation(acc_impI, 'Imp I')
plot_aproximation(acc_impII, 'Imp II')
plot_aproximation(acc_impIII, 'Imp III')

plt.legend()
plt.show()

# %%
