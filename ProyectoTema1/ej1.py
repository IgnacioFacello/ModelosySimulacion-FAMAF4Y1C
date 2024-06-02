# %%
from scipy.stats import norm, uniform, expon, gamma
import numpy as np
import matplotlib.pyplot as plt

# ===================================== Auxiliares ===================================================

EQUID = np.arange(10_000, 500_001, 10_000)  # Lista de 50 numeros equidistantes entre 0 y 500.000
REAL = 1 - norm.cdf(3)                      # Valor de P(X >= 3) con X ~ N(0,1)

# Funciones
def muestras(dens, gen):
    ''' Genera las 50 aproximaciones en el intervalo [0, 500.000]
    ''' 
    acc = {}
    for i in EQUID:
        aux = ejercicio1b(n=i, fun_imp=dens, fun_gen=gen)
        acc[i] = round(aux,7)
    return acc    

# Variables aleatorias
class Normal():
    def generate(mu, sigma, size):
        return [ n for n in norm.rvs(size=size, loc=mu, scale=sigma) ]

    def dens_prob(mu, sigma, y):      # mu media, sigma desv estandar
        return (2 * np.pi * (sigma ** 2)) ** (-1/2) * np.exp(- ((y - mu) ** 2) / 2 * (sigma ** 2))

class Exponential():
    def generate(lamda, size):
        return [ e for e in expon.rvs(size=size, scale=1/lamda) ]

    def dens_prob(lamda, y):          
        return lamda * np.exp(-lamda * y)

def fact(n):
    return n * fact(n-1) if n >= 1 else 1

class Gamma():
    def generate(alfa, beta, size):
        return [ g for g in gamma.rvs(size=size, a=alfa, scale=beta) ]

    def dens_prob(alfa, beta, y):     # alfa debe ser natural
        return 1/fact(alfa-1) * (beta ** (-alfa)) * (y ** (alfa - 1)) * np.exp(-y/beta)

# ===================================== Ejercicio A ===================================================

def ejercicio1a(n):
    ''' Metodo de Monte Carlo en [0,1] para calcular P(X >= 3) con X ~ N(0,1)
    '''
    np.random.seed(1234)
    acc = 0
    ys = uniform.rvs(size=n)                # Muestra de n Uniformes U(0,1)
    for y in ys:
        acc += Normal.dens_prob(0, 1, 1/y + 2)/(y**2)
    return acc/n

# ===================================== Ejercicio B ===================================================

def indicadora(y):
    ''' Funcion Indicadora en (3, inf)
    '''
    if (y >= 3):
        return 1
    else:
        return 0

def ejercicio1b(n, fun_imp, fun_gen):       # Agregamos la entrada fun_gen para generalizar
    ''' Metodo Importance Sampling para calcular P(X >= 3) con X ~ N(0,1)
    '''
    np.random.seed(1234)
    acc = 0
    ys = fun_gen(n)                         # Muestra de n variables Y
    for y in ys:
        acc += Normal.dens_prob(0, 1, y) * indicadora(y) / fun_imp(y)
    return acc/n

# ===================================== Graficas ===================================================

## Generacion de aproximaciones
# %%
# Montecarlo
print('Generando muestra usando Montecarlo Estandard... ')
acc_mtc = {}
for i in EQUID:
    aux = ejercicio1a(i)
    acc_mtc[i] = round(aux,6)
print('Completado')

# %%
# Importance Sampling: Normal
print('Generando muestra usando IS: Normal 4, 1...')
acc_impI = muestras(
    lambda y : Normal.dens_prob(4, 1, y), 
    lambda n : Normal.generate(4, 1, n)
    )
print('Completado')

# %%
# Importance Sampling: Exponencial
print('Generando muestra usando IS: Exponencial 1/4...')
acc_impII = muestras(
    lambda y : Exponential.dens_prob(1/4, y),
    lambda n : Exponential.generate(1/4, n)
    )
print('Completado')

# %%
# Importance Sampling: Gamma
print('Generando muestra usando IS: Gamma 3, 2...')
acc_impIII = muestras(
    lambda y : Gamma.dens_prob(3, 3, y), 
    lambda n : Gamma.generate(3, 3, n)
    )
print('Completado')

# %%
# Grafica Comparativa
plt.figure(figsize=(20,8))
plt.xticks(EQUID, rotation=45)
plt.plot([10_000,500_000],[REAL,REAL], label='Real')

accs = {
    'Control': acc_mtc,
    'Imp I': acc_impI,
    'Imp II': acc_impII,
    'Imp III': acc_impIII,
}

for k, v in accs.items():
    plt.plot(v.keys(), v.values(), label=k)

plt.legend()
plt.show()

# %%
# Grafica Comparativa
plt.figure(figsize=(20,8))
plt.xticks(EQUID, rotation=45)
# plt.plot([10_000,500_000],[REAL,REAL], label='Real')

accs = {
    'Control': { k: abs(y - REAL) for k, y in acc_mtc.items() },
    'Imp I': { k: abs(y - REAL) for k, y in acc_impI.items() },
    'Imp II': { k: abs(y - REAL) for k, y in acc_impII.items() },
    'Imp III': { k: abs(y - REAL) for k, y in acc_impIII.items() },
}

for k, v in accs.items():
    plt.plot(v.keys(), v.values(), label=k)

plt.legend()
plt.show()
# %%
