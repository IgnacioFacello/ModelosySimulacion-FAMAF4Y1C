# %%
from scipy.stats import norm, uniform, expon, gamma
import numpy as np
import matplotlib.pyplot as plt
import pylab 
import time

# ===================================== Auxiliares ===================================================

# Constantes
EQUID = np.arange(10_000, 500_001, 10_000)  # Lista de 50 numeros equidistantes entre 0 y 500.000
REAL = 1 - norm.cdf(3)                      # Valor de P(X >= 3) con X ~ N(0,1)
NORMAL_MU, NORMAL_SIGMA = 4, 1
EXP_LAMBDA = 1/4
GAMMA_ALFA, GAMMA_BETA = 4, 1
SEED = 1567
TIMES = 50

np.random.seed(SEED)

# Variables aleatorias
class Normal():
    def generate(mu, sigma, size):
        return [ n for n in norm.rvs(size=size, loc=mu, scale=sigma) ]

    def dens_prob(mu, sigma, y):            # mu media, sigma desv estandar
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

    def dens_prob(alfa, beta, y):           # alfa debe ser natural
        return 1/fact(alfa-1) * (beta ** (-alfa)) * (y ** (alfa - 1)) * np.exp(-y/beta)

# ===================================== Ejercicio A ===================================================

def ejercicio1a(n):
    ''' Metodo de Monte Carlo en [0,1] para calcular P(X >= 3) con X ~ N(0,1)
    '''
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
    acc = 0
    ys = fun_gen(n)                         # Muestra de n variables Y
    for y in ys:
        acc += Normal.dens_prob(0, 1, y) * indicadora(y) / fun_imp(y)
    return acc/n

# ===================================== Graficas ===================================================

T1 = time.perf_counter()

# Funcion para generar aproximaciones
def muestras(gen):
    ''' Genera las 50 aproximaciones en el intervalo [0, 500.000]
    ''' 
    t1 = time.perf_counter()
    C = 10_000
    nsim = 0
    acc = {}                    
    acc[nsim] = 0               
    for i in range(1, 51): # Del 10 mil a 500 mil
        anterior, nsim = (i-1) * C, i * C
        aux = 0
        for _ in range(TIMES):
            aux += gen(C) * C
        aux = aux / TIMES
        acc[nsim] = acc[anterior] + aux
    acc.pop(0)
    acc = { k: v/k for k,v in acc.items() }
    t2 = time.perf_counter()
    return acc, t2-t1

# Generacion de aproximaciones
# Montecarlo
print('Generando muestra usando Montecarlo Estandar... ')
mtc = lambda x : ejercicio1a(x)
acc_mtc, t = muestras(mtc)
print(f'Completado en {t} segundos')

# Importance Sampling: Normal
print(f'Generando muestra usando IS: Normal {NORMAL_MU}, {NORMAL_SIGMA}...')
impI = lambda x : ejercicio1b(n=x, 
            fun_imp=lambda y : Normal.dens_prob(NORMAL_MU, NORMAL_SIGMA, y), 
            fun_gen=lambda n : Normal.generate(NORMAL_MU, NORMAL_SIGMA, n)
            )
acc_impI, t = muestras(impI)
print(f'Completado en {t} segundos')

# Importance Sampling: Exponencial
print(f'Generando muestra usando IS: Exponencial {EXP_LAMBDA}...')
impII = lambda x: ejercicio1b(n=x,
    fun_imp=lambda y : Exponential.dens_prob(EXP_LAMBDA, y),
    fun_gen=lambda n : Exponential.generate(EXP_LAMBDA, n))
acc_impII, t = muestras(impII)
print(f'Completado en {t} segundos')

# Importance Sampling: Gamma
print(f'Generando muestra usando IS: Gamma {GAMMA_ALFA}, {GAMMA_BETA}...')
impIII = lambda x : ejercicio1b(n=x,
    fun_imp=lambda y : Gamma.dens_prob(GAMMA_ALFA, GAMMA_BETA, y), 
    fun_gen=lambda n : Gamma.generate(GAMMA_ALFA, GAMMA_BETA, n)
    )
acc_impIII, t = muestras(impIII)
print(f'Completado en {t} segundos')

T2 = time.perf_counter()
print(f'Timepo: {T2-T1} segundos')

# Grafica Comparativa LINEAS
plt.figure(figsize=(20,4), layout='tight')
plt.xticks(EQUID, rotation=45)
plt.yscale('log')
plt.xlabel('Nro de Muestras')
plt.ylabel('Distancia')

accs = {
    'Control': { k: np.abs(v - REAL) for k, v in acc_mtc.items() },
    'Normal': { k: np.abs(v - REAL) for k, v in acc_impI.items() },
    'Exponencial': { k: np.abs(v - REAL) for k, v in acc_impII.items() },
    'Gamma': { k: np.abs(v - REAL) for k, v in acc_impIII.items() },
}

for k, v in accs.items():
    plt.plot(v.keys(), v.values(), label=k)

plt.grid()
plt.legend()
#pylab.show()
plt.savefig('Ejercicio1.png')