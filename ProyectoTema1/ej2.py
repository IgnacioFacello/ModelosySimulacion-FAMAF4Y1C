from scipy.stats import norm, uniform, expon, gamma
import numpy as np
import matplotlib.pyplot as plt

# ===================================== Auxiliares ===================================================

# Constantes
EQUID = np.arange(10_000, 500_001, 10_000)  # Lista de 50 numeros equidistantes entre 0 y 500.000
REAL = 1 - gamma(a=9, scale=0.5).cdf(10)    # Valor de P(S >= 10) con S ~ Gamma(9, 1/2)
NORMAL_MU, NORMAL_SIGMA = 11, 1
EXP_LAMBDA = 1/11
GAMMA_ALFA, GAMMA_BETA = 11, 1
SEED = 1234

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

def ejercicio2a(n):
    ''' Metodo de Monte Carlo en [0,1] para calcular P(S >= 10) con S ~ Gamma(9, 1/2)
    '''
    np.random.seed(SEED)
    acc = 0
    ys = uniform.rvs(size=n)                # Muestra de n Uniformes U(0,1)
    for y in ys:
        acc += Gamma.dens_prob(9, 1/2, 1/y + 9)/(y**2)
    return acc/n

# ===================================== Ejercicio B ===================================================

def indicadora(y):
    ''' Funcion Indicadora en (10, inf)
    '''
    if (y >= 10):
        return 1
    else:
        return 0

def ejercicio2b(n, fun_imp, fun_gen):       # Agregamos la entrada fun_gen para generalizar
    ''' Metodo Importance Sampling para calcular P(S >= 10) con S ~ Gamma(9, 1/2)
    '''
    np.random.seed(SEED)
    acc = 0
    ys = fun_gen(n)                         # Muestra de n variables Y
    for y in ys:
        acc += Gamma.dens_prob(9, 1/2, y) * indicadora(y) / fun_imp(y)
    return acc/n

# ===================================== Graficas ===================================================

# Generacion de aproximaciones
# Montecarlo
print('Generando muestra usando Montecarlo Estandar... ')
acc_mtc = {}
for i in EQUID:
    aux = ejercicio2a(i)
    acc_mtc[i] = round(aux, 6)
print('Completado')

# Funcion para generar aproximaciones
def muestras(dens, gen):
    ''' Genera las 50 aproximaciones en el intervalo [0, 500.000]
    ''' 
    acc = {}
    for i in EQUID:
        aux = ejercicio2b(n=i, fun_imp=dens, fun_gen=gen)
        acc[i] = round(aux, 6)
    return acc    

# Importance Sampling: Normal
print(f'Generando muestra usando IS: Normal {NORMAL_MU}, {NORMAL_SIGMA}...')
acc_impI = muestras(
    lambda y : Normal.dens_prob(NORMAL_MU, NORMAL_SIGMA, y), 
    lambda n : Normal.generate(NORMAL_MU, NORMAL_SIGMA, n)
    )
print('Completado')

# Importance Sampling: Exponencial
print(f'Generando muestra usando IS: Exponencial {EXP_LAMBDA}...')
acc_impII = muestras(
    lambda y : Exponential.dens_prob(EXP_LAMBDA, y),
    lambda n : Exponential.generate(EXP_LAMBDA, n)
    )
print('Completado')

# Importance Sampling: Gamma
print(f'Generando muestra usando IS: Gamma {GAMMA_ALFA}, {GAMMA_BETA}...')
acc_impIII = muestras(
    lambda y : Gamma.dens_prob(GAMMA_ALFA, GAMMA_BETA, y), 
    lambda n : Gamma.generate(GAMMA_ALFA, GAMMA_BETA, n)
    )
print('Completado')

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