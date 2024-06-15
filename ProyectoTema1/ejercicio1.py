from scipy.stats import norm, uniform, expon, gamma
import numpy as np
import matplotlib.pyplot as plt
import pylab 
import time

# ===================================== Auxiliares ===================================================

# Constantes
STEP = 10_000
EQUID = np.arange(10_000, 500_001, STEP) # Lista de 50 numeros equidistantes entre 0 y 500.000
REAL = 1 - norm.cdf(3)                      # Valor de P(X >= 3) con X ~ N(0,1)
SEED = 1567
TIMES_TO_AVERAGE = 10
NORMAL_MU, NORMAL_SIGMA = 4, 1
EXP_LAMBDA = 1
GAMMA_ALFA, GAMMA_BETA = 4, 1

np.random.seed(SEED)

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

# ===================================== Ejercicio A ===================================================

def ejercicio1a(n):
    ''' Calcular la varianza del estimador para n valores
    '''
    acc = 0
    ys = uniform.rvs(size=n)                # Muestra de n Uniformes U(0,1)
    xs = []
    for y in ys:
        x = NORMAL_EST.funcion_densidad_probabilidad(1/y + 2)/(y**2)
        acc += x
        xs.append(x)
    media = acc/n
    varianza = varianza_muestral(xs, media)
    return media, varianza

# ===================================== Ejercicio B ===================================================

def importance_sampling_valores(n, X_fdp, ind, Y):
    ''' Muestras para el metodo de Importance Sampling
    '''
    ys = Y.generador(n)                         # Muestra de n variables aleatorias Y
    rs = []
    for y in ys:
        x =  X_fdp(y) * ind(y) / Y.funcion_densidad_probabilidad(y)
        rs.append(x)
    return rs

def indicadora(y):
    ''' Funcion Indicadora en (3, inf)
    '''
    if (y >= 3):
        return 1
    else:
        return 0

def ejercicio1b(n, Y):
    ''' Calcular la varianza del estimador para n valores
    '''
    valores = importance_sampling_valores(
            n,
            lambda y : NORMAL_EST.funcion_densidad_probabilidad(y),
            indicadora,
            Y,
        )
    media = np.mean(valores)
    varianza = varianza_muestral(valores, media)
    return media, varianza

# ===================================== Muestras ===================================================

def medir_tiempo(fun, nombre):
    print(f'Generando muestra usando {nombre}... ')
    t1 = time.perf_counter()
    res = fun()
    t2 = time.perf_counter()
    print(f'Completado en {round(t2-t1,0)} segundos')
    return res

# Funcion para generar aproximaciones
def muestras(gen):
    ''' Genera las 50 aproximaciones en el intervalo [0, 500.000]
    ''' 
    nsim = 0
    acc1 = { 0:0 }
    for i in range(1, 51 ): # Del 10 mil a 500 mil
        anterior, nsim = (i-1) * STEP, i * STEP
        media = 0
        for _ in range(TIMES_TO_AVERAGE):
            aux = gen(STEP)
            media += aux * STEP
        media = media / TIMES_TO_AVERAGE
        acc1[nsim] = acc1[anterior] + media
    acc1.pop(0)
    acc1 = { k: v/k for k,v in acc1.items() }
    return acc1

T1 = time.perf_counter()

# Generacion de aproximaciones
# Montecarlo
mtc = lambda x : ejercicio1a(x)[0]
acc1_mtc = medir_tiempo(
    lambda : muestras(mtc), 
    "Montecarlo Estandar"
    )

# Importance Sampling: Normal
impI = lambda x : ejercicio1b(n=x, Y=NORMAL)[0]
acc1_impI = medir_tiempo(
    lambda : muestras(impI), 
    f"IS: Normal {round(NORMAL_MU, 4)}, {round(NORMAL_SIGMA, 4)}"
    )

# Importance Sampling: Exponencial
impII = lambda x: ejercicio1b(n=x, Y=EXPONENTIAL)[0]
acc1_impII = medir_tiempo(
    lambda : muestras(impII), 
    f"IS: Exponencial {round(EXP_LAMBDA, 4)}"
    )

# Importance Sampling: Gamma
impIII = lambda x : ejercicio1b(n=x, Y=GAMMA)[0]
acc1_impIII = medir_tiempo(
    lambda : muestras(impIII), 
    f"IS: Gamma {round(GAMMA_ALFA, 4)}, {round(GAMMA_BETA, 4)}"
    )

T2 = time.perf_counter()
print(f'Tiempo total: {round(T2-T1,0)} segundos')

# ===================================== Graficas ===================================================
# Grafica Comparativa Media VARIANZAS

nsim_var = 500
varianzas = {
    'Control': ejercicio1a(nsim_var)[1],
    f'Normal({NORMAL_MU}, {NORMAL_SIGMA})': ejercicio1b(n=nsim_var, Y=NORMAL)[1],
    f'Exponencial({EXP_LAMBDA})': ejercicio1b(n=nsim_var, Y=EXPONENTIAL)[1],
    f'Gamma({GAMMA_ALFA}, {GAMMA_BETA})': ejercicio1b(n=nsim_var, Y=GAMMA)[1],
}

plt.xticks(EQUID, rotation=45)
plt.yscale('log')
plt.xlabel('Distribucion')
plt.ylabel('Varianza')

print('\nVarianzas')
for i, (k, v) in enumerate(varianzas.items()):
    print(f'\t{k}: {v}')
    plt.bar(k, v, label=k)

plt.title("Varianzas1")
plt.legend()
pylab.show()

# Grafica Comparativa Media LINEAS
aproximaciones = {
    'Control': { k: np.abs(v - REAL) for k, v in acc1_mtc.items() },
    f'Normal({NORMAL_MU}, {NORMAL_SIGMA})': { k: np.abs(v - REAL) for k, v in acc1_impI.items() },
    f'Exponencial({EXP_LAMBDA})': { k: np.abs(v - REAL) for k, v in acc1_impII.items() },
    f'Gamma({GAMMA_ALFA}, {GAMMA_BETA})': { k: np.abs(v - REAL) for k, v in acc1_impIII.items() },
}

plt.figure(figsize=(20,4), layout='tight')
plt.xticks(EQUID, rotation=45)
plt.yscale('log')
plt.xlabel('Tamaño muestra')
plt.ylabel('Distancia')

for k, v in aproximaciones.items():
    plt.plot(v.keys(), v.values(), label=k)
    
plt.title("Ejercicio1")
plt.grid()
plt.legend()
pylab.show()
