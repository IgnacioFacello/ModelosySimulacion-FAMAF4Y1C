# Informe
Integrantes:
- Facello Ignacio
- Nieto Manuel

## Introduccion
Los métodos tradicionales de integración por Monte Carlo no siempre son los más eficientes.
El muestreo de importancia o Importance Sampling es una forma de hacer que las simulaciones de Monte Carlo converjan más rápido.
La idea se centra en elegir una distribución diferente para muestrear puntos que pueda generar datos mas relevantes.

'Porque funciona?' (((((((((((((Igna)))))))))))))

2 al 4 de la guia

'Encontrar funciones de importancia'
https://www.youtube.com/watch?v=C3p2wI4RAi8


En este proyecto buscamos comparar el desempeño del metodo de Importance Sampling para distintas funciones de importancia entre si y con respecto al metodo de Monte Carlo tradicional para estimar $P(Z > 3)$ con $Z\sim\mathcal N(0,1)$ y $P(X > 10)$ con $X\sim\mathcal \Gamma\left(9, \frac12\right)$

## Algoritmo
Se usaron las siguientes librerias: 
- Scipy (generar puntos), 
- numpy, 
- pyplot


Nos basamos en la implementacion del metodo de montecarlo de la guia teorica con ligeras modificaciones y el desarrollo hecho en la introduccion para implementar en codigo el metodo de importance sampling.

En una implementacion normal de montecarlo, se generan n valores de una variable uniforme continua $U\sim\mathcal U(0,1)$ y luego se evalua la expresion a estimar sobre ellos. El promedio de las evaluaciones va a tender al valor real de la expresion.

 Se genera una muestra de n puntos de una variable aleatoria $Y$ mediante `Y()`.

Luego evaluamos la expresion 

Sean:
- Y distribucion elegida para el sampling 
- g_Y la funcion de densisad para la distribucion Y
- f la funcion de densisad para la distribucion original
- h la transformacion de montecarlo

```python
def montecarlo_importance_sampling(nsim):
    integral = 0
    for _ in range(nsim):
        y # Variable aleatoria con dist Y
        integral += f(y) * h(y) / g_Y(y)
    return integral/nsim
```

## Resultados

## Conclusiones


