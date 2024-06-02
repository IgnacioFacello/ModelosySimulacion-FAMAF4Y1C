# Informe
Integrantes:
- Facello Ignacio
- Nieto Manuel

## Introduccion
'''
Importance Sampling - Idea Resumida
Los métodos tradicionales de integración por Monte Carlo no siempre son los más eficientes.
El muestreo de importancia o Importance Sampling es una forma de hacer que las simulaciones de Monte Carlo converjan más rápido.
La idea se centra en elegir una distribución diferente para muestrear puntos que pueda generar datos mas relevantes.

'Porque funciona'
2 al 4 de la guia

'Encontrar funciones de importancia'
https://www.youtube.com/watch?v=C3p2wI4RAi8


'''


En este proyecto buscamos comparar el desempeño del metodo de Importance Sampling para distintas funciones de importancia entre si y con respecto al metodo de Monte Carlo tradicional para estimar $P(X > 3)$ con $X\sim\mathcal N(0,1)$ y $P(Y > 10)$ con $Y\sim\mathcal \Gamma\left(9, \frac12\right)$

## Algoritmo
'''
Librerias: Scipy (generar puntos), numpy, pyplot
Algoritmo de montecarlo de la guia
Para importance sampling modificamos el algoritmo de montecarlo para que divida g por la funcion de importancia y genere los valores correctos
'''
## Resultados

## Conclusiones


