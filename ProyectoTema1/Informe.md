# Informe
Integrantes:
- Facello Ignacio
- Nieto Manuel

https://www.youtube.com/watch?v=C3p2wI4RAi8

## Introduccion
Los métodos tradicionales de integración por el metodo de Monte Carlo no siempre son los más eficientes.
El metodo de muestreo de importancia o Importance Sampling es una forma de hacer que las simulaciones de Monte Carlo converjan más rápido.
La idea de mismo se centra en elegir una distribución diferente a la uniforme para muestrear puntos que pueda generar datos mas relevantes.

Recordemos que Monte Carlo busca aproximar una expresion $\theta$ desconocida, sabiendo que este valor puede calcularse como $E[h(X)]$ para cierta variable aleatoria $X$ que posee una distribucion $F$. 
Luego la teoría detras de Importance Sampling se puede resumir en el siguiente resultado, teniendo en cuenta que $f_X$
representa la funcion de densidad de una variable $X$ y $g_Y$ la funcion de densidad una variable $Y$. Denominamos a $f_X$ como la funcion **objetivo** y a $g_Y$ como la funcion de **importancia**. 

$$
\begin{align}
\theta &= E[h(X)] \\
&= \int _{-\infty}^{\infty}h(t)f_X(t)dt \\
&= \int _{-\infty}^{\infty}h(t)f_X(t)\left(\frac{g_Y(t)}{g_Y(t)}\right)dt \\
&= \int _{-\infty}^{\infty}\frac{h(t)f_X(t)}{g_Y(t)}g_Y(t)dt\\
&= E\left[\frac{h(Y)f_X(Y)}{g_Y(Y)}\right]\\
\end{align}
$$

Cabe destacar que el metodo de Importance Sampling no siempre converge mas rapido que Monte Carlo y depende mucho de la funcion de importancia que se elija. 
Para lograr una mas rapida convergencia es importante elegir una distribucion $Y$ tal que $g_Y$ cumpla que $Var\left[\frac{h(Y)f_X(Y)}{g_Y(Y)}\right] < Var[h(X)].$ O coloquialmente, $g_Y(t)$ debe ser alto donde $|h(t)f_X(t)|$ sea alto. Por lo general no es sencillo encontrar una funcion $g_Y$ con esta propiedad.

El objetivo de este proyecto fue comparar el desempeño de Importance Sampling para distintas funciones de importancia entre si y con respecto a Monte Carlo tradicional para estimar $P(Z > 3)$ con $Z\sim\mathcal N(0,1)$ y $P(W > 10)$ con $W\sim\mathcal \Gamma\left(9, \frac12\right)$

## Algoritmo

Para la implementacion de este metodo nos basamos en la de Monte Carlo del apunte con ligeras modificaciones y en la justificacion teorica de Importance Sampling que desarrolamos en la introduccion.

Recordemos que para Monte Carlo, se genera una muestra de $n$ puntos de una variable uniforme continua $U\sim\mathcal U(0,1)$ y luego se evalua la expresion $h(t)f_X(t)$ para cada valor de la muestra. El promedio de las evaluaciones tiende a $\theta$.

Por otro lado, en nuestro algoritmo se genera una muestra de $n$ puntos de una variable aleatoria continua $Y$ y luego se evalua la expresion $\frac{h(t)f_X(t)}{g_Y(t)}$ para cada valor de la muestra. Por la justificacion teorica, el promedio de las evaluaciones sigue tendiendo a $\theta$.

A continuacion presentamos nuestro algoritmo de Importance Sampling en pseudocodigo. Sean:
- `Y()` un generador de variables $Y$   
- `g_Y` la funcion de densidad para la distribucion $Y$
- `f_X` la funcion de densidad para la distribucion original $X$
- `h` la transformacion de Monte Carlo

```python
def importance_sampling(nsim):
    integral = 0
    for _ in range(nsim):
        y = Y() # Variable aleatoria con dist Y
        integral += f(y) * h(y) / g_Y(y)
    return integral/nsim
```

## Resultados

Para comparar la velocidad de convergencia entre los metodos, se realizaron 50 estimaciones con distintas valores de $n$ equidistantes que van desde 10 mil a 500 mil con un incremento de 10 mil por estimacion.
Definimos una estimacion como: Ejecutar el algoritmo 10 veces para un mismo $n$ y tomar el promedio sobre los valores obtenidos.
En los siguientes graficos podemos observar la distancia entre dichas estimaciones y el valor real de $\theta$, calculado como el valor absoluto entre la diferencia de ambos. 

Cabe destacar que en ambos graficos, *Control* se refiere a la estimacion realizada utilizando el metodo de Monte Carlo estandar.

### Ejercicio 1

Para el inciso 1 buscamos aproximar $\theta = P(Z > 3)$ con $Z\sim\mathcal N(0,1)$. Tomamos las siguientes distribuciones para analizar el algoritmo de Importance Sampling:
- $Normal \sim \mathcal N(4, 1) $
- $Exponencial \sim \mathcal E(1/4)$
- $Gamma \sim \Gamma(4, 1)$

![[Ejercicio1_10.png]]

Observaciones:

| Caso        | Velocidad de Convergencia | Observaciones |
| ----------- | ------------------------- | ------------- |
| Control     | Media                     | Cercana al valor real y constante |
| Normal      | Lenta                     | Lenta pero en un caso tiene la menor distancia |
| Gamma       | Rapida                    | Comienza lejos y se acerca rapidamente al valor real |
| Exponencial | Lenta                     | Lejana al valor real y muy lenta |

### Ejercicio 2

Para el inciso 2 tenemos $\theta = P(W > 10)$ con $W\sim\mathcal \Gamma\left(9, \frac12\right)$. Las distribuciones que elegimos fueron:
- $Normal \sim \mathcal N(11, 1) $
- $Exponencial \sim \mathcal E(1/11)$
- $Gamma \sim \Gamma(11, 1)$

![[Ejercicio2_10.png]]

Observaciones:

| Caso        | Velocidad de Convergencia | Observaciones |
| ----------- | ------------------------- | ------------- |
| Control     | Lenta                     | Constante y lejana al valor real |
| Normal      | Rapida                    | Mas lenta que la Gamma, aunque es la que mas se acerco al valor real |
| Gamma       | Rapida                    | Comienza lejos del valor real pero converge muy rapidamente |
| Exponencial | Media                     | La distancia de las aproximaciones varia mucho a lo largo del experimento |

## Conclusiones (Falta de Hacer)
