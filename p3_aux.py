from random import random

def von_neumann(s):
    return (s**2 // 100) % 10_000

def cong(y,a,c,m):
    return ( a*y + c ) % m

def montecarlo_01(g, nsim):
    integral = 0
    for _ in range(nsim):
        integral += g(random())
    return integral/nsim

def montecarlo_ab(g, a, b, nsim):
    integral = 0
    for _ in range(nsim):
        integral += g(a + (b-a) * random())
    return integral * (b-a)/nsim
