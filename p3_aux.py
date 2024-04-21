import random as rnd

def von_neumann(s):
    ''' Sequencia de von neumann
    '''
    return (s**2 // 100) % 10_000

def cong(y,a,c,m):
    ''' Generador de numeros congruencial
    '''
    return ( a*y + c ) % m

def montecarlo_ab(g, a, b, nsim):
    ''' Funcion para el metodo de montecarlo en el intervalo [a, b]
    '''
    integral = 0
    for _ in range(nsim):
        integral += g(a + (b-a)*rnd.random())
    return integral * (b-a)/nsim

def montecarlo_01(g, nsim):
    ''' Funcion para el metodo de montecarlo en el intervalo [0, 1]
    '''
    integral = 0
    for _ in range(nsim):
        integral += g(rnd.random())
    return integral/nsim

# integral Monte Carlo en el intervalo (0,inf)
def montecarlo_inf(fun, nsim):
    ''' Funcion para el metodo de montecarlo en el intervalo [0, inf]
    '''
    integral=0
    for _ in range(nsim):
        u=rnd.random()
        integral+= fun(1/u-1)/(u**2)
    return integral/nsim


# integrales múltiples, 2 variables

def montecarlo_01_2(fun, nsim):
    ''' integral Monte Carlo en el intervalo (0,1)x(0,1)
    '''
    integral = 0
    for _ in range(nsim):
        integral += fun(rnd.random(), rnd.random())
    return integral/nsim

def montecarlo_ab_2(fun,a,b,c,d, nsim):
    ''' integral Monte Carlo en el intervalo (a,b)x(c,d)
    '''
    integral = 0
    for _ in range(nsim):
        integral += fun(a + (b-a)*rnd.random(), c + (d-c)*rnd.random())
    return integral*(b-a)*(d-c)/nsim

def montecarlo_inf_2(g, nsim):
    ''' integral Monte Carlo en el intervalo (0,inf)x(0,inf)
    '''
    integral=0
    for _ in range(nsim):
        u1=rnd.random()
        u2=rnd.random()
        integral+= g(1/u1-1, 1/u2-1)/((u1**2)*(u2**2))
    return integral/nsim
