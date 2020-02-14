import numpy as np
from scipy.optimize import rosen



def create_population(fitness_function, lwr_bnd, upp_bnd, n, d):
    population = np.random.random((n, d))
    population = lwr_bnd + population * (upp_bnd - lwr_bnd)
    fitnesses = np.apply_along_axis(fitness_function, 1, population)
    return population, fitnesses




def de(fitness_function, lwr_bnd, upp_bnd, F, n = 5, d = 30,  iterations = 500,
       pc = 0.6, strategy = 1):
    
    population, fitnesses = create_population(fitness_function, lwr_bnd, 
                                              upp_bnd, n, d)
    new_population = np.zeros((n, d))
    new_fitnesses = np.zeros(n)
    
    for t in range(iterations):
        
        if strategy == 1:
            idx = np.reshape(np.random.choice(range(n), n * 3), (n, 3))
        else:
            idx = np.reshape(np.random.choice(range(n), n * 2), (n, 2))
            best = np.repeat(np.argmin(fitnesses), n)
            idx = np.concatenate((idx, best[:, None]), axis = 1)
        
        v = population[idx[:, 0], :] + F * (population[idx[:, 1], :] - 
                      population[idx[:, 2], :])
        print(v)

dimensions = 30
np.random.seed(1)
lwr_bnd = np.repeat(-5, dimensions)
upp_bnd = np.repeat(5, dimensions)
F = 0.6
de(rosen, lwr_bnd, upp_bnd, F, n=30, iterations=1, strategy=2 )