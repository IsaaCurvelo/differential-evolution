import numpy as np
from scipy.optimize import rosen



def create_population(fitness_function, lwr_bnd, upp_bnd, n, d):
    population = np.random.random((n, d))
    population = lwr_bnd + population * (upp_bnd - lwr_bnd)
    fitnesses = np.apply_along_axis(fitness_function, 1, population)
    return population, fitnesses




def de(fitness_function, lwr_bnd, upp_bnd, F, n=50, d=30,  iterations=500,
       pc = 0.8, strategy = 1):
    
    population, fitnesses = create_population(fitness_function, lwr_bnd, 
                                              upp_bnd, n, d)
    new_population = np.zeros((n, d))
    
    for t in range(iterations):
        print(min(fitnesses))
        if strategy == 1:
            idx = np.reshape(np.random.choice(range(n), n * 3), (n, 3))
        else:
            idx = np.reshape(np.random.choice(range(n), n * 2), (n, 2))
            best = np.repeat(np.argmin(fitnesses), n)
            idx = np.concatenate((idx, best[:, None]), axis = 1)
                
        v = population[idx[:, 0], :] + F * (population[idx[:, 1], :] - 
                      population[idx[:, 2], :])

        for i in range(n):
            idx = np.where(v[i, :] < lwr_bnd)
            v[i, idx] = lwr_bnd[idx]
            idx = np.where(v[i, :] > upp_bnd)
            v[i, idx] = upp_bnd[idx]
        
        r = np.random.random((n, d))
        idx = r < pc
        new_population[idx] = v[idx]
        idx = np.logical_not(idx)
        new_population[idx] = population[idx]
        
        new_fitnesses = np.apply_along_axis(fitness_function, 1, 
                                            new_population)
        idx = new_fitnesses < fitnesses
        fitnesses[idx] = new_fitnesses[idx]
        population[idx, :] = new_population[idx, :]
    
    return population, fitnesses
        

dimensions = 30
#np.random.seed(0)
lwr_bnd = np.repeat(-5, dimensions)
upp_bnd = np.repeat(5, dimensions)
F = 0.6

import time
res = list()
tmp = list()
for r in range(30):
    start = time.time()
    p, ft = de(rosen, lwr_bnd, upp_bnd, F, n=50, d=dimensions, iterations= 1000)
    tmp.append(time.time() - start)
    res.append(min(ft))
