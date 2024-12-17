from typing import Tuple

import cma
import numpy as np

from evp.algos.algorithm_interface import AlgorithmInterface

class CMAES(AlgorithmInterface):
    def __init__(self, func, dim, sigma0, bounds, population_size):
        self.cma_evolution_strategy = cma.CMAEvolutionStrategy(np.zeros(dim), sigma0, options={'bounds': bounds, 'popsize': population_size})
        self.func = func

        self.dim = dim
        self.sigma0 = sigma0
        self.bounds = bounds
        self.population_size = population_size

    def ask_and_eval(self) -> Tuple[list, list]:
        return self.cma_evolution_strategy.ask_and_eval(self.func)

    def tell(self, solutions, function_values):
        self.cma_evolution_strategy.tell(solutions, function_values)

    def restart(self):
        self.cma_evolution_strategy = cma.CMAEvolutionStrategy(np.zeros(self.dim), self.sigma0, options={'bounds': self.bounds, 'popsize': self.population_size})

    def get_dimension(self) -> int:
        return self.dim

    def get_population_size(self) -> int:
        return self.population_size

