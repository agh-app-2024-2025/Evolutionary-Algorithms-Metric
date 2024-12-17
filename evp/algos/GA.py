import numpy as np

from evp.algos.algorithm_interface import AlgorithmInterface

#
# class GA(AlgorithmInterface):
#     def __init__(self, func, dim, bounds, population_size, mutation_rate, crossover_rate, elite_ratio):
#         self.ga = GeneticAlgorithm(func, bounds, population_size, mutation_rate, crossover_rate, elite_ratio, dim)
#
#     def ask_and_eval(self):
#         return self.ga.ask_and_eval(self.ga.objective_function)
#
#     def tell(self, solutions, function_values):
#         self.ga.tell(solutions, function_values)
#
#     def restart(self):
#         self.ga.reset()
#
#     def get_dimension(self) -> int:
#         return self.ga.dimensions
#
#     def get_population_size(self) -> int:
#         return self.ga.population_size

class GeneticAlgorithm:
    def __init__(self, func, dim, mutation_rate, crossover_rate, elite_ratio, bounds, population_size):
        self.func = func

        self.dim = dim
        self.bounds = bounds
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio

        self.population = np.random.uniform(bounds[0], bounds[1], (population_size, dim))
        self.fitness = np.apply_along_axis(func, 1, self.population)
        self.countiter = 0

    def rank_fitness_scaling(self):
        ranks = np.argsort(np.argsort(self.fitness))
        scaled_fitness = ranks / (self.population_size - 1)
        return scaled_fitness

    def stochastic_universal_sampling(self, scaled_fitness):
        pointers = np.linspace(0, 1, self.population_size, endpoint=False) + np.random.uniform(0, 1 / self.population_size)
        selected_indices = []
        cumulative_fitness = np.cumsum(scaled_fitness)
        for pointer in pointers:
            selected_indices.append(np.searchsorted(cumulative_fitness, pointer))
        return self.population[selected_indices]

    def whole_arithmetic_crossover(self, parent1, parent2):
        alpha = np.random.uniform(0, 1, self.dim)
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1
        return child1, child2

    def uniform_mutation(self, child):
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                child[i] = np.random.uniform(self.bounds[0], self.bounds[1])
        return child

    def absorb_bounds(self, offspring):
        return np.clip(offspring, self.bounds[0], self.bounds[1])

    def ask(self):
        scaled_fitness = self.rank_fitness_scaling()
        parents = self.stochastic_universal_sampling(scaled_fitness)

        offspring = []
        for i in range(0, self.population_size, 2):
            p1, p2 = parents[i], parents[(i+1) % self.population_size]
            if np.random.rand() < self.crossover_rate:
                child1, child2 = self.whole_arithmetic_crossover(p1, p2)
            else:
                child1, child2 = p1, p2
            child1 = self.uniform_mutation(child1)
            child2 = self.uniform_mutation(child2)
            offspring.extend([child1, child2])

        offspring = np.array(offspring)
        offspring = self.absorb_bounds(offspring)
        return offspring

    def ask_and_eval(self):
        offspring = self.ask()
        offspring_fitness = np.apply_along_axis(self.func, 1, offspring)
        return offspring, offspring_fitness

    def tell(self, offspring, offspring_fitness):
        elite_count = int(self.elite_ratio * self.population_size)
        elite_indices = np.argsort(self.fitness)[:elite_count]
        elite_individuals = self.population[elite_indices]
        elite_fitness = self.fitness[elite_indices]

        combined_population = np.vstack((elite_individuals, offspring))
        combined_fitness = np.hstack((elite_fitness, offspring_fitness))

        best_indices = np.argsort(combined_fitness)[:self.population_size]
        self.population = combined_population[best_indices]
        self.fitness = combined_fitness[best_indices]
        self.countiter += 1

    # def best_solution(self):
    #     best_idx = np.argmin(self.fitness)
    #     return self.population[best_idx], self.fitness[best_idx]

    def restart(self):
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        self.fitness = np.apply_along_axis(self.func, 1, self.population)
        self.countiter = 0

    def get_dimension(self) -> int:
        return self.dim

    def get_population_size(self) -> int:
        return self.population_size
