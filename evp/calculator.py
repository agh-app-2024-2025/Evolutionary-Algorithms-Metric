import copy

import numpy as np

from evp.algos.algorithm_interface import AlgorithmInterface


class EvolvabilityOfPopulationCalculator:

    def __init__(self, optimization_algorithm: AlgorithmInterface, max_functions_evaluations: int,
                 number_of_neighbours: int):
        dimension = optimization_algorithm.get_dimension()
        number_of_generations = int(max_functions_evaluations / optimization_algorithm.get_population_size())

        self.optimization_algorithm = optimization_algorithm

        self.number_of_neighbours = number_of_neighbours

        self.parents = np.zeros((number_of_generations, dimension))
        self.neighbours = np.zeros((number_of_generations, number_of_neighbours, dimension))
        self.neighbours_best_fitness = np.zeros((number_of_generations, number_of_neighbours))
        self.parent_best_fitness = np.zeros(number_of_generations)
        self.parent_fitness_std = np.zeros(number_of_generations)

        self.epp_values = np.zeros(number_of_generations)
        self.eap_values = np.zeros(number_of_generations)
        self.evp_values = np.zeros(number_of_generations)

        self.diversity = np.zeros(number_of_generations)

        self.current_generation = 0

    def add_generation(self, X: list, fit: list):
        self.evolve_neighbours(X, fit)
        self.save_best_parent(X, fit)
        self.save_diversity(X)
        self.current_generation += 1

    def calculate_metrics(self):
        parent_best_fitness_repeated = np.tile(self.parent_best_fitness, (self.number_of_neighbours, 1)).T
        improvement_matrix = self.neighbours_best_fitness < parent_best_fitness_repeated
        improvement_counts = np.sum(improvement_matrix, axis=1)

        self.calculate_epp_values(improvement_counts)
        self.calculate_eap_values(parent_best_fitness_repeated, improvement_matrix, improvement_counts)
        self.calculate_evp_values()

    def calculate_metrics_for_generation(self, X: list, fit: list):
        self.add_generation(X, fit)

        parent_best_fitness_repeated = np.tile(self.parent_best_fitness[self.current_generation - 1],
                                               self.number_of_neighbours)
        improvement_matrix = self.neighbours_best_fitness[self.current_generation - 1] < parent_best_fitness_repeated
        improvement_counts = np.sum(improvement_matrix)

        epp_value = improvement_counts / self.number_of_neighbours
        eap_value = self.calculate_eap_value(parent_best_fitness_repeated, improvement_matrix, improvement_counts)
        evp_value = epp_value * eap_value

        self.epp_values[self.current_generation - 1] = epp_value
        self.eap_values[self.current_generation - 1] = eap_value
        self.evp_values[self.current_generation - 1] = evp_value

        return epp_value, eap_value, evp_value

    def calculate_eap_value(self, parent_best_fitness_repeated, improvement_matrix, improvement_counts: int):
        if improvement_counts >= 1:
            numerator = np.sum(
                np.absolute(parent_best_fitness_repeated[improvement_matrix] - self.neighbours_best_fitness[
                    self.current_generation - 1, improvement_matrix]) / np.maximum(
                    self.optimization_algorithm.get_population_size() * self.parent_fitness_std[
                        self.current_generation - 1], 1e-8))
            denominator = improvement_counts
            return numerator / denominator
        return 0

    def calculate_epp_values(self, improvement_counts):
        self.epp_values = improvement_counts / self.number_of_neighbours

    def calculate_eap_values(self, parent_best_fitness_repeated, improvement_matrix, improvement_counts):
        generations_with_improvement = np.where(improvement_counts >= 1)[0]

        for generation in generations_with_improvement:
            improved_neighbours = improvement_matrix[generation, :]
            numerator = np.sum(np.absolute(
                parent_best_fitness_repeated[generation, improved_neighbours] -
                self.neighbours_best_fitness[generation, improved_neighbours]) / np.maximum(
                self.optimization_algorithm.get_population_size() * self.parent_fitness_std[generation], 1e-8))
            denominator = improvement_counts[generation]
            self.eap_values[generation] = numerator / denominator

    def calculate_evp_values(self):
        self.evp_values = self.epp_values * self.eap_values

    def evolve_neighbours(self, X: list, fit: list):
        for neighbour_number in range(self.number_of_neighbours):
            optimization_algorithm_for_neighbour = copy.deepcopy(self.optimization_algorithm)
            np.random.seed(np.random.randint(0, 2 ** 32))

            optimization_algorithm_for_neighbour.tell(X, fit)
            X_offspring, fit_offspring = optimization_algorithm_for_neighbour.ask_and_eval()

            best_index = np.argmin(fit_offspring)
            self.neighbours[self.current_generation, neighbour_number] = X_offspring[best_index]
            self.neighbours_best_fitness[self.current_generation, neighbour_number] = fit_offspring[best_index]

    def save_best_parent(self, X: list, fit: list):
        best_index = np.argmin(fit)
        self.parents[self.current_generation] = X[best_index]
        self.parent_best_fitness[self.current_generation] = fit[best_index]
        self.parent_fitness_std[self.current_generation] = np.std(fit)

    def save_diversity(self, X: list):
        centroid = np.mean(X, axis=0)
        self.diversity[self.current_generation] = np.mean(np.linalg.norm(np.array(X) - centroid, axis=1))
