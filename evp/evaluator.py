import numpy as np
from tqdm import tqdm

from evp.algos.algorithm_interface import AlgorithmInterface
from evp.calculator import EvolvabilityOfPopulationCalculator
from evp.plotter import plot_metrics, plot_neighbours_parallel, plot_neighbours_dimensional


class EvolvabilityOfPopulationEvaluator:
    def __init__(self, optimization_algorithm: AlgorithmInterface, max_functions_evaluations: int = 10000,
                 number_of_neighbours: int = 5):
        self.optimization_algorithm = optimization_algorithm
        self.max_functions_evaluations = max_functions_evaluations
        self.number_of_neighbours = number_of_neighbours

        self.current_generation = 0

        self.calculator = EvolvabilityOfPopulationCalculator(optimization_algorithm, max_functions_evaluations,
                                                             number_of_neighbours)

    def evaluate(self, with_plot: bool = False):
        number_of_generations = int(self.max_functions_evaluations / self.optimization_algorithm.get_population_size())

        for _ in tqdm(range(number_of_generations)):
            X, fit = self.optimization_algorithm.ask_and_eval()
            self.calculator.add_generation(X, fit)
            self.optimization_algorithm.tell(X, fit)
            self.current_generation += 1

        self.calculator.calculate_metrics()

        if with_plot:
            self.plot_metrics()

    def plot_metrics(self):
        function_evaluations_per_generation = np.arange(self.optimization_algorithm.get_population_size(),
                                                        self.max_functions_evaluations + 1,
                                                        self.optimization_algorithm.get_population_size())
        plot_metrics(function_evaluations_per_generation, self.calculator.epp_values, self.calculator.eap_values,
                     self.calculator.evp_values,
                     self.calculator.parent_best_fitness, self.calculator.diversity)

    def plot_neighbours_parallel(self, number_of_generation: int):
        plot_neighbours_parallel(self.calculator.parents[number_of_generation], self.calculator.neighbours[number_of_generation], number_of_generation)

    def plot_neighbours_dimensional(self, number_of_generation: int, dimension: int = 2):
        plot_neighbours_dimensional(self.calculator.parents[number_of_generation], self.calculator.neighbours[number_of_generation], number_of_generation, dimension)

    def get_best_fitness(self):
        return np.amin(self.calculator.parent_best_fitness)

    def get_convergence_speed(self):
        return self.max_functions_evaluations
