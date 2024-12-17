import numpy as np
import pandas as pd
from river.drift import KSWIN
from tqdm import tqdm

from evp.algos.algorithm_interface import AlgorithmInterface
from evp.calculator import EvolvabilityOfPopulationCalculator
from evp.plotter import plot_metrics, plot_neighbours_parallel, plot_neighbours_dimensional


class EvolvabilityOfPopulationStrategy:

    def __init__(self, optimization_algorithm: AlgorithmInterface, max_functions_evaluations: int = 10000,
                 number_of_neighbours: int = 5):
        self.optimization_algorithm = optimization_algorithm
        self.max_functions_evaluations = max_functions_evaluations
        self.number_of_neighbours = number_of_neighbours

        self.current_generation = 0

        self.calculator = EvolvabilityOfPopulationCalculator(optimization_algorithm, max_functions_evaluations,
                                                             number_of_neighbours)

        self.evp_drifts = []

    def evaluate(self):
        drift_detector = KSWIN(alpha=0.0001)

        number_of_generations = int(self.max_functions_evaluations / self.optimization_algorithm.get_population_size())

        for _ in tqdm(range(number_of_generations)):
            X, fit = self.optimization_algorithm.ask_and_eval()
            evp = self.calculator.calculate_metrics_for_generation(X, fit)[2]
            self.optimization_algorithm.tell(X, fit)

            drift_detector.update(evp)

            if drift_detector.drift_detected:
                #     log drift
                self.evp_drifts.append(self.current_generation)
                self.optimization_algorithm.restart()

            self.current_generation += 1

    # TODO: move to abstract class with evaluator methods
    def plot_metrics(self, with_drifts: bool = True):
        function_evaluations_per_generation = np.arange(self.optimization_algorithm.get_population_size(),
                                                        self.max_functions_evaluations + 1,
                                                        self.optimization_algorithm.get_population_size())

        drifts = [drift * self.optimization_algorithm.get_population_size() for drift in self.evp_drifts] if with_drifts else None

        plot_metrics(function_evaluations_per_generation, self.calculator.epp_values, self.calculator.eap_values,
                     self.calculator.evp_values,
                     self.calculator.parent_best_fitness, self.calculator.diversity, drifts)

    def plot_neighbours_parallel(self, number_of_generation: int):
        plot_neighbours_parallel(self.calculator.parents[number_of_generation],
                                 self.calculator.neighbours[number_of_generation], number_of_generation)

    def plot_neighbours_dimensional(self, number_of_generation: int, dimension: int = 2):
        plot_neighbours_dimensional(self.calculator.parents[number_of_generation],
                                    self.calculator.neighbours[number_of_generation], number_of_generation, dimension)

    def score(self):
        best_fitnesses_between_detected_drifts = []
        drifts = self.evp_drifts
        if len(drifts) == 0:
            return pd.DataFrame({'best_fitness': np.amin(self.calculator.parent_best_fitness),
                                 'mean_best_fitness': np.amin(self.calculator.parent_best_fitness),
                                 'std_best_fitness': 0,
                                 'mean_functions_evaluations': self.max_functions_evaluations,
                                 'std_functions_evaluations': 0},
                                index=[0])
        drifts.insert(0, 0)
        for i in range(1, len(drifts)):
            best_fitnesses_between_detected_drifts.append(
                np.amin(self.calculator.parent_best_fitness[drifts[i - 1]:drifts[i]]))

        functions_evaluations_between_detected_drifts = np.diff(drifts) * self.optimization_algorithm.get_population_size()

        return pd.DataFrame({'best_fitness': np.amin(best_fitnesses_between_detected_drifts),
                             'mean_best_fitness': np.mean(best_fitnesses_between_detected_drifts),
                             'std_best_fitness': np.std(best_fitnesses_between_detected_drifts),
                             'mean_functions_evaluations': np.mean(functions_evaluations_between_detected_drifts),
                             'std_functions_evaluations': np.std(functions_evaluations_between_detected_drifts)},
                            index=[0])
