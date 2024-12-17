from typing import Tuple, Callable, Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from evp.evaluator import EvolvabilityOfPopulationEvaluator


def calculate_evaluator_score(get_evaluator: Callable[[], EvolvabilityOfPopulationEvaluator], number_of_runs: int = 5):
    best_fitnesses = np.zeros(number_of_runs)
    convergence_speeds = np.zeros(number_of_runs)

    for i in range(number_of_runs):
        evaluator = get_evaluator()
        evaluator.evaluate()
        best_fitnesses[i] = evaluator.get_best_fitness()
        convergence_speeds[i] = evaluator.get_convergence_speed()

    return pd.DataFrame({'best_fitness': np.amin(best_fitnesses),
                         'mean_best_fitness': [np.mean(best_fitnesses)],
                         'std_best_fitness': [np.std(best_fitnesses)],
                         'mean_convergence_speed': [np.mean(convergence_speeds)],
                         'std_convergence_speed': [np.std(convergence_speeds)]})


def plot_function(function_name: str, function: Callable[[np.array], float], lb_: float, ub_: float):
    x = np.linspace(lb_, ub_, 400)
    y = np.linspace(lb_, ub_, 400)
    X, Y = np.meshgrid(x, y)

    # Calculate the function values
    Z = np.array([function([xi, yi]) for xi, yi in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)

    # Plot the 3D surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(X, Y, Z, cmap='viridis')

    # Add a color bar which maps values to colors
    fig.colorbar(surface)

    # Set plot labels and title
    ax.set_title(function_name)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')

    plt.show()
