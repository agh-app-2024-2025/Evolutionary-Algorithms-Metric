import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def plot_neighbours_dimensional(parent: np.array, neighbours: np.array, generation_number: int,
                                number_of_dimensions: int):
    pca = PCA(n_components=number_of_dimensions)
    transformed = pca.fit_transform(np.vstack([parent, *neighbours]))

    fig = plt.figure(figsize=(16, 9))
    projection_type = {2: 'rectilinear', 3: '3d'}.get(number_of_dimensions, None)
    if projection_type is None:
        raise ValueError('Number of dimensions must be either 2 or 3')

    ax = fig.add_subplot(111, projection=projection_type)
    ax.scatter(*transformed[0, :number_of_dimensions], color='blue', label='Parent')
    for i in range(neighbours.shape[0]):
        ax.scatter(*transformed[i + 1, :number_of_dimensions], color='orange', label=f'Neighbour {i + 1}')

    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    if number_of_dimensions == 3:
        ax.set_zlabel('PC3', fontsize=12)
    ax.set_title(f'Parent and its Neighbours at {generation_number}th Generation')
    ax.legend()
    plt.show()


def plot_neighbours_parallel(parent: np.array, neighbours: np.array, generation_number: int):
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    ax.plot(parent, color='blue', linestyle='-', linewidth=2, label='Parent')
    for i, neighbour in enumerate(neighbours):
        ax.plot(neighbour, color='orange', linestyle='--', linewidth=1, label=f'Neighbour {i + 1}')
    ax.set_xlabel('Dimension', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'Parent and Neighbours at Generation {generation_number}')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()


def smooth_values(epp_values: np.array, eap_values: np.array, evp_values: np.array, window_size: int = 5):
    epp_values_smooth = pd.Series(epp_values).rolling(window=window_size, min_periods=1).mean().to_numpy()
    eap_values_smooth = pd.Series(eap_values).rolling(window=window_size, min_periods=1).mean().to_numpy()
    evp_values_smooth = pd.Series(evp_values).rolling(window=window_size, min_periods=1).mean().to_numpy()

    return epp_values_smooth, eap_values_smooth, evp_values_smooth


def plot_metrics(function_evaluations: np.array, epp_values: np.array, eap_values: np.array,
                 evp_values: np.array,
                 best_fitness: np.array, diversity: np.array, drifts: np.array = None):
    epp_values_smooth, eap_values_smooth, evp_values_smooth = smooth_values(epp_values, eap_values, evp_values)

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(16, 15))

    ax1.semilogy(function_evaluations, epp_values_smooth, color='blue', linestyle='-', linewidth=2)
    ax1.set_ylim(1e-5, 1.1)
    ax1.set_ylabel(r'$log(epp)$', fontsize=12)
    ax1.set_title('Evolutionary Probability of Population')
    ax1.grid(True)

    ax2.semilogy(function_evaluations, eap_values_smooth, color='orange', linestyle='--', linewidth=2)
    ax2.set_ylim(1e-5, 1.1)
    ax2.set_ylabel(r'$log(eap)$', fontsize=12)
    ax2.set_title('Evolutionary Ability of Population')
    ax2.grid(True)

    ax3.semilogy(function_evaluations, evp_values_smooth, color='green', linestyle=':', linewidth=2)
    ax3.set_ylim(1e-5, 1.1)
    ax3.set_ylabel(r'$log(evp)$', fontsize=12)
    ax3.set_xlabel('Function Evaluations (FEs)', fontsize=12)
    ax3.set_title('Evolvability of Population')
    ax3.grid(True)
    if drifts is not None:
        for drift in drifts:
            ax3.axvline(x=drift, color='red', linestyle='--', linewidth=1)

    ax4.semilogy(function_evaluations, best_fitness, color='purple', linestyle='-', linewidth=2)
    ax4.set_ylabel(r'$log(best_fitness)$', fontsize=12)
    ax4.set_title('Best Fitness Function Value')
    ax4.grid(True)

    ax5.semilogy(function_evaluations, diversity, color='brown', linestyle='-', linewidth=2)
    ax5.set_ylabel(r'$log(diversity)', fontsize=12)
    ax5.set_xlabel('Function Evaluations (FEs)', fontsize=12)
    ax5.set_title('Diversity (Average Distance to Centroid)')
    ax5.grid(True)

    plt.tight_layout()
    plt.show()
