# Evolvability-of-Population
## Description
Evolvability-of-Population is a Python project focused on implementing Evolvability of Population metric from [Population Evolvability: Dynamic Fitness Landscape Analysis for Population-Based Metaheuristic Algorithms](https://ieeexplore.ieee.org/document/8016373) article and evaluating various evolutionary algorithms.

## Project Structure
- **evp/**: Contains the main code for the project.
  - **algos/**: Contains the implementations of the optimization algorithms.
    - `algorithm_interface.py`: Defines the interface for optimization algorithms.
    - `cmaes.py`: Implements the CMA-ES algorithm.
    - `GA.py`: Implements the Genetic Algorithm.
  - `calculator.py`: Contains the `EvolvabilityOfPopulationCalculator` class for calculating evolvability metrics.
  - `evaluator.py`: Contains the `EvolvabilityOfPopulationEvaluator` class for evaluating the evolvability of populations.
  - `plotter.py`: Contains functions for plotting metrics and visualizations.
  - **funcs/**: Contains benchmark functions used for optimization.
    - `radar.py`: Contains the `radar_function` used as a benchmark.
  - `strategy.py`: Contains the `EvolvabilityOfPopulationStrategy` class for running the optimization strategy.
- **notebooks/**: Contains Jupyter notebooks for running experiments and visualizing results.
  - `cma-es.ipynb`: Notebook for running and visualizing CMA-ES experiments.
  - `genetic_algorithm.ipynb`: Notebook for running and visualizing Genetic Algorithm experiments.
  - **results/**: Directory for storing results of experiments.
- `requirements.txt`: Lists the dependencies required for the project.

## Installation
To install the required dependencies, you can use `pip` or `poetry`.

### Using pip
```sh
pip install -r requirements.txt
```

### Using poetry
```sh
poetry install
```

### Using docker
```sh
docker build -t eop:0.0.1 .
docker run -it --rm -p 8888:8888 -v .:/home/jovyan/work eop:0.0.1
```

## Usage
You can run the provided Jupyter notebooks to see the implementations and evaluations of the algorithms:
- [cma-es.ipynb](notebooks/cma-es.ipynb)
- [genetic_algorithm.ipynb](notebooks/genetic_algorithm.ipynb)



## Project Dependencies
Key dependencies include:
- numpy
- cma
- matplotlib
- pandas
- river
- scikit-learn
- tqdm
- ioh

## Authors
- Michal Strzezon <mstrzezon@gmail.com>
- Pawel Magnuszewski <pawel.magnu@gmail.com>
- Tomasz Turek <t.turek2000@gmail.com>