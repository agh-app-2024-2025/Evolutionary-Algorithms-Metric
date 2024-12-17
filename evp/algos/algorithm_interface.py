from abc import ABC, abstractmethod
from typing import Tuple


class AlgorithmInterface(ABC):
    @abstractmethod
    def ask_and_eval(self) -> Tuple[list, list]:
        pass

    @abstractmethod
    def tell(self, solutions, function_values):
        pass

    @abstractmethod
    def restart(self):
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        pass

    @abstractmethod
    def get_population_size(self) -> int:
        pass