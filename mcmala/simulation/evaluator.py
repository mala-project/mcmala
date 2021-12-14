from abc import ABC, abstractmethod

"""
Abstract base class for evaluators.
"""


class Evaluator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_total_energy(self, configuration):
        """
        Calculate the total energy of a given configuration.

        Parameters
        ----------
        configuration : Any
            Configuration based on which the total energy will be calculated.

        Returns
        -------
        total_energy : float
            Total energy of the given configuration.
        """
        pass
