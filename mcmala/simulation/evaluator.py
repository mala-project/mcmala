from abc import ABC, abstractmethod


class Evaluator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_total_energy(self, configuration):
        pass
