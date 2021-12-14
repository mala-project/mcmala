from abc import ABC, abstractmethod

"""
Abstract base class for configuration suggesters.
"""


class ConfigurationSuggester(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def suggest_new_configuration(self, old_configuration):
        """
        Suggest a new configuration based on the old.

        Parameters
        ----------
        old_configuration: Any
            Old configuration based on which a new one will be suggested.

        Returns
        -------
        new_configuration: Any
            New configuration as suggested by algorithm.

        """
        pass
