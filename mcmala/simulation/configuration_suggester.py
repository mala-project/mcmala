"""Abstract base class for configuration suggesters."""

from abc import ABC, abstractmethod


class ConfigurationSuggester(ABC):
    """Abstract base class for configuration suggesters."""

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

    @abstractmethod
    def get_info(self):
        """
        Access a dictionary with identifying information.

        Returns
        -------
        info : dict

        """
