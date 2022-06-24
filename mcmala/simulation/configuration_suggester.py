"""Abstract base class for configuration suggesters."""

from abc import ABC, abstractmethod

from mcmala.common.parallelizer import get_size, get_rank, barrier, get_comm


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
    def to_json(self):
        """
        Convert object into JSON seriazable content.

        Returns
        -------
        info : dict
            Info that can be saved to a dict.
        """
        pass

    @staticmethod
    def collect_configuration(local_configuration):
        barrier()
        if get_size() == 1:
            return local_configuration
        else:
            positions = get_comm().bcast(local_configuration.get_positions(), root=0)
            local_configuration.set_positions(positions)
            return local_configuration
