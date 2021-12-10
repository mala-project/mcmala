from abc import ABC, abstractmethod


class ConfigurationSuggester(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def suggest_new_configuration(self, old_configuration):
        pass
