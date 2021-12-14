from typing import List

from mcmala import Evaluator, ConfigurationSuggester
from .markovchain import MarkovChain


class MonteCarloSimulation:
    def __init__(self, temperatureK, evaluator: Evaluator,
                 configuration_suggester: ConfigurationSuggester,
                 initial_configuration, number_markov_chains,
                 calculate_observables_after_steps=1):
        self.temperatureK = temperatureK
        self.evaluator = evaluator
        self.configuration_suggester = configuration_suggester
        self.configuration = initial_configuration
        self.calculate_observables_after_steps = \
            calculate_observables_after_steps
        self.number_markov_chains = number_markov_chains
        self.markov_chains: List[MarkovChain] = []
        for i in range(0, self.number_markov_chains):
            self.markov_chains.\
                append(MarkovChain(self.temperatureK, self.evaluator,
                        self.configuration_suggester, initial_configuration,
                        calculate_observables_after_steps=\
                            calculate_observables_after_steps))

        # Observables.
        self.energy = 0.0

    def run(self, steps_to_evolve, print_energies=False):
        # Currently, only serial is implemented.
        for i in range(0, self.number_markov_chains):
            self.markov_chains[i].run(steps_to_evolve,
                                      print_energies=print_energies)

        # Average over observables.
        for i in range(0, self.number_markov_chains):
            self.energy += self.markov_chains[i].averaged_energy

        self.energy /= self.number_markov_chains
