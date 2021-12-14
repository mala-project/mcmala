from typing import List

from mcmala import Evaluator, ConfigurationSuggester
from .markovchain import MarkovChain


class MonteCarloSimulation:
    def __init__(self, temperatureK, evaluator: Evaluator,
                 configuration_suggester: ConfigurationSuggester,
                 initial_configuration, number_markov_chains,
                 calculate_observables_after_steps=1):
        """
        Represent a full Monte Carlo simulation.

        This includes an arbitrary numbrer of Markov Chains.

        Parameters
        ----------
        temperatureK : float
            Temperature in Kelvin.

        evaluator : mcmala.simulation.evaluator.Evaluator
            Evaluator object used to calculate the total energy.

        configuration_suggester : mcmala.simulation.configuration_suggester.ConfigurationSuggester
            Suggests changes to configuration of MarkovChain. Depends on the
            type of model.

        initial_configuration : Any
            Initial configuration of atoms, spin grid, etc.

        number_markov_chains : int
            Number of Markov chains to time evolve and then average over.

        calculate_observables_after_steps : int
            Controls after how many steps the obervables are calculated.
            Does not apply to the energy, which is calculated at any step
            for obvious reasons.
        """
        self.temperatureK = temperatureK
        self.evaluator = evaluator
        self.configuration_suggester = configuration_suggester
        self.configuration = initial_configuration
        self.calculate_observables_after_steps = \
            calculate_observables_after_steps
        self.number_markov_chains = number_markov_chains

        # Create a list of Markov chains.
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
        """
        Run all the Markov chains in this simulations.

        Parameters
        ----------
        steps_to_evolve : int
            Number of steps to run the simulation for.

        print_energies : bool
            If True, the energies are printed at each step of the simulation.
        """
        # Currently, only serial is implemented.
        for i in range(0, self.number_markov_chains):
            self.markov_chains[i].run(steps_to_evolve,
                                      print_energies=print_energies)

        # Average over observables.
        for i in range(0, self.number_markov_chains):
            self.energy += self.markov_chains[i].averaged_energy

        self.energy /= self.number_markov_chains
