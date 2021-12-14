"""
Describes a single Markov chain.
"""
from random import random

from ase import Atoms
import numpy as np

from mcmala import Evaluator, ConfigurationSuggester

# Boltzmann constant in atomic units.
boltzmannConstant = 8.617333262e-5


class MarkovChain:

    def __init__(self, temperatureK, evaluator: Evaluator,
                 configuration_suggester: ConfigurationSuggester,
                 initial_configuration, calculate_observables_after_steps=1):
        self.temperatureK = temperatureK
        self.evaluator = evaluator
        self.configuration_suggester = configuration_suggester
        self.configuration = initial_configuration
        self.calculate_observables_after_steps = \
            calculate_observables_after_steps

        # Observables.
        self.averaged_energy = 0.0

    def run(self, steps_to_evolve, print_energies=False):
        energy = self.evaluator.get_total_energy(self.configuration)
        self.averaged_energy = energy
        accepted_steps = 1
        for step in range(0, steps_to_evolve):
            new_configuration = self.configuration_suggester.\
                suggest_new_configuration(self.configuration)

            new_energy = self.evaluator.get_total_energy(new_configuration)
            deltaE = new_energy - energy

            if self.check_acceptance(deltaE):
                energy = new_energy
                accepted_steps += 1
                self.averaged_energy = ((self.averaged_energy*(accepted_steps-1)) +
                                   energy)/accepted_steps

                if print_energies is True:
                    print("Accepted step, energy is now: ", energy)
                self.configuration = new_configuration

    def check_acceptance(self, deltaE):
        if deltaE > 0.0:
            randomNumber = random()
            probability = np.exp(
                -1.0 * deltaE / (boltzmannConstant * self.temperatureK))
            if probability < randomNumber:
                return False
        return True


