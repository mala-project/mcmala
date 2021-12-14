from random import random

from ase.units import kB
import numpy as np

from mcmala import Evaluator, ConfigurationSuggester


class MarkovChain:
    def __init__(self, temperatureK, evaluator: Evaluator,
                 configuration_suggester: ConfigurationSuggester,
                 initial_configuration, calculate_observables_after_steps=1):
        """
        Represent a single Markov chain.

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

        # Observables.
        self.averaged_energy = 0.0

    def run(self, steps_to_evolve, print_energies=False):
        """
        Run this Markov chain for a specified number of steps.

        Parameters
        ----------
        steps_to_evolve : int
            Number of steps to run the simulation for.

        print_energies : bool
            If True, the energies are printed at each step of the simulation.

        """
        energy = self.evaluator.get_total_energy(self.configuration)
        self.averaged_energy = energy
        accepted_steps = 1
        for step in range(0, steps_to_evolve):
            new_configuration = self.configuration_suggester.\
                suggest_new_configuration(self.configuration)

            new_energy = self.evaluator.get_total_energy(new_configuration)
            deltaE = new_energy - energy

            if self.__check_acceptance(deltaE):
                energy = new_energy
                accepted_steps += 1
                self.averaged_energy = ((self.averaged_energy*(accepted_steps-1)) +
                                   energy)/accepted_steps

                if print_energies is True:
                    print("Accepted step, energy is now: ", energy)
                self.configuration = new_configuration

    def __check_acceptance(self, deltaE):
        if deltaE > 0.0:
            randomNumber = random()
            probability = np.exp(
                -1.0 * deltaE / (kB * self.temperatureK))
            if probability < randomNumber:
                return False
        return True


