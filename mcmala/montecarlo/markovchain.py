from random import random
import json

from ase.units import kB
from ase.calculators.calculator import Calculator
import numpy as np

from mcmala import ConfigurationSuggester
from datetime import datetime


class MarkovChain:
    def __init__(self, temperatureK, evaluator: Calculator,
                 configuration_suggester: ConfigurationSuggester,
                 initial_configuration, calculate_observables_after_steps=1,
                 markov_chain_id="mcmala_default"):
        """
        Represent a single Markov chain.

        Parameters
        ----------
        temperatureK : float
            Temperature in Kelvin.

        evaluator : ase.calculators.calculator.Calculator
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
        self.id = str(markov_chain_id)

        # Observables.
        self.observables = {"total_energy": 0}

    def run(self, steps_to_evolve, print_energies=False,
            save_run=True):
        """
        Run this Markov chain for a specified number of steps.

        Parameters
        ----------
        steps_to_evolve : int
            Number of steps to run the simulation for.

        print_energies : bool
            If True, the energies are printed at each step of the simulation.

        save_run : bool
            Is True by default; if False, the run will not be saved (e.g.
            for examples and such).

        """
        print("Starting Markov chain "+self.id+".")
        start_time = datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

        self.evaluator.calculate(self.configuration)
        energy = self.evaluator.results["energy"]
        self.observables["total_energy"] = energy
        accepted_steps = 1
        for step in range(0, steps_to_evolve):
            new_configuration = self.configuration_suggester.\
                suggest_new_configuration(self.configuration)

            self.evaluator.calculate(self.configuration)
            new_energy = self.evaluator.results["energy"]
            deltaE = new_energy - energy

            if self.__check_acceptance(deltaE):
                energy = new_energy
                accepted_steps += 1
                self.observables["total_energy"] =\
                    ((self.observables["total_energy"]
                                               * (accepted_steps - 1)) +
                                                energy) / accepted_steps

                if print_energies is True:
                    print("Accepted step, energy is now: ", energy)
                self.configuration = new_configuration
        end_time = datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

        print("Markov chain", self.id, "finished, saving results.")
        if save_run:
            # Construct meta data.
            metadata = {
                "id": self.id,
                "temperature": self.temperatureK,
                "configuration type": type(self.configuration).__name__,
                "configuration suggester": type(self.configuration_suggester).__name__,
                "evaluator": type(self.evaluator).__name__,
                "start_time": start_time,
                "end_time": end_time,
            }
            self.__save_run(metadata)

    def __save_run(self, metadata):
        save_dict = {"metadata": metadata, "averaged_observables":
                    self.observables}
        with open(self.id+".json", "w", encoding="utf-8") as f:
            json.dump(save_dict, f, ensure_ascii=False, indent=4)
        pass

    def __check_acceptance(self, deltaE):
        if deltaE > 0.0:
            randomNumber = random()
            probability = np.exp(
                -1.0 * deltaE / (kB * self.temperatureK))
            if probability < randomNumber:
                return False
        return True

    # Properties (Observables)
    @property
    def total_energy(self):
        """Total energy of the system in eV."""
        return self.observables["total_energy"]



