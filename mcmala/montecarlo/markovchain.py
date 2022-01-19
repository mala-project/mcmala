"""Markov chain for Monte Carlo simulation."""
from random import random
import json

from ase import Atoms
from ase.units import kB
from ase.io.trajectory import TrajectoryWriter
from ase.calculators.calculator import Calculator
import numpy as np

from mcmala import ConfigurationSuggester
from datetime import datetime


class MarkovChain:
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

    additonal_observables : list
        A list of additional (=not the total energy) observables
        that will be calculated at each
        calculate_observables_after_steps-th step.

    ensemble : string
        Determines based on which ensemble the acceptance will be handled.
        "nvt" : NVT (canonical) ensemble
        "debug" : each configuration will be accepted.
    """

    def __init__(self, temperatureK, evaluator: Calculator,
                 configuration_suggester: ConfigurationSuggester,
                 initial_configuration, calculate_observables_after_steps=1,
                 markov_chain_id="mcmala_default", additonal_observables=[],
                 ensemble="nvt", equilibration_steps=0):
        self.temperatureK = temperatureK
        self.evaluator = evaluator
        self.configuration_suggester = configuration_suggester
        self.configuration = initial_configuration
        self.calculate_observables_after_steps = \
            calculate_observables_after_steps
        self.id = str(markov_chain_id)
        self.ensemble = ensemble
        self.equilibration_steps = equilibration_steps

        # Observables.
        self.observables = {"total_energy": 0.0}
        for entry in additonal_observables:
            if entry == "rdf":
                self.observables[entry] = {"rdf": None, "distances": None}
            else:
                self.observables[entry] = 0.0
            if entry == "ion_ion_energy":
                self.observables[entry] = 0.0
            if entry == "static_structure_factor":
                self.observables[entry] = {"static_structure_factor": None, "kpoints": None}
            if entry == "tpcf":
                self.observables[entry] = {"tpcf": None, "radii": None}

    def run(self, steps_to_evolve, print_energies=False,
            save_run=True, log_energies=False, log_trajectory=False):
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
        if steps_to_evolve < self.equilibration_steps:
            raise Exception("Will not attempt to run for less steps then are "
                            "necessary for equilibration.")
        print("Starting Markov chain "+self.id+".")
        start_time = datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

        self.evaluator.calculate(self.configuration)
        energy = self.evaluator.results["energy"]
        self.observables["total_energy"] = energy
        accepted_steps = 0
        all_observables_counter = 0

        # Prepare logging, if necessary.
        if isinstance(self.configuration, Atoms) is False:
           log_trajectory = False
        if log_trajectory:
            trajectory_logger = TrajectoryWriter(self.id+".traj")
        if log_energies:
            energy_file = open(self.id+"_energies.log", "w")
            energy_file.write("step\ttotal energy\n")

        for step in range(0, steps_to_evolve):
            new_configuration = self.configuration_suggester.\
                suggest_new_configuration(self.configuration)

            self.evaluator.calculate(self.configuration)
            new_energy = self.evaluator.results["energy"]
            deltaE = new_energy - energy

            if self.__check_acceptance(deltaE):
                energy = new_energy
                self.configuration = new_configuration
                if step >= self.equilibration_steps:
                    accepted_steps += 1
                    self.observables["total_energy"] =\
                        ((self.observables["total_energy"]
                                                   * (accepted_steps - 1)) +
                                                    energy) / accepted_steps

                    if print_energies is True:
                        print("Accepted step, energy is now: ", energy)

                # Calculate the observables.
                all_observables_counter += 1
                if all_observables_counter == \
                        self.calculate_observables_after_steps\
                        and step >= self.equilibration_steps:
                    self.evaluator.calculate_properties(
                                            self.configuration,
                                            properties=list(self.
                                                            observables.
                                                            keys()))
                    for entry in self.observables.keys():
                        if entry == "rdf":
                            if self.observables[entry]["rdf"] is None:
                                self.observables[entry]["rdf"] = \
                                    self.evaluator.results[entry][0]
                            else:
                                self.observables[entry]["rdf"] = \
                                    ((self.observables[entry]["rdf"]
                                      * (accepted_steps - 1)) +
                                     self.evaluator.results[entry][0]) / \
                                    accepted_steps
                            self.observables[entry]["distances"] = self.evaluator.results[entry][1]
                        if entry == "static_structure_factor":
                            if self.observables[entry]["static_structure_factor"] is None:
                                self.observables[entry]["static_structure_factor"] = \
                                    self.evaluator.results[entry][0]
                            else:
                                self.observables[entry]["static_structure_factor"] = \
                                    ((self.observables[entry]["static_structure_factor"]
                                      * (accepted_steps - 1)) +
                                     self.evaluator.results[entry][0]) / \
                                    accepted_steps
                            self.observables[entry]["kpoints"] = self.evaluator.results[entry][1]
                        if entry == "tpcf":
                            if self.observables[entry]["tpcf"] is None:
                                self.observables[entry]["tpcf"] = \
                                    self.evaluator.results[entry][0]
                            else:
                                self.observables[entry]["tpcf"] = \
                                    ((self.observables[entry]["tpcf"]
                                      * (accepted_steps - 1)) +
                                     self.evaluator.results[entry][0]) / \
                                    accepted_steps
                            self.observables[entry]["radii"] = self.evaluator.results[entry][1]
                        if entry == "ion_ion_energy":
                            self.observables[entry] = self.evaluator.results[entry]
                    all_observables_counter = 0

                # Finally, the logging.
                if log_trajectory:
                    trajectory_logger.write(new_configuration)
                if log_energies:
                    energy_file.write("{0} \t {1:10.4f}\n".format(step, energy))

        end_time = datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

        if log_energies:
            energy_file.close()

        print("Markov chain", self.id, "finished, saving results.")
        if save_run:
            # Construct meta data.
            metadata = {
                "id": self.id,
                "temperature": self.temperatureK,
                "configuration_suggester": self.configuration_suggester.get_info(),
                "configuration_type": type(self.configuration).__name__,
                "evaluator": type(self.evaluator).__name__,
                "start_time": start_time,
                "end_time": end_time,
                "ensemble": self.ensemble,
                "steps_evolved": steps_to_evolve,
                "accepted_steps": accepted_steps,
                "equilibration_steps": self.equilibration_steps,
            }
            self.__save_run(metadata)

    def __save_run(self, metadata):
        save_dict = {"metadata": metadata, "averaged_observables":
                    self.observables}
        with open(self.id+".json", "w", encoding="utf-8") as f:
            json.dump(save_dict, f, ensure_ascii=False, indent=4)
        try:
            from mala import ASECalculator
            if isinstance(self.evaluator, ASECalculator):
                print("Saving MALA parameters.")
                self.evaluator.params.save(self.id+"_mala_params.pkl")
        except:
            pass

    def __check_acceptance(self, deltaE):
        if self.ensemble == "nvt":
            if deltaE > 0.0:
                randomNumber = random()
                probability = np.exp(
                    -1.0 * deltaE / (kB * self.temperatureK))
                if probability < randomNumber:
                    return False
            return True
        elif self.ensemble == "debug":
            return True
        else:
            raise Exception("Unknown ensemble selected.")

    # Properties (Observables)
    @property
    def total_energy(self):
        """Total energy of the system in eV."""
        return self.observables["total_energy"]



