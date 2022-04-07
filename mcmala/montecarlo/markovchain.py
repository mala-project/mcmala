"""Markov chain for Monte Carlo simulation."""
from random import random
import json
import os
import pickle

from ase import Atoms
from ase.units import kB
from ase.io.trajectory import TrajectoryWriter, Trajectory
from ase.calculators.calculator import Calculator
import numpy as np

from mcmala import ConfigurationSuggester
from mcmala.common.parallelizer import get_rank, printout, barrier, get_comm,\
                                       get_size
from .markovchainresults import MarkovChainResults
from datetime import datetime


class MarkovChain(MarkovChainResults):
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
        self.ensemble = ensemble
        self.equilibration_steps = equilibration_steps

        super(MarkovChain, self).__init__(markov_chain_id=markov_chain_id,
                                          additonal_observables=additonal_observables)
        self.is_continuation_run = False

        # Create folder for this Markov chain.
        if get_rank() == 0:
            if not os.path.exists(self.id):
                os.makedirs(self.id)

    def run(self, steps_to_evolve, print_energies=False,
                           save_run=True, log_energies=False, log_trajectory=False,
                           checkpoints_after_steps=0):
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
        if steps_to_evolve <= self.equilibration_steps:
            raise Exception(
                "Will not attempt to run for less steps then are "
                "necessary for equilibration.")
        printout("Starting Markov chain " + self.id + ".")
        start_time = datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

        self.evaluator.calculate(atoms=self.configuration)
        # The first energy always will be for the first, initial configuration.
        # Maybe there is a way around that, but for now, it seems the best
        # we can do for restart calculations is just reading from file.
        if self.is_continuation_run:
            energy = self.energies[-1]
        else:
            energy = self.evaluator.results["energy"]
            self.observables["total_energy"] = energy

        all_observables_counter = 0
        checkpoint_counter = 0

        # Set the logging modes.
        logging_mode = "w" if self.is_continuation_run is False else "a"

        # Prepare logging, if necessary.
        if isinstance(self.configuration, Atoms) is False:
            log_trajectory = False

        if get_rank() == 0:
            if log_trajectory:
                trajectory_logger = TrajectoryWriter(
                    os.path.join(self.id, self.id + ".traj"),
                    mode=logging_mode)
            if log_energies:
                energy_file = open(
                    os.path.join(self.id, self.id + "_energies.log"),
                    logging_mode)
                if not self.is_continuation_run:
                    energy_file.write("step\ttotal energy\n")

        for step in range(self.steps_evolved, steps_to_evolve):
            new_configuration = self.configuration_suggester. \
                suggest_new_configuration(self.configuration)

            self.evaluator.calculate(new_configuration)
            new_energy = self.evaluator.results["energy"]
            deltaE = new_energy - energy

            if self.__check_acceptance(deltaE):
                energy = new_energy
                self.configuration = new_configuration

                # Logging.
                if print_energies is True:
                    printout("Accepted step, energy is now: ", energy)

                if get_rank() == 0:
                    if log_trajectory:
                        trajectory_logger.write(new_configuration)
                    if log_energies:
                        energy_file.write(
                            "{0} \t {1:10.4f}\n".format(step, energy))

                    # Calculate observables, given that were beyond the
                    # equilibration stage.
                    if step >= self.equilibration_steps:
                        # The energy is always calculated.
                        self.accepted_steps += 1
                        self.observables["total_energy"] = \
                            ((self.observables["total_energy"]
                              * (self.accepted_steps - 1)) +
                             energy) / self.accepted_steps

                        # All the other observables are only calculated
                        # each calculate_observables_after_steps steps.
                        all_observables_counter += 1
                        if all_observables_counter == \
                                self.calculate_observables_after_steps:
                            self.__get_additional_observables(
                                self.accepted_steps)
                            all_observables_counter = 0

                    # Create a checkpoint, if necessary.
                    if checkpoints_after_steps > 0:
                        if checkpoint_counter >= checkpoints_after_steps:
                            if log_energies:
                                energy_file.flush()

                            printout("Markov chain", self.id,
                                     "creating checkpoint.")
                            if save_run:
                                # step + 1 because it's NUMBER of steps,
                                # not step number.
                                self.__save_run(start_time,
                                                datetime.now().strftime(
                                                    "%d-%b-%Y (%H:%M:%S.%f)"),
                                                step + 1)
                            checkpoint_counter = 0
                checkpoint_counter += 1

        if get_rank() == 0:
            if log_energies:
                energy_file.close()

            printout("Markov chain", self.id, "finished, saving results.")
            if save_run:
                self.__save_run(start_time,
                                datetime.now().strftime(
                                    "%d-%b-%Y (%H:%M:%S.%f)"),
                                steps_to_evolve)

    @classmethod
    def load_run(cls, markov_chain_id, evaluator: Calculator,
                 configuration_suggester: ConfigurationSuggester,
                 path_to_folder=None):
        last_configurations = Trajectory(os.path.join(markov_chain_id,
                                                      markov_chain_id + ".traj"))

        # Load from the files.
        markov_chain_data, additonal_observables, energies = cls._load_files(markov_chain_id,
                                                                   path_to_folder,
                                                                   True)
        temperature = markov_chain_data["metadata"]["temperature"]
        calculate_observables_after_steps = \
            markov_chain_data["metadata"]["calculate_observables_after_steps"]
        ensemble = markov_chain_data["metadata"]["ensemble"]
        equilibration_steps = markov_chain_data["metadata"]["equilibration_steps"]

        # Create the object.
        loaded_result = MarkovChain(temperature, evaluator,
                                    configuration_suggester,
                                    last_configurations[-1],
                                    calculate_observables_after_steps=calculate_observables_after_steps,
                                    ensemble=ensemble,
                                    equilibration_steps=equilibration_steps,
                                    markov_chain_id=markov_chain_id,
                                    additonal_observables=additonal_observables)
        loaded_result.energies  = energies

        # We have to process the loaded data so that everything fits.
        loaded_result._process_loaded_obervables(markov_chain_data)
        loaded_result.is_continuation_run = True
        return loaded_result


    def __get_additional_observables(self, accepted_steps):
        """Read additional observables from MALA."""

        # The total energy is always calculated. If it is the ONLY thing
        # being calculated, we might as well just not calculate it.
        if len(self.observables) == 1:
            return

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
                self.observables[entry]["distances"] = \
                self.evaluator.results[entry][1]
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
                self.observables[entry]["kpoints"] = \
                self.evaluator.results[entry][1]
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
                self.observables[entry]["radii"] = \
                self.evaluator.results[entry][1]
            if entry == "ion_ion_energy":
                self.observables[entry] = self.evaluator.results[entry]

    def __save_run(self, start_time, end_time, step_evolved):
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
            "steps_evolved": step_evolved,
            "accepted_steps": self.accepted_steps,
            "equilibration_steps": self.equilibration_steps,
            "calculate_observables_after_steps": self.calculate_observables_after_steps,
            "number_of_ranks": get_size()
        }
        self.steps_evolved = step_evolved

        # We clean the observables, because not all can be saved in the JSON
        # file; some arrays are large and have to be saved in pickle files.
        cleaned_observables = {}
        for entry in self.observables.keys():
            if entry == "ion_ion_energy" or entry == "total_energy":
                cleaned_observables[entry] = self.observables[entry]

            if entry == "rdf" or entry == "tpcf" or entry == "static_structure_factor":
                filename = os.path.join(self.id, self.id+"_"+entry+".pkl")
                with open(filename, 'wb') as handle:
                    pickle.dump(self.observables[entry], handle, protocol=4)
                cleaned_observables[entry] = self.id+"_"+entry+".pkl"

        # Now we can save everything to pickle.
        save_dict = {"metadata": metadata, "averaged_observables":
                    cleaned_observables}
        with open(os.path.join(self.id, self.id+".json"), "w",
                  encoding="utf-8") as f:
            json.dump(save_dict, f, ensure_ascii=False, indent=4)

        # Save the MALA based data of the observables.
        self.evaluator.save_calculator(os.path.join(self.id,
                                                    self.id+"_evaluator.json"))

    def __check_acceptance(self, deltaE):
        return_value = False
        if self.ensemble == "nvt":
            if deltaE > 0.0:
                randomNumber = random()
                probability = np.exp(
                    -1.0 * deltaE / (kB * self.temperatureK))
                if probability < randomNumber:
                    return_value = False
                else:
                    return_value = True
            else:
                return_value = True
        elif self.ensemble == "debug":
            return_value = True
        else:
            raise Exception("Unknown ensemble selected.")

        # synchronize across nodes.
        barrier()
        if get_size() == 1:
            return return_value
        else:
            return_value = get_comm().bcast(return_value, root=0)
            return return_value





