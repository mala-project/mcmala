"""Markov chain for Monte Carlo simulation."""
from datetime import datetime
import importlib
import json
import os
import pickle
from random import random

import mala
from ase import Atoms
from ase.units import kB
from ase.io.trajectory import TrajectoryWriter, Trajectory
from ase.calculators.calculator import Calculator
import numpy as np

from mcmala import ConfigurationSuggester
from mcmala.common.parallelizer import get_rank, printout, barrier, get_comm,\
                                       get_size, get_world_comm
from mcmala.simulation.atom_displacer import AtomDisplacer
from mcmala.simulation import is_qepy_available, is_mala_available
from .markovchainresults import MarkovChainResults


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
                 ensemble="nvt", equilibration_steps=0, path_to_folder="."):
        self.temperatureK = temperatureK
        self.evaluator = evaluator
        self.configuration_suggester = configuration_suggester
        self.configuration = initial_configuration
        self.calculate_observables_after_steps = \
            calculate_observables_after_steps
        self.ensemble = ensemble
        self.equilibration_steps = equilibration_steps

        super(MarkovChain, self).__init__(markov_chain_id=markov_chain_id,
                                          additonal_observables=additonal_observables,
                                          path_to_folder=path_to_folder)
        self.is_continuation_run = False

        # Create folder for this Markov chain.
        if get_rank() == 0:
            if not os.path.exists(os.path.join(path_to_folder, self.id)):
                os.makedirs(os.path.join(path_to_folder, self.id))

        # For running.
        self.all_observables_counter = None
        self.checkpoint_counter = None
        self.start_time = None
        self.current_energy = None
        self.energy_file = None
        self.trajectory_logger = None

        # If we have an Espresso calculator, we need to updated some
        # file paths.
        if hasattr(evaluator, "update_paths"):
            evaluator.update_paths(path_to_folder, markov_chain_id)

    @classmethod
    def load_run(cls, markov_chain_id, path_to_folder="./"):
        markov_chain_id = str(markov_chain_id)
        last_configurations = Trajectory(os.path.join(path_to_folder,
                                                      markov_chain_id,
                                                      markov_chain_id + ".traj"))
        # Load from the files.
        markov_chain_data, additonal_observables = \
            cls._load_files(markov_chain_id, path_to_folder)

        # Treat the observables.
        temperature = markov_chain_data["metadata"]["temperature"]
        calculate_observables_after_steps = \
            markov_chain_data["metadata"]["calculate_observables_after_steps"]
        ensemble = markov_chain_data["metadata"]["ensemble"]
        equilibration_steps = markov_chain_data["metadata"]["equilibration_steps"]

        # Load the evaluator from the saved files.
        evaluator_type = markov_chain_data["metadata"]["evaluator"]
        if evaluator_type == "EspressoMC":
            if is_qepy_available:
                from mcmala.simulation.espresso_mc import EspressoMC
                input_file_name = markov_chain_id+"_espresso.pwi"
                evaluator = EspressoMC.\
                    from_input_file(os.path.join(path_to_folder,
                                                 markov_chain_id),
                                    input_file_name)
            else:
                raise Exception("QEPy not available on this system.")
        elif evaluator_type == "MALA":
            if is_mala_available:
                import mala
                params = mala.Parameters.load_from_file(
                    os.path.join(path_to_folder, markov_chain_id,
                                 markov_chain_id+".params.json"))
                network = mala.Network.\
                    load_from_file(params, os.path.join(path_to_folder, markov_chain_id,
                                   markov_chain_id+".network.pth"))
                iscaler = mala.DataScaler.\
                    load_from_file(os.path.join(path_to_folder, markov_chain_id,
                                   markov_chain_id+".iscaler.pkl"))
                oscaler = mala.DataScaler.\
                    load_from_file(os.path.join(path_to_folder, markov_chain_id,
                                   markov_chain_id+".oscaler.pkl"))
                data_handler = mala.\
                    DataHandler(params, input_data_scaler=iscaler,
                                output_data_scaler=oscaler)
                reference_path = os.path.join(path_to_folder, markov_chain_id,
                                                     markov_chain_id+".reference.json")
                evaluator = mala.MALA(params, network, data_handler,
                                      ["json", reference_path])
            else:
                raise Exception("QEPy not available on this system.")
        else:
            raise Exception("Evaluator not implemented for loading.")

        # Load the configuration suggester from the saved files.
        # Same as above - in the future we may want to do this dynamically.
        # For now, there is really only one suggester.
        configuration_suggester = AtomDisplacer.\
            from_json(markov_chain_data["metadata"]["configuration_suggester"])

        # Create the object.
        loaded_result = MarkovChain(temperature, evaluator,
                                    configuration_suggester,
                                    last_configurations[-1],
                                    calculate_observables_after_steps=calculate_observables_after_steps,
                                    ensemble=ensemble,
                                    equilibration_steps=equilibration_steps,
                                    markov_chain_id=markov_chain_id,
                                    additonal_observables=additonal_observables,
                                    path_to_folder=path_to_folder)

        # We have to process the loaded data so that everything fits.
        loaded_result._process_loaded_obervables(markov_chain_data,
                                                 path_to_folder,
                                                 markov_chain_id)
        loaded_result.is_continuation_run = True
        return loaded_result

    def run(self, steps_to_evolve, checkpoints_after_steps=0):
        """
        Run this Markov chain for a specified number of steps.

        Parameters
        ----------
        steps_to_evolve : int
            Number of steps to run the simulation for.
        """

        # Open log files, perform first calculation, etc.
        self._setup_run(steps_to_evolve)

        # Actually run for a set amount of steps.
        self._run(steps_to_evolve, checkpoints_after_steps)

        # Write final results.
        self._wrap_up_run(steps_to_evolve)

    def _setup_run(self, steps_to_evolve):
        if steps_to_evolve <= self.equilibration_steps:
            raise Exception(
                "Will not attempt to run for less steps then are "
                "necessary for equilibration.")
        printout("Starting Markov chain " + self.id + ".")
        self.start_time = datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")

        self.evaluator.calculate(atoms=self.configuration)
        # The first energy always will be for the first, initial configuration.
        # Maybe there is a way around that, but for now, it seems the best
        # we can do for restart calculations is just reading from file.
        if self.is_continuation_run:
            self.current_energy = self.observables["total_energy"][-1]
        else:
            self.current_energy = self.evaluator.results["energy"]
            self.observables["total_energy"].append(self.current_energy)

        self.all_observables_counter = 0
        self.checkpoint_counter = 0

        # Set the logging modes.
        logging_mode = "w" if self.is_continuation_run is False else "a"

        if get_rank() == 0:
            if isinstance(self.configuration, Atoms) is True:
                self.trajectory_logger = TrajectoryWriter(
                    os.path.join(self.path_to_folder, self.id, self.id + ".traj"),
                    mode=logging_mode, master=True)
        return

    def _run(self, steps_to_evolve, checkpoints_after_steps):
        for step in range(self.steps_evolved, steps_to_evolve):
            new_configuration = self.configuration_suggester. \
                suggest_new_configuration(self.configuration)

            # We are giving the properties here because certain properties
            # have to be calculated during the energy evaluation (e.g.
            # the stress tensor)
            self.evaluator.calculate(new_configuration,
                                     properties=list(self.observables.keys()))
            new_energy = self.evaluator.results["energy"]
            deltaE = new_energy - self.current_energy
            self.steps_evolved = step
            if self.__check_acceptance(deltaE):
                self.current_energy = new_energy
                self.configuration = new_configuration

                # Logging.
                uncertainty_string = ""
                if isinstance(self.evaluator, mala.MALAUncertainty):
                    uncertainty_string += "with uncertainty: " + \
                        str(self.evaluator.results["energy_uncertainty"])
                printout("Accepted step, energy is now: ",
                         self.current_energy, uncertainty_string)

                if get_rank() == 0:
                    if isinstance(self.configuration, Atoms) is True:
                        self.trajectory_logger.write(new_configuration)

                    # Calculate observables, given that were beyond the
                    # equilibration stage.
                    if step >= self.equilibration_steps:
                        # The energy is always calculated.
                        self.accepted_steps += 1
                        self.observables["total_energy"].append(self.
                                                                current_energy)

                        # All the other observables are only calculated
                        # each calculate_observables_after_steps steps.
                        self.all_observables_counter += 1
                        if self.all_observables_counter == \
                                self.calculate_observables_after_steps:
                            self.__get_additional_observables()
                            self.all_observables_counter = 0

                    # Create a checkpoint, if necessary.
                    if checkpoints_after_steps > 0:
                        if self.checkpoint_counter >= checkpoints_after_steps:

                            printout("Markov chain", self.id,
                                     "creating checkpoint.")
                            # step + 1 because it's NUMBER of steps,
                            # not step number.
                            self._save_run(self.start_time,
                                           datetime.now().strftime(
                                                "%d-%b-%Y (%H:%M:%S.%f)"),
                                           step + 1)
                            self.checkpoint_counter = 0
                self.checkpoint_counter += 1

    def _wrap_up_run(self, steps_to_evolve):
        if get_rank() == 0:
            printout("Markov chain", self.id, "finished, saving results.")
            self._save_run(self.start_time,
                           datetime.now().strftime(
                                "%d-%b-%Y (%H:%M:%S.%f)"),
                           steps_to_evolve)

    def __get_additional_observables(self):
        """Read additional observables from MALA."""

        # The total energy is always calculated. If it is the ONLY thing
        # being calculated, we might as well just not calculate it.
        if len(self.observables) == 1:
            return

        # Calculate additional properties using e.g. a MALA calculator.
        self.evaluator.calculate_properties(self.configuration,
                                            properties=list(self.observables.keys()))

        # Save the observables.
        for entry in self.observables.keys():
            if entry == "rdf":
                self.observables["rdf"].append(self.evaluator.results[entry][0])

                # We can just always update this.
                self.observables["rdf_distances"] = self.evaluator.results[entry][1]

            elif entry == "static_structure_factor":
                self.observables["static_structure_factor"].append(self.evaluator.results[entry][0])

                # We can just always update this.
                self.observables["static_structure_factor_kpoints"] = self.evaluator.results[entry][1]
            elif entry == "tpcf":
                self.observables["tpcf"].append(self.evaluator.results[entry][0])

                # We can just always update this.
                self.observables["tpcf_radii"] = self.evaluator.results[entry][1]
            else:
                if entry != "total_energy":
                    self.observables[entry].append(self.evaluator.results[entry])

    def _save_run(self, start_time, end_time, step_evolved):
        # Construct meta data.
        metadata = {
            "id": self.id,
            "temperature": self.temperatureK,
            "configuration_suggester": self.configuration_suggester.to_json(),
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

        # We clean the observables, because not all can be saved in the JSON
        # file; some arrays are large and have to be saved in pickle files.
        cleaned_observables = {}
        for entry in self.observables.keys():
            filename = os.path.join(self.id, self.id + "_" + entry + ".npy")
            np.save(filename, self.observables[entry])
            cleaned_observables[entry] = self.id + "_" + entry + ".npy"

        # Now we can save everything to pickle.
        save_dict = {"metadata": metadata, "observables":
                     cleaned_observables}
        with open(os.path.join(self.path_to_folder, self.id, self.id+".json"), "w",
                  encoding="utf-8") as f:
            json.dump(save_dict, f, ensure_ascii=False, indent=4)

        # Save the MALA based data of the observables.
        self.evaluator.save_calculator(os.path.join(self.id, self.id))

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





