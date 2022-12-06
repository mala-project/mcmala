import json
import os
import pickle

import numpy as np

class MarkovChainResults:
    """
    Represents the averaged results of one Markov chain.

    Can be used to load and analyze the results from a single run-

    Parameters
    ----------
    additonal_observables : list
        A list of additional (=not the total energy) observables
        that will be calculated at each
        calculate_observables_after_steps-th step.


    """
    def __init__(self, markov_chain_id="mcmala_default",
                 additonal_observables=[],
                 path_to_folder="."):
        # Observables.
        self.id = str(markov_chain_id)
        self.path_to_folder = path_to_folder
        self.observables = {"total_energy": []}
        for entry in additonal_observables:
            if entry == "rdf":
                self.observables[entry] = []
                self.observables[entry+"_distances"] = []
            elif entry == "ion_ion_energy":
                self.observables[entry] = []
            elif entry == "static_structure_factor":
                self.observables[entry] = []
                self.observables[entry+"_kpoints"] = []
            elif entry == "tpcf":
                self.observables[entry] = []
                self.observables[entry+"_radii"] = []
            else:
                self.observables[entry] = []

        self.steps_evolved = 0
        self.accepted_steps = 0

    @classmethod
    def load_run(cls, markov_chain_id, path_to_folder="./"):
        # Load from the files.
        markov_chain_data, additonal_observables = \
            cls._load_files(markov_chain_id, path_to_folder)

        # Create the object.
        loaded_result = MarkovChainResults(markov_chain_id=markov_chain_id,
                                           additonal_observables=
                                           additonal_observables)
        # We have to process the loaded data so that everything fits.
        loaded_result._process_loaded_obervables(markov_chain_data,
                                                 path_to_folder,
                                                 markov_chain_id)
        return loaded_result

    @staticmethod
    def _load_files(markov_chain_id, path_to_folder):
        folder_to_load = os.path.join(path_to_folder, markov_chain_id)

        # Load the JSON file of the run.
        with open(os.path.join(folder_to_load, markov_chain_id + ".json"),
                  encoding="utf-8") as json_file:
            markov_chain_data = json.load(json_file)

        # Now we can create the MarkovChainResults object.
        additonal_observables = list(markov_chain_data["observables"].keys())
        additonal_observables.remove("total_energy")

        return markov_chain_data, additonal_observables

    def _process_loaded_obervables(self, markov_chain_data,
                                   path_to_folder,
                                   markov_chain_id):
        folder_to_load = os.path.join(path_to_folder, markov_chain_id)
        self.steps_evolved = markov_chain_data["metadata"][
            "steps_evolved"]
        self.accepted_steps = markov_chain_data["metadata"][
            "accepted_steps"]
        for entry in markov_chain_data["observables"].keys():
            # The json file contains the paths to the observable files.
            filename = os.path.join(folder_to_load,
                                    markov_chain_data["observables"][entry])
            self.observables[entry] = list(np.load(filename))

    # Properties (Observables)
    @property
    def total_energy(self):
        """Total energy of the system (in eV)."""
        return self.observables["total_energy"]

    @property
    def rdf(self):
        """Radial distribution function of system"""
        return self.observables["rdf"]

    @property
    def rdf_grid(self):
        return self.observables["rdf_distances"]

    @property
    def tpcf(self):
        """Three particle correlation function of system."""
        return self.observables["tpcf"]

    @property
    def tpcf_grid(self):
        return self.observables["tpcf_radii"]

    @property
    def static_structure_factor(self):
        """Static structure factor of system."""
        return self.observables["static_structure_factor"]

    @property
    def static_structure_factor_grid(self):
        return self.observables["static_structure_factor_kpoints"]

    @property
    def ion_ion_energy(self):
        """Ion ion energy in eV"""
        return self.observables["ion_ion_energy"]

