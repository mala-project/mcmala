import json
import os
import pickle

class MarkovChainResults():
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
    def __init__(self, markov_chain_id="mcmala_default", additonal_observables=[]):
        # Observables.
        self.id = str(markov_chain_id)
        self.observables = {"total_energy": 0.0}
        for entry in additonal_observables:
            if entry == "rdf":
                self.observables[entry] = {"rdf": None, "distances": None}
            else:
                self.observables[entry] = 0.0
            if entry == "ion_ion_energy":
                self.observables[entry] = 0.0
            if entry == "static_structure_factor":
                self.observables[entry] = {"static_structure_factor": None,
                                           "kpoints": None}
            if entry == "tpcf":
                self.observables[entry] = {"tpcf": None, "radii": None}

    @classmethod
    def load_run(cls, markov_chain_id, path_to_folder=None):
        if path_to_folder is None:
            folder_to_load = markov_chain_id
        else:
            folder_to_load = os.path.join(path_to_folder, markov_chain_id)

        # Load the JSON file of the run.
        with open(os.path.join(folder_to_load, markov_chain_id + ".json"),
                  encoding="utf-8") as json_file:
            markov_chain_data = json.load(json_file)

        # Now we can create the MarkovChainResults object.
        additonal_observables = list(markov_chain_data["averaged_observables"].keys())
        additonal_observables.remove("total_energy")
        loaded_result = MarkovChainResults(markov_chain_id=markov_chain_id,
                                           additonal_observables=additonal_observables)

        # Now we can load the values.
        for entry in markov_chain_data["averaged_observables"].keys():
            if entry == "ion_ion_energy" or entry == "total_energy":
                loaded_result.observables[entry] = markov_chain_data["averaged_observables"][entry]

            if entry == "rdf" or entry == "tpcf" or entry == "static_structure_factor":
                filename = os.path.join(folder_to_load,
                                        markov_chain_data["averaged_observables"][entry])
                with open(filename, 'rb') as handle:
                    loaded_result.observables[entry] = pickle.load(handle)
        return loaded_result

    # Properties (Observables)
    @property
    def total_energy(self):
        """Total energy of the system (in eV)."""
        return self.observables["total_energy"]

    @property
    def rdf(self):
        """Radial distribution function of system"""
        return self.observables["rdf"]["rdf"]

    @property
    def rdf_grid(self):
        return  self.observables["rdf"]["distances"]

    @property
    def tpcf(self):
        """Three particle correlation function of system."""
        return self.observables["tpcf"]["tpcf"]

    @property
    def tpcf_grid(self):
        return self.observables["tpcf"]["radii"]

    @property
    def static_structure_factor(self):
        """Static structure factor of system."""
        return self.observables["static_structure_factor"]["static_structure_factor"]

    @property
    def static_structure_factor_grid(self):
        return self.observables["static_structure_factor"]["kpoints"]

    @property
    def ion_ion_energy(self):
        """Ion ion energy in eV"""
        return self.observables["ion_ion_energy"]

