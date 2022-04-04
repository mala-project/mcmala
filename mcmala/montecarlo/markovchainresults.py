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

        # All the accepted energies, for plotting and such.
        self.energies = None
        self.steps_evolved = 0
        self.accepted_steps = 0

    @classmethod
    def load_run(cls, markov_chain_id, path_to_folder=None, read_energy=False):
        # Load from the files.
        markov_chain_data, additonal_observables = cls._load_files(markov_chain_id,
                                                                   path_to_folder)

        # Create the object.
        loaded_result = MarkovChainResults(markov_chain_id=markov_chain_id,
                                           additonal_observables=additonal_observables)

        # We have to process the loaded data so that everything fits.
        loaded_result._process_loaded_obervables(markov_chain_data)

        # Load energy, if requested.
        if read_energy:
            try:
                loaded_result.energies = []
                energy_file = open(
                    os.path.join(markov_chain_id, markov_chain_id + "_energies.log"), "r")
                lines = energy_file.readlines()
                for line in lines:
                    if "step" not in line:
                        loaded_result.energies.append(float(line.split()[1]))

            except FileNotFoundError:
                print("Could not find energy file.")

        return loaded_result

    @staticmethod
    def _load_files(markov_chain_id, path_to_folder):
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
        return markov_chain_data, additonal_observables

    def _process_loaded_obervables(self, markov_chain_data):
        self.steps_evolved = markov_chain_data["metadata"][
            "steps_evolved"]
        self.accepted_steps = markov_chain_data["metadata"][
            "accepted_steps"]
        for entry in markov_chain_data["averaged_observables"].keys():
            if entry == "ion_ion_energy" or entry == "total_energy":
                self.observables[entry] = markov_chain_data["averaged_observables"][entry]

            if entry == "rdf" or entry == "tpcf" or entry == "static_structure_factor":
                filename = os.path.join(folder_to_load,
                                        markov_chain_data["averaged_observables"][entry])
                with open(filename, 'rb') as handle:
                    self.observables[entry] = pickle.load(handle)


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

