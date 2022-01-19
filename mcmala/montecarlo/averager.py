"""Averager to extract information from multiple Markov chains."""
import json

import numpy as np


class Averager:
    """Averager class used to average over multiple Markov chains."""

    def __init__(self, read_logs=False):
        # Observables.
        self.observables = {"total_energy": [],
                            "rdf": {"rdf": [], "distances": None},
                            "static_structure_factor": {"static_structure_factor": [], "kpoints": None},
                            "tpcf": {"tpcf": [], "radii": None},
                            "ion_ion_energy": []
                            }
        self.read_logs = read_logs
        self.energies = {}

    def add_markov_chain(self, markov_chain_id):
        """
        Add a Markov chain to the analysis.

        Observables of this particular Markov chain will be used for averaging.

        Parameters
        ----------
        markov_chain_id : string
            String identifier of the Markov chain to be added to analysis.
        """
        # Open the Markov chain associated with this ID.
        with open(markov_chain_id + ".json", encoding="utf-8") as json_file:
            markov_chain_data = json.load(json_file)

        # TODO: Add some kind of consistency check here...

        # Add the observables.
        for entry in markov_chain_data["averaged_observables"].keys():
            if entry == "rdf":
                self.observables["rdf"]["rdf"].append(markov_chain_data
                                                      ["averaged_observables"]
                                                      ["rdf"]["rdf"])
                self.observables["rdf"]["distances"] = markov_chain_data\
                                                ["averaged_observables"]\
                                                ["rdf"]["distances"]
            if entry == "static_structure_factor":
                self.observables["static_structure_factor"]["static_structure_factor"].append(markov_chain_data
                                                      ["averaged_observables"]
                                                      ["static_structure_factor"]["static_structure_factor"])
                self.observables["static_structure_factor"]["kpoints"] = markov_chain_data\
                                                    ["averaged_observables"]\
                                                    ["static_structure_factor"]["kpoints"]
            if entry == "tpcf":
                self.observables["tpcf"]["tpcf"].append(markov_chain_data
                                                      ["averaged_observables"]
                                                      ["tpcf"]["tpcf"])
                self.observables["tpcf"]["radii"] = markov_chain_data\
                                                ["averaged_observables"]\
                                                ["tpcf"]["radii"]

            else:
                self.observables[entry].append(markov_chain_data
                                                  ["averaged_observables"]
                                                        [entry])
        # Try to read the energy file written by the Markov chain.
        if self.read_logs:
            try:
                energy_file = open(markov_chain_id + "_energies.log", "r")
                lines = energy_file.readlines()
                energy_list = []
                for line in lines:
                    if "total energy" not in line:
                        energy_list.append(float(line.split()[1]))
                self.energies[markov_chain_id] = energy_list
            except FileNotFoundError:
                print("Could not find log file for Markov chain",
                      markov_chain_id, "skipping reading of energy logs.")

    def get_energies_of_markov_chain(self, markov_chain_id):
        """Get the energies list for a certain markov chain."""
        return self.energies[markov_chain_id]

    # Properties (Observables)
    @property
    def total_energy(self):
        """Total energy of the system (in eV)."""
        return np.mean(self.observables["total_energy"])

    @property
    def rdf(self):
        """Radial distribution function of system"""
        return np.mean(self.observables["rdf"]["rdf"], axis=0), \
               self.observables["rdf"]["distances"]

    @property
    def tpcf(self):
        """Three particle correlation function of system."""
        return np.mean(self.observables["tpcf"]["tpcf"], axis=0), \
               self.observables["tpcf"]["radii"]

    @property
    def static_structure_factor(self):
        """Static structure factor of system."""
        return np.mean(self.observables["static_structure_factor"]["static_structure_factor"], axis=0), \
               self.observables["static_structure_factor"]["kpoints"]

    @property
    def ion_ion_energy(self):
        """Ion ion energy in eV"""
        return np.mean(self.observables["ion_ion_energy"])
