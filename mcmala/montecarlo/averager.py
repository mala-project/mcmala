"""Averager to extract information from multiple Markov chains."""
import json
import os

import numpy as np

from .markovchainresults import MarkovChainResults

class Averager:
    """Averager class used to average over multiple Markov chains."""

    def __init__(self, read_logs=False):
        # Observables.
        self.read_logs = read_logs
        self.markov_chain_results = []
        self.energies = {}

    def add_markov_chain(self, markov_chain_id, path_to_folder="./"):
        """
        Add a Markov chain to the analysis.

        Observables of this particular Markov chain will be used for averaging.

        Parameters
        ----------
        markov_chain_id : string
            String identifier of the Markov chain to be added to analysis.
        """
        # Open the Markov chain associated with this ID.
        self.markov_chain_results.append(
            MarkovChainResults.load_run(markov_chain_id, path_to_folder))

        # Try to read the energy file written by the Markov chain.
        if path_to_folder is None:
            folder_to_load = markov_chain_id
        else:
            folder_to_load = os.path.join(path_to_folder, markov_chain_id)
        if self.read_logs:
            try:
                energy_file = open(os.path.join(folder_to_load,
                                                markov_chain_id +
                                                "_energies.log", "r"))
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
        total_energy = 0.0
        for result in self.markov_chain_results:
            total_energy += result.total_energy
        return total_energy / len(self.markov_chain_results)

    @property
    def rdf(self):
        """Radial distribution function of system"""
        rdf = np.zeros_like(self.markov_chain_results[0].rdf)
        for result in self.markov_chain_results:
            rdf += result.rdf
        return rdf / len(self.markov_chain_results)

    @property
    def tpcf(self):
        """Three particle correlation function of system."""
        tpcf = np.zeros_like(self.markov_chain_results[0].tpcf)
        for result in self.markov_chain_results:
            tpcf += result.tpcf
        return tpcf / len(self.markov_chain_results)

    @property
    def static_structure_factor(self):
        """Static structure factor of system."""
        static_structure_factor = np.zeros_like(self.markov_chain_results[0].static_structure_factor)
        for result in self.markov_chain_results:
            static_structure_factor += result.static_structure_factor
        return static_structure_factor / len(self.markov_chain_results)

    @property
    def rdf_grid(self):
        return self.markov_chain_results[0].rdf_grid

    @property
    def tpcf_grid(self):
        return self.markov_chain_results[0].tpcf_grid

    @property
    def static_structure_factor_grid(self):
        return self.markov_chain_results[0].static_structure_factor_grid


    @property
    def ion_ion_energy(self):
        """Ion ion energy in eV"""
        ion_ion_energy = 0.0
        for result in self.markov_chain_results:
            ion_ion_energy += result.ion_ion_energy
        return ion_ion_energy / len(self.markov_chain_results)
