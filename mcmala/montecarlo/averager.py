import json

import numpy as np


class Averager:
    def __init__(self):
        """Averager class used to average over multiple Markov chains."""
        # Observables.
        self.observables = {"total_energy": [],
                            "rdf": {"rdf": [], "dr": 0.0,
                                           "rMax": 0}
                            }

    def add_markov_chain(self, markov_chain_id):
        """
        Adds a Markov chain to the analysis.

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
                self.observables["rdf"]["dr"] = markov_chain_data\
                                                ["averaged_observables"]\
                                                ["rdf"]["dr"]
                self.observables["rdf"]["rMax"] = markov_chain_data\
                                                  ["averaged_observables"]\
                                                  ["rdf"]["rMax"]

            else:
                self.observables[entry].append(markov_chain_data
                                                  ["averaged_observables"]
                                                        [entry])


    # Properties (Observables)
    @property
    def total_energy(self):
        """Total energy of the system (in eV)."""
        return np.mean(self.observables["total_energy"])

    @property
    def rdf(self):
        """Total energy of the system (in eV)."""
        return np.mean(self.observables["rdf"]["rdf"]),
        self.observables["rdf"]["rMax"], self.observables["rdf"]["dr"]
