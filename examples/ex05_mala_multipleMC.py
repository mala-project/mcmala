import os

from ase.io import read
import mala
from mala.datahandling.data_repo import data_repo_path
import matplotlib.pyplot as plt
import mcmala

data_path = os.path.join(data_repo_path, "Be2")

"""
ex05_mala_multipleMC: Shows how a Monte Carlo simulation can be done using
MALA as the evaluator. In comparison to ex04, this example uses multiple
Markov chains, and calculates some additional observables.
Please note that you have to have the environment variable MALA_DATA_REPO set
according to the MALA setup in order for this example to run.
Please run ex04 prior to running this example, as the network from ex04 is 
being used here.
"""


# Perform MC run.
def run_mc():
    # These parameter values were not necessary for training, but are
    # necessary for inference.
    evaluator = mala.MALA.load_model("ex04")
    evaluator.mala_parameters.targets.target_type = "LDOS"
    evaluator.mala_parameters.targets.ldos_gridsize = 11
    evaluator.mala_parameters.targets.ldos_gridspacing_ev = 2.5
    evaluator.mala_parameters.targets.ldos_gridoffset_ev = -5
    evaluator.mala_parameters.running.inference_data_grid = [18, 18, 27]
    evaluator.mala_parameters.descriptors.descriptor_type = "SNAP"
    evaluator.mala_parameters.descriptors.twojmax = 10
    evaluator.mala_parameters.descriptors.rcutfac = 4.67637
    evaluator.mala_parameters.targets.pseudopotential_path = os.path.join(data_repo_path, "Be2")

    # Specify how observables will be calculated.
    evaluator.mala_parameters.targets.ssf_parameters = {"number_of_bins": 100, "kMax": 12.0}

    # Initial configuration is one of the training snapshots.
    initial_configuration = read(os.path.join(data_path,
                                              "Be_snapshot1.out"),
                                 format="espresso-out")

    # Atomic displacer means one atom at a time is randomly displaced.
    suggester = mcmala.AtomDisplacer(0.2)

    # We need to use the same temperature training data was calculated at.
    # Then we can run.
    # Here we use a "debug" ensemble, which will accept each proposed
    # configuration to make sure we actually see something.
    # We also test the equilibration_steps option, which means the first
    # equilibration_steps are never used for observable calculation.
    # We also calculate a two more observables, but only for
    # every other configuration.
    simulation = mcmala.MarkovChain(298.0, evaluator, suggester,
                                    initial_configuration,
                                    markov_chain_id="ex05_01",
                                    equilibration_steps=10,
                                    ensemble="debug",
                                    additonal_observables=["ion_ion_energy",
                                                           "static_structure_factor"],
                                    calculate_observables_after_steps=2)
    simulation.run(20)

    # And we use two Markov chains.
    simulation = mcmala.MarkovChain(298.0, evaluator, suggester,
                                    initial_configuration,
                                    markov_chain_id="ex05_02",
                                    equilibration_steps=10,
                                    ensemble="debug",
                                    additonal_observables=["ion_ion_energy",
                                                           "static_structure_factor"],
                                    calculate_observables_after_steps=2)
    simulation.run(20)

    # Now to the averaging.
    averager = mcmala.Averager()
    averager.add_markov_chain("ex05_01")
    averager.add_markov_chain("ex05_02")
    print("Total energy: ", averager.total_energy)
    print("Ion-Ion energy: ", averager.ion_ion_energy)

    # Plot the static structure factor.
    plt.plot(averager.static_structure_factor_grid,
             averager.static_structure_factor)
    plt.show()


run_mc()

