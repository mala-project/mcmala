import os

from ase.io import read
import mala
from mala.datahandling.data_repo import data_repo_path
import matplotlib.pyplot as plt
import mcmala

data_path = os.path.join(os.path.join(data_repo_path, "Be2"), "training_data")

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
def run_mc(network, params, input_scaler, output_scaler):
    # These parameter values were not necessary for training, but are
    # necessary for inference.
    params.targets.target_type = "LDOS"
    params.targets.ldos_gridsize = 11
    params.targets.ldos_gridspacing_ev = 2.5
    params.targets.ldos_gridoffset_ev = -5
    params.running.inference_data_grid = [18, 18, 27]

    params.descriptors.descriptor_type = "SNAP"
    params.descriptors.twojmax = 10
    params.descriptors.rcutfac = 4.67637
    params.targets.pseudopotential_path = os.path.join(data_repo_path, "Be2")

    # Specify how observables will be calculated.
    params.targets.ssf_parameters = {"number_of_bins": 100, "kMax": 12.0}

    # Create data handler and with it calculator.
    inference_data_handler = mala.DataHandler(params,
                                              input_data_scaler=input_scaler,
                                              output_data_scaler=output_scaler)
    evaluator = mala.ASECalculator(params, network, inference_data_handler,
                                   ["qe.out",
                                    os.path.join(data_path,
                                                 "Be_snapshot1.out")])

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
    simulation.run(20, print_energies=True)

    # And we use two Markov chains.
    simulation = mcmala.MarkovChain(298.0, evaluator, suggester,
                                    initial_configuration,
                                    markov_chain_id="ex05_02",
                                    equilibration_steps=10,
                                    ensemble="debug",
                                    additonal_observables=["ion_ion_energy",
                                                           "static_structure_factor"],
                                    calculate_observables_after_steps=2)
    simulation.run(20, print_energies=True)

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


params = mala.Parameters.load_from_file("ex04_params.pkl")
network = mala.Network.load_from_file(params, "ex04_network.pkl")
input_scaler = mala.DataScaler.load_from_file("ex04_iscaler.pkl")
output_scaler = mala.DataScaler.load_from_file("ex04_oscaler.pkl")

run_mc(network, params, input_scaler, output_scaler)

