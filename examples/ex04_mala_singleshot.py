import os

from ase.io import read
import mcmala
import mala
from mala import printout
from mala.datahandling.data_repo import data_repo_path

data_path = os.path.join(data_repo_path, "Be2")

"""
ex04_mala_singleshot: Shows how a Monte Carlo simulation can be done using
MALA as the evaluator. Please not that you have to have the environment
variable MALA_DATA_REPO set according to the MALA setup in order for this
example to run.
This example first trains a small neural network, and then uses it for a
MC run.
"""


# Trains a network.
def initial_training():
    ####################
    # PARAMETERS
    # All parameters are handled from a central parameters class that
    # contains subclasses.
    ####################

    test_parameters = mala.Parameters()
    test_parameters.data.data_splitting_type = "by_snapshot"
    test_parameters.data.input_rescaling_type = "feature-wise-standard"
    test_parameters.data.output_rescaling_type = "normal"
    test_parameters.network.layer_activations = ["ReLU"]
    test_parameters.running.max_number_epochs = 20
    test_parameters.running.mini_batch_size = 40
    test_parameters.running.learning_rate = 0.00001
    test_parameters.running.trainingtype = "Adam"

    ####################
    # DATA
    # Add and prepare snapshots for training.
    ####################

    data_handler = mala.DataHandler(test_parameters)

    # Add a snapshot we want to use in to the list.
    data_handler.add_snapshot("Be_snapshot1.in.npy", data_path,
                              "Be_snapshot1.out.npy", data_path,
                              add_snapshot_as="tr",
                              output_units="1/(eV*Bohr^3)")
    data_handler.add_snapshot("Be_snapshot2.in.npy", data_path,
                              "Be_snapshot2.out.npy", data_path,
                              add_snapshot_as="va",
                              output_units="1/(eV*Bohr^3)")
    data_handler.prepare_data()
    printout("Read data: DONE.")

    ####################
    # NETWORK SETUP
    # Set up the network and trainer we want to use.
    # The layer sizes can be specified before reading data,
    # but it is safer this way.
    ####################

    test_parameters.network.layer_sizes = [data_handler.input_dimension,
                                           100,
                                           data_handler.output_dimension]

    # Setup network and trainer.
    test_network = mala.Network(test_parameters)
    test_trainer = mala.Trainer(test_parameters, test_network, data_handler)
    printout("Network setup: DONE.")

    ####################
    # TRAINING
    # Train the network.
    ####################

    printout("Starting training.")
    test_trainer.train_network()
    printout("Training: DONE.")
    test_trainer.save_run("ex04",
                          additional_calculation_data=
                          os.path.join(data_path, "Be_snapshot1.out"))


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

    # Initial configuration is one of the training snapshots.
    initial_configuration = read(os.path.join(data_path,
                                              "Be_snapshot1.out"),
                                 format="espresso-out")

    # Atomic displacer means one atom at a time is randomly displaced.
    suggester = mcmala.AtomDisplacer(0.2)

    # We need to use the same temperature training data was calculated at.
    # Then we can run.
    simulation = mcmala.MarkovChain(298.0, evaluator, suggester,
                                    initial_configuration)
    simulation.run(20)


# initial_training()
run_mc()

