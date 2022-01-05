import os

from ase.io import read
import mcmala
import mala
from mala import printout
from mala.datahandling.data_repo import data_repo_path

data_path = os.path.join(os.path.join(data_repo_path, "Be2"), "training_data")

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
    # Currently, the splitting in training, validation and test set are
    # done on a "by snapshot" basis. Specify how this is
    # done by providing a list containing entries of the form
    # "tr", "va" and "te".
    test_parameters.data.data_splitting_type = "by_snapshot"
    test_parameters.data.data_splitting_snapshots = ["tr", "va"]

    # Specify the data scaling.
    test_parameters.data.input_rescaling_type = "feature-wise-standard"
    test_parameters.data.output_rescaling_type = "normal"

    # Specify the used activation function.
    test_parameters.network.layer_activations = ["ReLU"]

    # Specify the training parameters.
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
                              "Be_snapshot1.out.npy", data_path)
    data_handler.add_snapshot("Be_snapshot2.in.npy", data_path,
                              "Be_snapshot2.out.npy", data_path)
    data_handler.prepare_data()
    printout("Read data: DONE.")

    ####################
    # NETWORK SETUP
    # Set up the network and trainer we want to use.
    # The layer sizes can be specified before reading data,
    # but it is safer this way.
    ####################

    test_parameters.network.layer_sizes = [data_handler.get_input_dimension(),
                                           100,
                                           data_handler.get_output_dimension()]

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

    ####################
    # SAVING
    # In order to be operational at a later point we need to save 4 objects:
    # Parameters, input/output scaler, network.
    ####################

    return test_network, test_parameters, data_handler.input_data_scaler,\
           data_handler.output_data_scaler


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
    simulation = mcmala.MarkovChain(298.0, evaluator, suggester,
                                    initial_configuration)
    simulation.run(20, print_energies=True, save_run=True)


network, params, input_scaler, output_scaler = initial_training()

# You can save and load all the MALA variables using the following commands,
# in case you want to play around with the MC part more.

# network.save_network("ex04_network.pkl")
# params.save("ex04_params.pkl")
# input_scaler.save("ex04_iscaler.pkl")
# output_scaler.save("ex04_oscaler.pkl")
# params = mala.Parameters.load_from_file("ex04_params.pkl")
# network = mala.Network.load_from_file(params, "ex04_network.pkl")
# input_scaler = mala.DataScaler.load_from_file("ex04_iscaler.pkl")
# output_scaler = mala.DataScaler.load_from_file("ex04_oscaler.pkl")

run_mc(network, params, input_scaler, output_scaler)

