from mcmala.common.parallelizer import get_size, printout, get_rank, \
    split_comm, get_world_comm, barrier
from mcmala.montecarlo.markovchain import MarkovChain
import numpy as np
from ase.units import kB
from random import random

class ParallelTempering:

    # TODO: Fix the handling of evaluator input
    def __init__(self, temperatures, evaluator_class, evaluator_input_file,
                 configuration_suggester,
                 initial_configuration, exchange_after_step, calculate_observables_after_steps=1,
                 markov_chain_id="mcmala_default", additonal_observables=[],
                 ensemble="nvt", equilibration_steps=0):
        self.world_comm = get_world_comm()
        self.exchange_after_step = exchange_after_step
        self.temperatures = temperatures

        # Create the communicators and output the parallelization scheme.
        number_of_ranks = get_size()
        number_of_instances = len(temperatures)
        remainder = number_of_ranks % number_of_instances
        ranks_per_instance = number_of_ranks // number_of_instances
        printout("Running parallel tempering with {0} instances on {1} "
                 "rank(s), {2} ranks per instance."
                 .format(number_of_instances, number_of_ranks,
                         ranks_per_instance))
        self.local_roots = []
        if remainder:
            printout("Number of instances is not an integer divider of number"
                     " of ranks. Thus, some instances will move slower, "
                     "reducing overall performance.")

        self.instance_number = None
        key = None
        for idx, temperature in enumerate(temperatures):
            first_rank = idx*ranks_per_instance
            last_rank = (idx+1)*ranks_per_instance
            self.local_roots.append(first_rank)
            if (idx+1) == number_of_instances and last_rank != number_of_ranks:
                last_rank = number_of_ranks
            if last_rank > get_rank() >= first_rank:
                self.instance_number = idx
                key = get_rank()-first_rank

        split_comm(self.instance_number, key)
        print(get_rank(), get_rank(self.world_comm))
        # Create the Markov chain. Each instance of the ParallelTempering
        # class will have one with the right (local) communicator.

        # TODO: Find a smart way (maybe in the evaluator class itself)
        # to handle the input files.
        self.temperature = temperatures[self.instance_number]
        evaluator = evaluator_class(inputfile=evaluator_input_file,
                                    temperature=self.temperature,
                                    temp_folder=markov_chain_id+str(self.temperature)+"/temp")
        self.markov_chain = MarkovChain(self.temperature,
                                        evaluator, configuration_suggester,
                                        initial_configuration,
                                        calculate_observables_after_steps=calculate_observables_after_steps,
                                        markov_chain_id=markov_chain_id+str(self.temperature),
                                        additonal_observables=additonal_observables, ensemble=ensemble,
                                        equilibration_steps=equilibration_steps)

    def run(self, steps_to_evolve, print_energies=False,
            save_run=True, log_energies=False, log_trajectory=False,
            checkpoints_after_steps=0):

        # Set up the run for all Markov chains.
        self.markov_chain._setup_run(steps_to_evolve, log_energies,
                                     log_trajectory)

        # Run for the number of steps to be evolved, but only in short
        # bursts. In between, check if an exchange needs/can be done.
        current_step = 0
        starting_swap_at_zero = True
        while current_step < steps_to_evolve:
            # Run the Markov chain of this instance for
            self.markov_chain._run(current_step+self.exchange_after_step, print_energies, log_trajectory,
                                   log_energies, checkpoints_after_steps, save_run)

            # Wait till all Markov chains have finished.
            barrier(self.world_comm)
            if get_rank(self.world_comm) == 0:
                printout(current_step, "PARALLEL TEMPERING EXCHANGE START.")
            start_value = 0 if starting_swap_at_zero else 1
            starting_swap_at_zero = not starting_swap_at_zero
            if get_rank() == 0:
                for i in range(0, len(self.temperatures)):
                    instance1 = start_value+2*i
                    instance2 = start_value+2*i+1
                    if instance2 < len(self.temperatures):
                        rank1 = self.local_roots[instance1]
                        rank2 = self.local_roots[instance2]

                        if get_rank(self.world_comm) == rank1:
                            other_energy = np.empty([1], dtype=np.float64)
                            self.world_comm.Recv(other_energy, source=rank2, tag=rank1)
                            other_energy = other_energy[0]
                            print(self.markov_chain.current_energy)
                            print("Rank {0} and {1} with temperatures {2} and {3} comparing energies"
                                  " {4} and {5}".format(rank1, rank2,
                                                        self.temperatures[instance1],
                                                        self.temperatures[instance2],
                                                        self.markov_chain.current_energy,
                                                        other_energy))
                            do_exchange = self.\
                                __check_acceptance(self.markov_chain.current_energy,
                                                  other_energy,
                                                   self.temperatures[instance1],
                                                   self.temperatures[instance2],
                                                   rank1, rank2)
                            self.world_comm.Send(np.array(do_exchange, dtype=np.bool8), dest=rank2, tag=rank1)
                            if do_exchange:
                                # Receive from partner rank.
                                configuration_length = len(self.markov_chain.configuration)*3
                                new_positions = np.empty((configuration_length), dtype=np.float64)
                                self.world_comm.Recv(new_positions,
                                                     source=rank2, tag=rank1)

                                # Send to partner rank.
                                old_positions = np.reshape(self.markov_chain.configuration.get_positions(),
                                                           (configuration_length))
                                self.world_comm.Send(old_positions, dest=rank2, tag=rank1)
                                self.world_comm.Send(np.array(self.markov_chain.current_energy), dest=rank2, tag=rank1)

                                # Set the values.
                                self.markov_chain.configuration.set_positions(np.reshape(new_positions, (len(self.markov_chain.configuration), 3)))
                                self.markov_chain.current_energy = other_energy

                        if get_rank(self.world_comm) == rank2:
                            self.world_comm.Send(np.array(self.markov_chain.current_energy), dest=rank1, tag=rank1)
                            do_exchange = np.empty([1], dtype=np.bool8)
                            self.world_comm.Recv(do_exchange, source=rank1, tag=rank1)
                            do_exchange = do_exchange[0]
                            if do_exchange:
                                # Send to partner rank.
                                configuration_length = len(self.markov_chain.configuration)*3
                                old_positions = np.reshape(self.markov_chain.configuration.get_positions(),
                                                           (configuration_length))
                                self.world_comm.Send(old_positions, dest=rank1, tag=rank1)

                                # Receive from partner rank.
                                new_positions = np.empty((configuration_length), dtype=np.float64)
                                other_energy = np.empty([1], dtype=np.float64)
                                self.world_comm.Recv(new_positions, source=rank1, tag=rank1)
                                self.world_comm.Recv(other_energy, source=rank1, tag=rank1)

                                # Set the values.
                                self.markov_chain.configuration.set_positions(np.reshape(new_positions, (len(self.markov_chain.configuration), 3)))
                                self.markov_chain.current_energy = other_energy

            barrier(self.world_comm)
            current_step += self.exchange_after_step
            if get_rank(self.world_comm) == 0:
                printout(current_step, "PARALLEL TEMPERING EXCHANGE END.")

        self.markov_chain._wrap_up_run(log_energies, save_run, steps_to_evolve)

    def __check_acceptance(self, energy1, energy2, temperature1, temperature2,
                           rank1, rank2):
        deltaE = energy2-energy1
        deltaBeta = (1/(kB * temperature2))-(1/(kB * temperature1))
        exponent = deltaBeta*deltaE
        probability = np.min([1, np.exp(exponent)])
        randomNumber = random()
        print("Rank {0} and {1} comparison: {2}, {3}, {4}, {5}".format(rank1, rank2, probability, np.exp(exponent), randomNumber, not(probability < randomNumber)))
        # return True
        if probability < randomNumber:
            return False
        else:
            return True


