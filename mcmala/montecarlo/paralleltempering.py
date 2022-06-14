from mcmala.common.parallelizer import get_size, printout, get_rank, \
    split_comm, get_world_comm, barrier
from mcmala.montecarlo.markovchain import MarkovChain


class ParallelTempering:

    # TODO: Fix the handling of evaluator input
    def __init__(self, temperatures, evaluator_class, evaluator_input_file,
                 configuration_suggester,
                 initial_configuration, exchange_after_step, calculate_observables_after_steps=1,
                 markov_chain_id="mcmala_default", additonal_observables=[],
                 ensemble="nvt", equilibration_steps=0):
        self.world_comm = get_world_comm()
        self.exchange_after_step = exchange_after_step

        # Create the communicators and output the parallelization scheme.
        number_of_ranks = get_size()
        number_of_instances = len(temperatures)
        remainder = number_of_ranks % number_of_instances
        ranks_per_instance = number_of_ranks // number_of_instances
        printout("Running parallel tempering with {0} instances on {1} "
                 "rank(s), {2} ranks per instance."
                 .format(number_of_instances, number_of_ranks,
                         ranks_per_instance))
        if remainder:
            printout("Number of instances is not an integer divider of number"
                     " of ranks. Thus, some instances will move slower, "
                     "reducing overall performance.")

        self.instance_number = None
        key = None
        for idx, temperature in enumerate(temperatures):
            first_rank = idx*ranks_per_instance
            last_rank = (idx+1)*ranks_per_instance
            if (idx+1) == number_of_instances and last_rank != number_of_ranks:
                last_rank = number_of_ranks
            if last_rank > get_rank() >= first_rank:
                self.instance_number = idx
                key = get_rank()-first_rank
                break

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
        while current_step < steps_to_evolve:
            # Run the Markov chain of this instance for
            self.markov_chain._run(self.exchange_after_step, print_energies, log_trajectory,
                                   log_energies, checkpoints_after_steps, save_run)

            # Wait till all Markov chains have finished.
            barrier(self.world_comm)
            if get_rank() == 0:
                print("Doing the exchange.")
                print(get_rank(self.world_comm),
                      self.markov_chain.current_energy)
            quit()


            current_step += self.exchange_after_step




