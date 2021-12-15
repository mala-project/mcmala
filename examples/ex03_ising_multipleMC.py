from mcmala import MarkovChain, IsingGrid, IsingModelConfigurations, \
    IsingModelEvaluator, Averager
from ase.units import kB

"""
ex03_ising_multipleMC: Shows how multiple Markov chains can be used to estimate
observables.
"""

# Simple Ising Model.
inital_configuration = IsingGrid(20, initType="negative")
evaluator = IsingModelEvaluator(4.0)
suggester = IsingModelConfigurations()

# Perform a MC simulation at a certain temperature, using two Markov chains.
# Naturally, in production settings, these two runs would be in separate
# files. The Markov chains will have to have different IDs or elsewise,
# the second Markov chain will overwrite the first one.
simulation = MarkovChain(10.0 / kB, evaluator, suggester, inital_configuration,
                         markov_chain_id="ex03_mc01")
simulation.run(5000)

simulation = MarkovChain(10.0 / kB, evaluator, suggester, inital_configuration,
                         markov_chain_id="ex03_mc02")
simulation.run(5000)

# After having run these two, an Averager object can be used to calculate the
# observables.
averager = Averager()
averager.add_markov_chain("ex03_mc01")
averager.add_markov_chain("ex03_mc02")
print(averager.total_energy)
