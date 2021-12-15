from mcmala import MarkovChain, IsingGrid, IsingModelConfigurations, \
    IsingModelEvaluator
from ase.units import kB
import matplotlib.pyplot as plt

"""
ex01_ising_singleshot: Performs and visualizes one MC simulation of the 
Ising model at a specific temperature.
"""

# Simple Ising Model.
inital_configuration = IsingGrid(20, initType="negative")
evaluator = IsingModelEvaluator(4.0)
suggester = IsingModelConfigurations()

# Perform a MC simulation at a certain temperature, using only one Markov
# Chain.
simulation = MarkovChain(10.0/kB, evaluator, suggester, inital_configuration)
simulation.run(5000, print_energies=True, save_run=False)

# Visualize the result. The initial configuration is all negative.
simulation.configuration.visualize()
plt.show()
