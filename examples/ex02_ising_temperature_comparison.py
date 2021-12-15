from mcmala import MarkovChain, IsingGrid, IsingModelConfigurations, \
    IsingModelEvaluator
from ase.units import kB
import matplotlib.pyplot as plt
import numpy as np

"""
ex02_ising_temperature_comparison: Runs a MC simulation of the Ising model
for multiple temperatures and gives the observables (currently only energy) 
for them.
"""

# Calculate energies over a range of temperatures.
temperatures = list(np.arange(2.0, 30.0, 2.0))
energies = []

for temperature in temperatures:
    # Simple Ising Model.
    inital_configuration = IsingGrid(20, initType="negative")
    evaluator = IsingModelEvaluator(4.0)
    suggester = IsingModelConfigurations()

    # Perform a MC simulation at a certain temperature, using only one Markov
    # Chain.
    simulation = MarkovChain(temperature/kB, evaluator, suggester,
                             inital_configuration)
    simulation.run(10000, save_run=False)
    energies.append(simulation.energy)
    print(temperature, simulation.energy)

# Visualize the result.
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(temperatures, energies, marker="o")
plt.show()
