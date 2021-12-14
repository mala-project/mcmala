from mcmala import MonteCarloSimulation, IsingGrid, IsingModelConfigurations, \
    IsingModelEvaluator, boltzmannConstant
import matplotlib.pyplot as plt
import numpy as np

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
    simulation = MonteCarloSimulation(temperature/boltzmannConstant, evaluator,
                                      suggester, inital_configuration, 1)
    simulation.run(10000)
    energies.append(simulation.energy)
    print(temperature, simulation.energy)

# Visualize the result.
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(temperatures, energies, marker="o")
plt.show()
