from mcmala import MonteCarloSimulation, IsingGrid, IsingModelConfigurations, \
    IsingModelEvaluator, boltzmannConstant
import matplotlib.pyplot as plt

# Simple Ising Model.
inital_configuration = IsingGrid(20, initType="negative")
evaluator = IsingModelEvaluator(4.0)
suggester = IsingModelConfigurations()

# Perform a MC simulation at a certain temperature, using only one Markov
# Chain.
simulation = MonteCarloSimulation(10.0/boltzmannConstant, evaluator,
                                  suggester, inital_configuration, 1)
simulation.run(5000, print_energies=True)

# Visualize the result. The initial configuration is all negative.
simulation.markov_chains[0].configuration.visualize()
plt.show()
