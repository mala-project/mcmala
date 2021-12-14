from mcmala import IsingGrid
import matplotlib.pyplot as plt

configuration = IsingGrid(20, initType="random")
configuration.visualize()
plt.show()
