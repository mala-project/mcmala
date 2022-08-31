import os

from ase.io import read, write
import mcmala
from mcmala.simulation.espresso_mc import EspressoMC
from mala.datahandling.data_repo import data_repo_path


data_path = os.path.join(os.path.join(data_repo_path, "Be2"), "training_data")

input_data = {
    "calculation": 'scf',
    "restart_mode": 'from_scratch',
    "pseudo_dir": '/home/fiedlerl/tools/pslibrary/pbe/PSEUDOPOTENTIALS/',
    "tstress": True,
    "tprnfor": True,
    "outdir": 'temp',
    "prefix": 'Be',
    "ibrav": 0,
    "nbnd": 4,
    "ecutwfc": 40,
    "ecutrho": 160,
    "nosym": True,
    "noinv": True,
    "occupations": 'smearing',
    "degauss": 0.0018874,
    "smearing": 'fermi-dirac',
    "ntyp": 1,
    "nat": 2,
    "conv_thr": 0.02,
    "mixing_mode": 'plain',
    "mixing_beta": 0.1
}
pseudopotentials = {"Be": "Be.pbe-n-rrkjus_psl.1.0.0.UPF"}
kpts = (4, 4, 4)
mcmala.use_mpi()

# Initial configuration is one of the training snapshots.
initial_configuration = read(os.path.join(data_path, "Be_snapshot1.out"),
                             format="espresso-out")

evaluator = EspressoMC(initial_configuration, input_data,
                              pseudopotentials, kpts)

# Atomic displacer means one atom at a time is randomly displaced.
suggester = mcmala.AtomDisplacer(0.2)

# We need to use the same temperature training data was calculated at.
# Then we can run.
simulation = mcmala.MarkovChain(298.0, evaluator, suggester,
                                initial_configuration, ensemble="debug",
                                markov_chain_id="ex06")
simulation.run(20, print_energies=True, checkpoints_after_steps=2,
               log_energies=True, log_trajectory=True)
