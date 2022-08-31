from ase.io import read, write
from mala.datahandling.data_repo import data_repo_path
import mcmala
from mcmala.simulation.espresso_mc import EspressoMC
from mcmala.montecarlo.paralleltempering import ParallelTempering
import os
data_path = os.path.join(os.path.join(data_repo_path, "Be2"), "training_data")

mcmala.use_mpi()
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
    "mixing_beta": 0.1,
    "verbosity": "low"
}
pseudopotentials = {"Be": "Be.pbe-n-rrkjus_psl.1.0.0.UPF"}
kpts = (4, 4, 4)

suggester = mcmala.AtomDisplacer(0.2)
initial_configuration = read(os.path.join(data_path, "Be_snapshot1.out"),
                             format="espresso-out")

evaluator = EspressoMC(initial_configuration, input_data,
                              pseudopotentials, kpts)

parallel_temp = ParallelTempering([3000, 3500, 4000, 4500], 5,
                                  evaluator=evaluator,
                                  configuration_suggester=suggester,
                                  initial_configuration=initial_configuration,
                                  ensemble="debug",
                                  parallel_tempering_id="ex07")

parallel_temp.run(20, create_checkpoints=True, log_energies=True,
                  log_trajectory=True)

