"""Functions for safely printing in parallel."""
try:
    from mpi4py import MPI
except ModuleNotFoundError:
    pass


_use_mpi = False
comm = None
world_comm = None


def set_mpi_status(new_value):
    """
    Set the MPI status.

    By setting the horovod status via this function it can be ensured that
    printing works in parallel. The Parameters class does that for the user.

    Parameters
    ----------
    new_value : bool
        Value the horovod status has.

    """
    global _use_mpi
    _use_mpi = new_value
    if _use_mpi:
        global comm
        comm = MPI.COMM_WORLD
        global world_comm
        world_comm = MPI.COMM_WORLD

    # else:
    #     global comm
    #     comm = MockCommunicator()


def use_mpi():
    set_mpi_status(True)


def get_rank(_comm=None):
    """
    Get the rank of the current thread.

    Always returns 0 in the serial case.

    Returns
    -------
    rank : int
        The rank of the current thread.

    """
    if _comm is None:
        _comm = comm
    if _use_mpi:
        return _comm.Get_rank()
    return 0


def get_size(_comm=None):
    """
    Get the number of ranks.

    Returns
    -------
    size : int
        The number of ranks.
    """
    if _comm is None:
        _comm = comm
    if _use_mpi:
        return _comm.Get_size()
    else:
        return 1


# TODO: This is hacky, improve it.
def get_comm():
    """
    Return the MPI communicator, if MPI is being used.

    Returns
    -------
    comm : MPI.COMM_WORLD
        A MPI communicator.

    """
    return comm


def get_world_comm():
    """
    Return the MPI communicator, if MPI is being used.

    Returns
    -------
    comm : MPI.COMM_WORLD
        A MPI communicator.

    """
    return world_comm


def split_comm(color, key):
    global comm
    new_comm = comm.Split(color, key)
    comm = new_comm


def printout(*values, sep=' '):
    """
    Interface to built-in "print" for parallel runs. Can be used like print.

    Parameters
    ----------
    values
        Values to be printed.

    sep : string
        Separator between printed values.
    """
    outstring = sep.join([str(v) for v in values])

    if get_rank() == 0:
        print(outstring)


def barrier(_comm=None):
    if _comm is None:
        _comm = comm
    if _use_mpi:
        return _comm.Barrier()
    else:
        return
