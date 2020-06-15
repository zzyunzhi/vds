import os, subprocess, sys

def mpi_fork(n, bind_to_core=False):
    """Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children
    """
    if n<=1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        args = ["mpirun", "-np", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        return "parent"
    else:
        return "child"

def mpi_fork_run_as_root(n, bind_to_core=False):
    """Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children
    Allow run as root, which is required by docker.
    This is unadvertised.
    """
    if n<=1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        args = ["mpirun", "-np", str(n), "--allow-run-as-root"]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        return "parent"
    else:
        return "child"
