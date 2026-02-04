import sys
from driver import GaussDriver, OSVMP2Driver, ExitSignal, TimeOutSignal
import os
from mpi4py import MPI

#Set up MPI environment
comm = MPI.COMM_WORLD
nproc = comm.Get_size()   # Size of communicator
iproc = comm.Get_rank()   # Ranks in communicator
inode = MPI.Get_processor_name()    # Node where this MPI process runs
shm_comm = comm.Split_type(MPI.COMM_TYPE_SHARED) # Create sub-comm for each node
shm_rank = shm_comm.rank # rank index in sub-comm

def get_atomlist(xyz_name):
    atom_list = []
    with open(xyz_name, 'r') as xyz:
        lines = xyz.readlines()
        for idx, l in enumerate(lines):
            if idx > 1:
                atom_list.append(l.split()[0])
    return atom_list
xyz_name = sys.argv[1]
atom_list = get_atomlist(xyz_name)
basis = os.environ.get("basis", 'def2-svp').replace('-', '').lower()
port = int(os.environ.get("port", 31415))
mode = os.environ.get("mode", 'mp2').replace('-', '').lower()
if mode == 'dft':
    driver = GaussDriver(port, "127.0.0.1", atom_list, basis)
else:
    driver = OSVMP2Driver(port, "127.0.0.1", atom_list, basis)
while True:
    try:
        driver.parse()
    except ExitSignal as e:
        if mode == 'dft':
            driver = OSVMP2Driver(port, "127.0.0.1", atom_list, basis)
        else:
            driver = OSVMP2Driver(port, "127.0.0.1", atom_list, basis)
    except TimeOutSignal as e:
        if iproc == 0:
            print("Time out. Check whether the server is closed.")
        exit()
