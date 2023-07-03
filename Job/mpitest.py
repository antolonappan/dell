from mpi4py import MPI

mpi = MPI
com = MPI.COMM_WORLD
rank = com.Get_rank()
size = com.Get_size()

import curvedsky as cs
import lenspyx
import pysm3
import numpy as np


print(f"Imports worked in {rank}/{size}")


jobs = np.arange(10)
for i in jobs[rank::size]:
    print(f"Iteration:{i} from processor:{rank}")