import numpy as np
import mpi

vec = np.ones(10)
if mpi.rank < 2:
    vec = np.zeros_like(vec)

if mpi.rank == 0:
    total = np.zeros_like(vec)
else:
    total = None

mpi.com.Reduce(vec,total,op=mpi.com.SUM,root=0)

mpi.barrier()

print(total)

