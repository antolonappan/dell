import mpi
import numpy as np
import curvedsky as cs

jobs = np.arange(mpi.size)

for i in jobs[mpi.rank::mpi.size]:
    print(f"Job {i} from rank {mpi.rank}")
          