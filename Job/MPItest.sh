#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH -J MPItest
#SBATCH -o out/mpi.out
#SBATCH -e out/mpi.err
#SBATCH --time=00:10:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it



source /global/homes/l/lonappan/.bashrc

cd /global/u2/l/lonappan/workspace/dell/Job


mpirun -np $SLURM_NTASKS python mpitest.py

