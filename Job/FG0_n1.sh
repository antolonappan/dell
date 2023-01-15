#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=haswell
#SBATCH --nodes=8
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=1
#SBATCH -J FG0_n1
#SBATCH -o out/fg0_n1.out
#SBATCH -e out/fg0_n1.err
#SBATCH --time=00:30:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc
conda activate cmblens
cd /global/u2/l/lonappan/workspace/LBlens

export ini=LB_FG0_n1.ini

#mpirun -np $SLURM_NTASKS python simulation.py $ini -maps 
#mpirun -np $SLURM_NTASKS python simulation.py $ini -noise
#mpirun -np $SLURM_NTASKS python filtering.py $ini -cinv
#mpirun -np $SLURM_NTASKS python quest.py $ini -qlms
#mpirun -np $SLURM_NTASKS python quest.py $ini -N0
#mpirun -np $SLURM_NTASKS python quest.py $ini -qlms_input
#mpirun -np $SLURM_NTASKS python quest.py $ini -resp
