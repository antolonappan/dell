#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=haswell
#SBATCH --nodes=32
#SBATCH --ntasks=500
#SBATCH --cpus-per-task=1
#SBATCH -J Map_FG1
#SBATCH -o out/map_FG1.out
#SBATCH -e out/map_FG1.err
#SBATCH --time=00:30:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc
conda activate cmblens
cd /global/u2/l/lonappan/workspace/LBlens

export ini=LB_FG1.ini

#mpirun -np $SLURM_NTASKS python simulation.py $ini -maps 
#mpirun -np $SLURM_NTASKS python simulation.py $ini -noise
mpirun -np $SLURM_NTASKS python simulation.py $ini -beam
