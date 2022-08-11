#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=haswell
#SBATCH --nodes=16
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=2
#SBATCH -J Lensing
#SBATCH -o out/lensing.out
#SBATCH -e out/lensing.err
#SBATCH --time=00:10:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc
conda activate cmblens
cd /global/u2/l/lonappan/workspace/LBlens
export ini=LB_FG2.ini

mpirun -np $SLURM_NTASKS python simulation.py $ini  -lens