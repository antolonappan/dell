#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=haswell
#SBATCH --nodes=32
#SBATCH --ntasks=500
#SBATCH --cpus-per-task=1
#SBATCH -J Filt_FG2
#SBATCH -o out/filt_FG2.out
#SBATCH -e out/filt_FG2.err
#SBATCH --time=00:30:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc
conda activate cmblens
cd /global/u2/l/lonappan/workspace/LBlens

export ini=LB_FG2.ini

mpirun -np $SLURM_NTASKS python filtering.py $ini -cinv