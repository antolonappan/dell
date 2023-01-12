#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=haswell
#SBATCH --nodes=32
#SBATCH --ntasks=500
#SBATCH --cpus-per-task=1
#SBATCH -J NoFG
#SBATCH -o out/nofg.out
#SBATCH -e out/nofg.err
#SBATCH --time=02:00:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc
conda activate cmblens
cd /global/u2/l/lonappan/workspace/LBlens

export ini=LB_FG0.ini

#mpirun -np $SLURM_NTASKS python simulation.py $ini -maps 
#mpirun -np $SLURM_NTASKS python simulation.py $ini -noise
#mpirun -np $SLURM_NTASKS python filtering.py $ini -cinv
#mpirun -np $SLURM_NTASKS python quest.py $ini -qlms
#mpirun -np $SLURM_NTASKS python quest.py $ini -N0
mpirun -np $SLURM_NTASKS python quest.py $ini -RDN0
#mpirun -np $SLURM_NTASKS python quest.py $ini -qlms_input
#mpirun -np $SLURM_NTASKS python quest.py $ini -resp
