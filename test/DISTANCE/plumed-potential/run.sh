#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --time=00:05:00


source ~/zqc/.bashrc
#module purge
#module load cmake/3.31.8
#module load plumed/2.9.3-for-dpmd-excelsior
#export lmp='/home/excelsior/apps/_compile/lammps-29Aug2024/src/lmp_mpi'
#mpirun lmp -in in.test-set0F
mpirun lmp -in in.test-2atoms-toys
python plot_hills.py in.test-2atoms-toys HILLS
