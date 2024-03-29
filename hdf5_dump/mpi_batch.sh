#!/bin/bash -l                                                                                                                                              
                                                                                                                                                             
#                                                                                                                                                            
#SBATCH --ntasks 15                                                                                                                                           
#SBATCH --cpus-per-task 1                                                                                                                                   
#SBATCH -o L2p8_m9_z1.5_output_file.%J.out                                                                                                                        
#SBATCH -e L2p8_m9_z1.5_error_file.%J.err                                                                                                                         
#SBATCH -p cosma8                                                                                                                                            
#SBATCH -A dp004                                                                                                                                             
#SBATCH -t 13:00:00                                                                                                                                          
#SBATCH --mail-type=ALL                          # notifications for job done & fail                                                                         
#SBATCH --mail-user=lilia.correamagnus@postgrad.manchester.ac.uk                                                                                             
                                                                                                                                                             
module purge
#load the modules used to build your program.                                                                                                                \
module load python/3.6.5
module load gnu_comp/7.3.0
module load openmpi/3.0.1
module load parallel_hdf5/1.10.3
                                                                                                                                                            
mpirun -np 15 python3 smr_saver.py
