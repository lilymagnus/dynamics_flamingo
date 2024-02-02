#!/bin/bash -l                                                                                                                                              
                                                                                                                                                             
#                                                                                                                                                            
#SBATCH --ntasks 60                                                                                                                                           
#SBATCH --cpus-per-task 1                                                                                                                                   
#SBATCH -o standard_output_file.%J.out                                                                                                                       
#SBATCH -e standard_error_file.%J.err                                                                                                                        
#SBATCH -p cosma8                                                                                                                                            
#SBATCH -A dp004                                                                                                                                             
#SBATCH -t 05:00:00                                                                                                                                          
#SBATCH --mail-type=ALL                          # notifications for job done & fail                                                                         
#SBATCH --mail-user=lilia.correamagnus@postgrad.manchester.ac.uk                                                                                             
                                                                                                                                                             
module purge
#load the modules used to build your program.                                                                                                                \
                                                                                                                                                              
module load intel_comp/2018
module load openmpi/3.0.1
module load python/3.10.12

mpirun -np 60 python3 map_saver.py
