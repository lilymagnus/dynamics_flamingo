#!/bin/bash -l                                                                                                                                                                                                                               
#                                                                                                                                                                                                                                           
#SBATCH --ntasks 15                                                                                                                                                                                                                          
#SBATCH --cpus-per-task=15
#SBATCH -o standard_output_file.acc.out                                                                                                                                                                                                      
#SBATCH -e standard_error_file.acc.err                                                                                                                                                                                                       
#SBATCH -p cosma8                                                                                                                                                                                                                            
#SBATCH -A dp004                                                                                                                                                                                                                             
#SBATCH -t 08:00:00                                                                                                                                                                                                                          
#SBATCH --mail-type=ALL                          # notifications for job done & fail                                                                                                                                                         
#SBATCH --mail-user=lilia.correamagnus@postgrad.manchester.ac.uk                                                                                                                                                                             
module purge
#load the modules used to build your program.                                                                                                                                                                                                
module load python/3.10.12

python3 map_saver.py

