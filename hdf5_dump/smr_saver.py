'''
script to find and store the BCG stellar mass 
and 4th most massive satellite (in terms of stellar mass)
'''
import h5py as h5
import functools as ft
from tqdm import tqdm
import os, sys
import unyt
from mpi4py import MPI
#import yt                                                                                                                                                                   
import numpy as np
from tqdm import tqdm                                                                                                                                    
sys.path.append('../')
#from map_maker import maps
import cat_reader as cat
from utils import snapshot, data_bin, paths, locate_gals
group_path = "/cosma8/data/dp004/flamingo/Runs/"
run = "HYDRO_FIDUCIAL"
catalogue = "/SOAP"
size = 'L1000N'
res = '1800'
outputPath="/cosma8/data/dp004/dc-corr2/magnitude_gap/saved_data/" + size + "/" + res + "/stellar_mass_ratio/"
outputFile=outputPath+'m9_SMR_z1.5_R200m_1e14_50kpc.hdf5'

def check_bounds(host_cop, sub_cop,box):

    for i, di in enumerate(sub_cop):
        dx = di - host_cop[i]
        if (dx<-box/2):
            sub_cop[i] += box
        if dx > box/2:
            sub_cop[i] -= box

    return sub_cop

if __name__ == "__main__":
    z = 1.5
    catalogue_path, snapshot_path= paths(res, z, group_path, catalogue, size, run)
    cc = cat.catalogue(catalogue_path, apert='50')    
    
    gal_smass = cc.stellar_mass 
    mass_cut = cc.M200m > 1e14                                                                                              
    host_cut = (cc.host_ID == -1) & mass_cut
    VR_ID_hosts = cc.VR_ID[host_cut]
    gal_smass_hosts = gal_smass[host_cut]
    R_hosts = cc.R200m[host_cut]
    CoP_hosts = cc.CoP[host_cut]
    M_hosts = cc.M200m[host_cut]

    num_processes = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)                                                                                         
    f = h5.File(outputFile, 'w', driver='mpio', comm=MPI.COMM_WORLD)
    
    #f = h5.File(outputFile, 'w')
    h5group=f.create_group("Cosmology")
    attrName="Critical density [internal units]"
    h5group.attrs[attrName]=cc.rhoCrit
    attrName="H [internal units]"
    h5group.attrs[attrName]=cc.Hz
    attrName="H0 [internal units]"
    h5group.attrs[attrName]=cc.H0
    attrName="Omega_b"
    h5group.attrs[attrName]=cc.OmegaBaryon
    attrName="Omega_m"
    h5group.attrs[attrName]=cc.OmegaMatter
    attrName="Redshift"
    h5group.attrs[attrName]=cc.redshift
    attrName="h"
    h5group.attrs[attrName]=cc.h
    
    id_dset = f.create_dataset('VR_ID', np.shape(VR_ID_hosts), dtype='f')
    bgg_id_dset = f.create_dataset('bgg_VR_ID', np.shape(VR_ID_hosts), dtype='f')
    sat_id_dset = f.create_dataset('sat_VR_ID', np.shape(VR_ID_hosts), dtype='f')
    bgg_sm_dset = f.create_dataset('bgg_sm', np.shape(VR_ID_hosts), dtype='f')
    sat_sm_dset = f.create_dataset('sat_sm', np.shape(VR_ID_hosts), dtype='f')
    mass_dset = f.create_dataset('M200m', np.shape(VR_ID_hosts), dtype='f')
    pbar = tqdm(total=len(VR_ID_hosts)) 
    for i, vr_ID in enumerate(VR_ID_hosts):
        pbar.update(1)
        #if i > 1000:
        #    breakpoint()
        if i % num_processes == rank:
            host_idx = i
            host_key = vr_ID
            host_cop = CoP_hosts[i]
            sub_idx = np.where(cc.host_ID == host_key)[0]
            subh_smass = cc.stellar_mass[sub_idx]
            subh_coords = cc.CoP[sub_idx]
            subh_VR = cc.VR_ID[sub_idx]
            sort = np.argsort(subh_smass)

            subh_smass = subh_smass[sort]
            subh_coords = subh_coords[sort]
            subh_VR = subh_VR[sort]
        
            #only include subs that are within R200                                                                                                    
            #correct for box boundary 
            for j, sub_cop in enumerate(subh_coords):
                if (size == "L1000N"):
                    subh_coords[j] = check_bounds(host_cop, sub_cop, 1000) #boxsize 1000 Mpc                                               
                else:
                    subh_coords[j] = check_bounds(host_cop, sub_cop, 2800)
            dist_to_host_cop = subh_coords - host_cop #Mpc                                                                               
            radius = R_hosts[host_idx] * (1+cc.redshift) #Mpc                                                                               
            r_sqrd = np.sum(dist_to_host_cop**2, axis=1)

            subs_smass_R = subh_smass[np.where(r_sqrd < (radius)**2)[0]]
            subs_ids_R = subh_VR[np.where(r_sqrd < (radius)**2)[0]]
            
            #ignore subhalo lists that are too small                                                                                                                             
            if (len(subs_smass_R[subs_smass_R > 0]) < 4):
                id_dset[i] = -1
                sat_id_dset[i] = -1
                bgg_sm_dset[i] = -1
                sat_sm_dset[i] = -1
                mass_dset[i] = -1
            else:
                id_dset[i] = host_key
                mass_dset[i] = M_hosts[host_idx]
                bgg_4th = subs_smass_R[-3]
                bgg4th_id = subs_ids_R[-3]
                bgg_2nd = subs_smass_R[-1]
                bgg = gal_smass_hosts[host_idx]
                #if BCG < 2nd-BCG 
                if bgg < bgg_2nd:
                    bgg = subs_smass_R[-1]
                    bgg_2nd = gal_smass_hosts[host_idx]                                  
                    bgg_id = subs_ids_R[-1]
                else:
                    bgg_id = host_key

                bgg_id_dset[i] = bgg_id
                sat_id_dset[i] = bgg4th_id
                bgg_sm_dset[i] = bgg
                sat_sm_dset[i] = bgg_4th
                
    f.close()



