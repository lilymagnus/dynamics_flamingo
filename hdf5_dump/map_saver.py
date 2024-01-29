"""
Saves cluster maps using multiprocessing
to hdf5 files
"""

import h5py as h5
import functools as ft
from tqdm import tqdm
import os, sys
import unyt
from mpi4py import MPI
sys.path.append('../')
from map_maker import maps
import cat_reader as cat
from utils import snapshot, data_bin, paths, locate_gals

group_path = "/cosma8/data/dp004/flamingo/Runs/"
run = "HYDRO_FIDUCIAL"
catalogue = "/SOAP"
size = 'L1000N'
res = '3600'

def save_hdf5_map(map_list, cc, halo_ids):
    outputPath="/cosma8/data/dp004/dc-corr2/magnitude_gap/saved_data/L1000N/3600/"
    outputFile=outputPath+'Xray_maps_z0_R500c.hdf5'
    h5file=h5.File(outputFile,'w')

    h5group=h5file.create_group("Cosmology")
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

    h5dset=h5file.create_dataset("VR_ID",halo_ids.shape,dtype='u8',compression="gzip")
    h5dset[:]=halo_ids

    h5dset=h5file.create_dataset("Xray_maps",map_list.shape,dtype='f',compression="gzip")

    h5dset[:,:]=map_list
    h5file.close()
    print("------------------------- SAVING COMPLETE -------------------------")


def map_maker(cc=None, snapshot_path=None, z=0, _id=None):
    map_,_,_,_ = maps(int(_id-1), cc.CoP, cc.R500c, snapshot_path, z, 9)
    return map_

if __name__ == "__main__":
    #breakpoint()
    method = str(input('use MPI or multithreads (mthreads)? '))
    
    z = 0
    catalogue_path, snapshot_path, ds = paths(res, z, group_path, catalogue, run, size, mass='200m',type_='mag', return_ds = True)
    cc = cat.catalogue(catalogue_path, apert='50')
    data = ds.data
    halo_ids = data['host_id']
    mag_gap = data['M14']
    
    if method == 'mthreads':
        #save maps in hdf5 file here                                                                                                                                                                                                             
        pool = mp.Pool(processes=15)
        map_list = list(tqdm(pool.map(ft.partial(map_maker,cc, snapshot_path, z), halo_ids)))
        pool.close()
        pool.join()
        save_hdf5_map(np.array(map_list), cc, np.array(halo_ids))
        quit()
    
    if method == 'MPI':
        
        num_processes = MPI.COMM_WORLD.size
        rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)
        f = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
        dset = f.create_dataset('test', (channels, n), dtype='f')
        

        for i in range(len(halo_ids)):
            map_maker(cc, snapshot_path, 0, _id=halo_id[i])
