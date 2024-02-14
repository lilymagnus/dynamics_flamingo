"""
Saves cluster maps using multiprocessing
or MPI to hdf5 files

NOTE if MPI is used, need to load these modules:
module load python/3.6.5
module load gnu_comp/7.3.0
module load openmpi/3.0.1
module load parallel_hdf5/1.10.3
and comment out the yt imports
and use: "import pdb; pdb.set_trace()" to debug

#functional test:
halo_ids = [1,2,3,4]
meps = np.meshgrid(np.arange(-256,256,1),np.arange(-256,256,1))
meps = [meps[0],meps[0],meps[0], meps[0]]

num_processes = MPI.COMM_WORLD.size
rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)                                                                         
f = h5.File('parallel_test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
id_dset = f.create_dataset('test', np.shape(halo_ids), dtype='f')
map_dset = f.create_dataset('test2', np.shape(meps), dtype='f')

for i in range(len(halo_ids)):
    #maps(cc, snapshot_path, 0, _id=halo_id[i])                                                                                                      
    if i % num_processes == rank:
    id_dset[i] = halo_ids[i]
    map_dset[i] = meps[i]

f.close()

"""

import h5py as h5
import functools as ft
from tqdm import tqdm
import os, sys
import unyt
from mpi4py import MPI
#import yt
import numpy as np
from tqdm import tqdm
#yt.enable_parallelism()
#from yt.funcs import get_pbar
sys.path.append('../')
from map_maker import maps
import cat_reader as cat
from utils import snapshot, data_bin, paths, locate_gals
group_path = "/cosma8/data/dp004/flamingo/Runs/"
run = "HYDRO_FIDUCIAL"
catalogue = "/SOAP"
size = 'L1000N'
res = '1800'
outputPath="/cosma8/data/dp004/dc-corr2/magnitude_gap/saved_data/" + size + "/" + res + "/" 
outputFile=outputPath+'Xray_maps_z0.5_R500c.hdf5'


def save_hdf5_map(map_list, cc, halo_ids):
    
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
    
    
    h5dset=h5file.create_dataset("VR_ID",np.shape(halo_ids),dtype='f',compression="gzip")
    h5dset[:]=halo_ids
    
    h5dset=h5file.create_dataset("Xray_maps",np.shape(map_list),dtype='f',compression="gzip")
    
    h5dset[:,:]=map_list
    h5file.close()
    print("------------------------- SAVING COMPLETE -------------------------")

def halo_map_maker(cc=None, snapshot_path=None, z=0, _id=None):
    map_,_,_,_ = maps(int(_id-1), cc.CoP, cc.R500c, snapshot_path, z, 9)
    return map_

if __name__ == "__main__":
    #breakpoint()    
    #method = str(input('use MPI or multithreads (mthreads)? '))
    
    method = 'MPI'
    z = 0.5
    catalogue_path, snapshot_path = paths(res, z, group_path, catalogue, run, size, mass='200m',type_='mag', return_ds = False)
    cc = cat.catalogue(catalogue_path, apert='50')
    filename = 'm9_mag_200m_z' + str(z) + '_1e14_50kpc.h5'

    #import pdb; pdb.set_trace()
    path = ("/cosma8/data/dp004/dc-corr2/magnitude_gap/saved_data/" + size + '/' + res + "/" + filename)
    h5file = h5.File(path, 'r')
    h5dset = h5file['data']['host_id']
    halo_ids = h5dset[...]
    h5file.close()
    
    if method == 'mthreads':
        #save maps in hdf5 file here                                                                                                                                                                                                             
        pool = mp.Pool(processes=15)
        map_list = list(tqdm(pool.map(ft.partial(halo_map_maker,cc, snapshot_path, z), halo_ids)))
        pool.close()
        pool.join()
        save_hdf5_map(np.array(map_list), cc, np.array(halo_ids))
        quit()
    
    if method == 'MPI':
        #import pdb; pdb.set_trace() 
         
        num_processes = MPI.COMM_WORLD.size
        rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)
        f = h5.File(outputFile, 'w', driver='mpio', comm=MPI.COMM_WORLD)
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
        
        id_dset = f.create_dataset('VR_ID', np.shape(halo_ids), dtype='f')
        map_dset = f.create_dataset('Xray_maps', (len(halo_ids),512,512), dtype='f')
        pbar = tqdm(total=len(halo_ids))
        for i in range(len(halo_ids)):
            pbar.update(1)
            if i % num_processes == rank:
                id_dset[i] = halo_ids[i]
                map_dset[i] = halo_map_maker(cc, snapshot_path, z, _id=halo_ids[i])

        f.close()

