import h5py as h5
import unyt
from scipy import spatial
import numpy as np
import yt
from yt.funcs import get_pbar
from swiftsimio.visualisation.smoothing_length_generation import generate_smoothing_lengths
from swiftsimio import mask
from swiftsimio import load

def snapshot(z, data_path, catalogue, res):
    start = 0
        
    print('...locating redshift..')
    if (res == "3600") or (res =='5040'):
        start = 78
    elif (res == "1800") or (res == "0900"):
        start = 77

    for i in range(start,0,-1):
        if (res == "3600") and i == 58:
            continue
        if (res == '5040')  and (i < 10):
            continue
        if (res == '5040')  and (i == 42):
            continue
        if (res == '5040')  and (i % 2 != 0):
            continue
        snapshot_ID = i
        halo_props_name = "/halo_properties_" + str(snapshot_ID).zfill(4) + ".hdf5"
        catalogue_path = data_path + catalogue + halo_props_name
        h5file = h5.File(catalogue_path, 'r')
        # Read in halo properties                                                                                                                                           
        groupName = "/SWIFT/Cosmology/"
        h5group = h5file[groupName]
        attrName =  'Redshift'
        redshift = h5group.attrs[attrName]
        h5file.close()
        #snaps divided as z = 0.95 and 1.05 etc...
        print(redshift)
        
        '''
        #no z= 1.0 entry for 3600 box
        if (z == 1) and (res == '3600'):
            if round(float(redshift),1) == z:
                return i
        '''
        if round(float(redshift),2) == z:
            return i

    raise Exception('Redshift not found...')

def data_bin(data2, data1, bin_num, stats):
    #sort and bin data in terms of other data set
    #e.g bin mass data (data2) in terms of mag gap (data1)

    array = tuple(zip(data2, data1)) 
    #sorting the tuple in order of data2 values                                                                                                                                                     
    sorted_list = sorted(array, key=lambda x:x[0])
    d2_sort, d1_sort = zip(*sorted_list)
    
    d2_list = list(d2_sort)
    d1_list = list(d1_sort)
                                                                                                                                                                                                                                                           
    bins = np.linspace(np.min(d2_list), np.max(d2_list), bin_num)
    n,_=np.histogram(d2_list,bins=bins)
    
    median = []
    bin_item = []
    boot_low = []
    boot_high = []
    perc_16 = []
    perc_84 = []
    pbar = yt.get_pbar('Binning data data', len(n))
    for i in range(len(n)):
        pbar.update(i)
        section = list(np.array(d1_list[0:n[i]]).flatten())
        #m = list(np.array(m_copy[0:n[i]]).flatten())                                                                                                                                                                                                                        
        median.append(np.median(section))
        bin_item.append(section)
        if stats == True:
            boot = scipy.stats.bootstrap((section,), np.median,  
            confidence_level=0.95, method='percentile').confidence_interval           
            perc_16.append(np.percentile(section,16))             
            perc_84.append(np.percentile(section,84))             
            boot_low.append(boot[0])                                 
            boot_high.append(boot[1])                                                                                                                                                                                                                                           
        del d1_list[0:n[i]]

    if stats == False:
        return median, d2_list, bin_item


def paths(res, z, group_path, catalogue, run, size, mass, type_, return_ds = False):    
    #gets paths for SOAP data and ds for mag_gap/acc data
    box = size + res
    data_path = group_path + box + "/" + run

    snapshot_ID = snapshot(z, data_path, catalogue, res)
    halo_props_name = "/halo_properties_" + str(snapshot_ID).zfill(4) + ".hdf5"
    catalogue_path = data_path + catalogue + halo_props_name

    snapshot_name = "/flamingo_" + str(snapshot_ID).zfill(4)
    snapshot_path = data_path + "/snapshots" + snapshot_name + snapshot_name + ".hdf5"
    
    if return_ds == False:
         return catalogue_path, snapshot_path
    else:
        #type: acc or mag
        #mass: 200m or 500c
        if res == '5040':
            min_mass = '1e14'
        else:
            min_mass = '1e13'

        if (res == '1800') or (res == '5040') :
            filename = 'm9_' + type_ + '_' + mass + '_z' + str(z) + '_' + min_mass + '_50kpc.h5' 
            ds = yt.load("/cosma8/data/dp004/dc-corr2/magnitude_gap/saved_data/" + size + '/' + res + "/" + filename)
        else:                         
            if z == 1:
                z = 0.95
            filename = 'm8_' + type_ + '_' + mass + '_z' + str(z) + '_'+ min_mass + '_50kpc.h5'
            ds = yt.load("/cosma8/data/dp004/dc-corr2/magnitude_gap/saved_data/" + size + '/' + res + "/" + filename)
    
        return catalogue_path, snapshot_path, ds

def check_bounds(host_cop, sub_cop,box):

    for i, di in enumerate(sub_cop):
        dx = di - host_cop[i]
        if (dx<-box/2):
            sub_cop[i] += box
        if dx > box/2:
            sub_cop[i] -= box

    return sub_cop

def region(cent, pos_list, z, radius, box):
    #only include subs that are within R
    for j, sub_cop in enumerate(pos_list):
        pos_list[j] = check_bounds(cent, sub_cop, box) #boxsize 1000 Mpc
        
    dist_to_host_cop = cent - pos_list #Mpc               
    R = radius * (1 + z) #Mpc                                                                                                                                                                      
    r_sqrd = np.sum(dist_to_host_cop**2, axis=1)
    return np.where(r_sqrd < (R)**2)[0]


def locate_gals(cc, idx_cluster, z, box):
    halo_id = cc.VR_ID[idx_cluster]

    print(cc.CoP[idx_cluster])
    gal_lums = cc.subhalo_luminosities[:,2]

    #find index of subhalos brightest galaxy                                                                                                                                                                                                                                   
    sub_lums = gal_lums[cc.host_ID == halo_id]
    sub_VRs = cc.VR_ID[cc.host_ID == halo_id]
    sub_CoP = cc.CoP[cc.host_ID == halo_id]
    #sub_star_mass = cc.stellar_mass[cc.host_ID == halo_id]
    #sub_mass = cc.bound_subhalo_mass[cc.host_ID == halo_id]
     

    sub_ids = sub_VRs[region(cc.CoP[idx_cluster], sub_CoP, z, cc.R500c[idx_cluster], box)]
    lums = sub_lums[region(cc.CoP[idx_cluster], sub_CoP, z, cc.R500c[idx_cluster], box)]
    sub_pos = sub_CoP[region(cc.CoP[idx_cluster], sub_CoP, z, cc.R500c[idx_cluster], box)]
    #star_mass =  sub_star_mass[region(cc.CoP[idx_cluster], sub_CoP, z, cc.R200c[idx_cluster])]
    #tot_mass = sub_mass[region(cc.CoP[idx_cluster], sub_CoP, z, cc.R200c[idx_cluster])]
                                         
    return sub_ids, lums, sub_pos
