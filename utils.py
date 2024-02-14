import h5py as h5
import unyt
from scipy import spatial
import numpy as np
from tqdm import tqdm

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
        
        if round(float(redshift),2) == z:
            return i

    raise Exception('Redshift not found...')

def data_bin(data2, data1, bin_num, stats):
    #sort and bin data in terms of other data set
    #e.g bin mag gap (data1) in terms of mass (data2)

    array = tuple(zip(data2, data1)) 
    #sorting the tuple in order of data2 values                                                                                                                                                     
    sorted_list = sorted(array, key=lambda x:x[0])
    d2_sort, d1_sort = zip(*sorted_list)
    
    d2_list = list(d2_sort)
    d1_list = list(d1_sort)
                                                                                                                                                                                                                                                           
    bins = np.linspace(np.min(d2_list), np.max(d2_list), bin_num)
    n,_=np.histogram(d2_list,bins=bins)
    print(n)
    median = []
    bin_item = []
    boot_low = []
    boot_high = []
    perc_16 = []
    perc_84 = []
    pbar = tqdm(total = len(n))
    for i in range(len(n)):
        pbar.update(1)
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
        return median, d2_list, bins


def paths(res, z, group_path, catalogue, size, run):    
    #gets paths for SOAP data and ds for mag_gap/acc data
    box = size + res
    data_path = group_path + box + "/" + run

    snapshot_ID = snapshot(z, data_path, catalogue, res)
    halo_props_name = "/halo_properties_" + str(snapshot_ID).zfill(4) + ".hdf5"
    catalogue_path = data_path + catalogue + halo_props_name

    snapshot_name = "/flamingo_" + str(snapshot_ID).zfill(4)
    snapshot_path = data_path + "/snapshots" + snapshot_name + snapshot_name + ".hdf5"
    
    return catalogue_path, snapshot_path
    
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


def locate_gals(cc, idx_cluster, z, box, radius):
    halo_id = cc.VR_ID[idx_cluster]

    print(cc.CoP[idx_cluster])
    gal_lums = cc.subhalo_luminosities[:,2]

    #find index of subhalos brightest galaxy                                                                                                                                                                                                                                   
    sub_lums = gal_lums[cc.host_ID == halo_id]
    sub_VRs = cc.VR_ID[cc.host_ID == halo_id]
    sub_CoP = cc.CoP[cc.host_ID == halo_id]
    #sub_star_mass = cc.stellar_mass[cc.host_ID == halo_id]
    #sub_mass = cc.bound_subhalo_mass[cc.host_ID == halo_id]
    
    if radius == '500c':
        sub_ids = sub_VRs[region(cc.CoP[idx_cluster], sub_CoP, z, cc.R500c[idx_cluster], box)]
        lums = sub_lums[region(cc.CoP[idx_cluster], sub_CoP, z, cc.R500c[idx_cluster], box)]
        sub_pos = sub_CoP[region(cc.CoP[idx_cluster], sub_CoP, z, cc.R500c[idx_cluster], box)]
    #star_mass =  sub_star_mass[region(cc.CoP[idx_cluster], sub_CoP, z, cc.R200c[idx_cluster])]
    #tot_mass = sub_mass[region(cc.CoP[idx_cluster], sub_CoP, z, cc.R200c[idx_cluster])]
    elif radius == '200m':
        sub_ids = sub_VRs[region(cc.CoP[idx_cluster], sub_CoP, z, cc.R200m[idx_cluster], box)]
        lums = sub_lums[region(cc.CoP[idx_cluster], sub_CoP, z, cc.R200m[idx_cluster], box)]
        sub_pos = sub_CoP[region(cc.CoP[idx_cluster], sub_CoP, z, cc.R200m[idx_cluster], box)]
    return sub_ids, lums, sub_pos

#-------- dynamic quants ------------#
def centroid_shift(dis_from_cent=0, pixel_x=None, pixel_y=None,no_pixels=0, cc=None, _id_map_tuple_=None):
    _id = _id_map_tuple_[0]
    map_ = _id_map_tuple_[1]
    EM = 10**map_
    EM_flat = EM.flatten()
    #convert shift to Mpc                                                                                                                                    
    idx = int(float(_id)-1)
    pixel_length = (2 * cc.R500c[idx]) / no_pixels
    #for a number of increasing apertures find the centroid                                                                                                 
    N_ap = 8
    #take the mean of this                                                                                                                                  
    aperture_radii = np.linspace(0.15, 1, N_ap) * cc.R500c[idx]
    centroid_change = np.zeros(N_ap)
    centroid_x = np.zeros(N_ap)
    centroid_y = np.zeros(N_ap)
    for i in range(N_ap):
        subset = np.where(dis_from_cent * pixel_length < aperture_radii[i])[0]
        centroid_x[i] = np.nansum(EM_flat[subset] * pixel_x[subset] * pixel_length) / np.nansum(EM_flat[subset])
        centroid_y[i] = np.nansum(EM_flat[subset] * pixel_y[subset] * pixel_length) / np.nansum(EM_flat[subset])
        centroid_change[i] = np.sqrt(centroid_x[i]**2 + centroid_y[i]**2)

    avg_centroid_dist = np.nanmean(centroid_change)
    centroid_shift = np.sqrt(np.sum((centroid_change - avg_centroid_dist)**2)/(N_ap-1))
    #centroid_x & centroid_y needed for image                                                                                                                
    #centroid_shift Mpc                                                                                                                                      
    return centroid_shift, centroid_x, centroid_y

def concentration(maps):
    #map x,y radial locations onto Xray map                                                                                                                  
    m= 10**maps
    X,Y = np.meshgrid(np.arange(0,512,1),np.arange(0,512,1))
    center = (256,256)
    radius = 256
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    full_mask = dist_from_center <= radius
    part_mask = dist_from_center <= radius*0.15

    Lx = np.sum(m[full_mask])
    Lx_small = np.sum(m[part_mask])

    return Lx_small/Lx
