'''
Script to store data in .h5 file using yt parallelisation
'''
import yaml
import yt
import unyt
import numpy as np
import h5py as h5
from collections import defaultdict
yt.enable_parallelism()
from yt.funcs import get_pbar
import cat_reader as cat

group_path = "/cosma8/data/dp004/flamingo/Runs/"
res = '5040'
size = "L2800N"
box = size + res
run = "HYDRO_FIDUCIAL"
data_path = group_path + box + "/" + run
catalogue = "/SOAP"

def check_z(z):
    start = 0 
    print('...locating redshift..')
    if (res == "3600") or (res =='5040'):
        start = 78
    elif (res == "1800") or (res == "0900"):
        start = 77
        
    for i in range(start,0,-1):
        if (res == "3600") and i == 58:
            continue
        if (res == '5040')  and (i not in [78,76,72,68,58,48,38,18,14,10]):
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
        if round(float(redshift),2) == z:
            return i
        
    raise Exception('Redshift not found...')

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
    #### read in data ####
    print('--------------------------------')
    print('--------------------------------')
    print('reading in data')
    snapshot_ID = check_z(z)
    print(snapshot_ID)
    halo_props_name = "/halo_properties_" + str(snapshot_ID).zfill(4) + ".hdf5"

    catalogue_path = data_path + catalogue + halo_props_name

    cc = cat.catalogue(catalogue_path,'50')

    #get the r-band lums
    gal_lums = cc.subhalo_luminosities[:,2]
    mass_cut = cc.M200m > 1e14
    #print(np.max(cc.M500c))
    #print('CoP:' + str(cc.CoP[np.argmax(cc.M500c),:]))
    #mass cut the halos so we only have field halo ids, masses and lums
    #host_cut = (cc.host_ID == -1)
    host_cut = (cc.host_ID == -1) & mass_cut
    VR_ID_hosts = cc.VR_ID[host_cut]
    gal_lums_hosts = gal_lums[host_cut]
    R_hosts = cc.R200m[host_cut]
    CoP_hosts = cc.CoP[host_cut]
    M_hosts = cc.M200m[host_cut]
    
    pbar = yt.get_pbar('Fetching subhalo data', len(VR_ID_hosts))
    storage = {}
            
    for my_storage, i in yt.parallel_objects(range(len(VR_ID_hosts)), storage=storage, dynamic=False):
        temp_dict = {'host_id':[],'lum_bgg':[],'lum_4thbgg':[], 'lum_2ndbgg':[], 'M200m':[],
                     'M12':[],'M14':[], 'bgg_VR_id':[], 'bgg4th_VR_id':[],'CoP':[]}
        
        host_key = VR_ID_hosts[i]
        pbar.update(i)

        host_idx = i
        host_cop = CoP_hosts[host_idx]
        sub_idx = np.where(cc.host_ID == host_key)[0]
        subh_lums = gal_lums[sub_idx]
        subh_coords = cc.CoP[sub_idx]
        subh_VR = cc.VR_ID[sub_idx]
        sort = np.argsort(subh_lums)
        
        subh_lums = subh_lums[sort]
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
        #radius = unyt.unyt_quantity(R200c_hosts[host_idx], 'Mpc')
        r_sqrd = np.sum(dist_to_host_cop**2, axis=1)
        subs_lums_R = subh_lums[np.where(r_sqrd < (radius)**2)[0]]
        subs_ids_R = subh_VR[np.where(r_sqrd < (radius)**2)[0]]

        #ignore subhalo lists that are too small
        if (len(subs_lums_R[subs_lums_R > 0]) < 4):            
            my_storage.result_id = host_key
            my_storage.result = 0
            continue
        else:
            temp_dict['host_id'] = host_key
            temp_dict['CoP'] = CoP_hosts[host_idx]
            temp_dict['M200m'] =  M_hosts[host_idx] 
            #luminosity of bgg is just luminosity of host halo at central 50kpc
            #to avoid getting offsets, take the brightest subhalo in the list
            #this should align with the CoP
            #breakpoint()
            bgg_4th = subs_lums_R[-3]
            bgg4th_id = subs_ids_R[-3]
            bgg_2nd = subs_lums_R[-1]
            bgg = gal_lums_hosts[host_idx]
            
            #if BCG < 2nd-BCG
            if bgg < bgg_2nd:
                bgg = subs_lums_R[-1]
                bgg_2nd = gal_lums_hosts[host_idx]
                #brightest subhalo will correspond to the central 
                bgg_id = subs_ids_R[-1]
            else:
                bgg_id = host_key
            
            temp_dict['bgg_VR_id'] = bgg_id
            temp_dict['bgg4th_VR_id'] = bgg4th_id

            temp_dict['lum_bgg'] = bgg
            temp_dict['lum_2ndbgg'] = bgg_2nd
            temp_dict['lum_4thbgg'] = bgg_4th
        
            #bgg is more negative, so mag gap is -(-bgg) + (-4th bgg)
            M14 = -(-2.5 * np.log10(bgg)) + (-2.5* np.log10(bgg_4th))
            M12 = -(-2.5 * np.log10(bgg)) + (-2.5* np.log10(bgg_2nd))
        
            if (np.isnan(M14)) or (np.isnan(M12)):
                my_storage.result_id = host_key
                my_storage.result = 0
                continue
            else:
                temp_dict['M14'] = float(M14)
                temp_dict['M12'] = float(M12)
                my_storage.result_id = host_key
                my_storage.result = temp_dict
            
            
    if yt.is_root():
        data = defaultdict(list)
        for key, value in storage.items():
            if value == 0:
                continue
            else:
                data['M200m'].append(value['M200m'])
                data['host_id'].append(value['host_id'])
                data['lum_bgg'].append(value['lum_bgg'])
                data['lum_4thbgg'].append(value['lum_4thbgg'])
                data['lum_2ndbgg'].append(value['lum_2ndbgg'])
                data['M14'].append(value['M14'])
                data['M12'].append(value['M12'])
                data['bgg_VR_id'].append(value['bgg_VR_id'])
                data['bgg4th_VR_id'].append(value['bgg4th_VR_id'])
                data['CoP'].append(value['CoP'])

        my_units = {'host_id':'','lum_bgg':'Lsun','lum_4thbgg':'Lsun', 'lum_2ndbgg':'Lsun', 'M200m':'Msun', 'M12':'','M14':'', 
                    'bgg_VR_id':'', 'bgg4th_VR_id':'', 'CoP':'Myr'}
        
        for field in data:
            data[field] = unyt.unyt_array(data[field],my_units[field])

        fake_ds = {'H0': cc.H0, 'om_L':cc.OmegaLambda, 'om_M':cc.OmegaMatter} 
        if (res == '1800') or (res == '5040'):
            m = 'm9'
        else:
            m = 'm8'
        yt.save_as_dataset(fake_ds, 'saved_data/'+ size + '/' + res + '/' + m + '_mag_200m_z' + str(z) + '_1e14_50kpc.h5', data= data) 
