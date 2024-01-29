'''
calculate the centroid offset, peak offset and the concentration
for all clusters above the resolution threshold for each box 

script then makes a corner plot of Gamma, c, log<w> and M14

NOTE: maps are in log10 and normalised by the highest valued pixel
such that the peak Xray pixel has value = 1
'''
from scipy.stats import norm
import corner
import numpy as np
import h5py as h5
import unyt
import yt
from matplotlib.patches import Circle
from utils import snapshot, data_bin, paths, locate_gals
import unyt
from map_maker import maps
import  multiprocessing as mp
#yt.enable_parallelism()
import cat_reader as cat
from yt.funcs import get_pbar
import matplotlib.pyplot as plt
from collections import defaultdict
import functools as ft
from tqdm import tqdm
 
group_path = "/cosma8/data/dp004/flamingo/Runs/"
run = "HYDRO_FIDUCIAL"
catalogue = "/SOAP"
size = 'L1000N'
res = '1800'

if res == '1800':
    mass_ = 'm9'
else:
    mass_ = 'm8'

def map_maker(cc=None, snapshot_path=None, z=0, _id=None):
    map_,_,_,_ = maps(int(_id-1), cc.CoP, cc.R500c, snapshot_path, z, 9)
    return map_

def centroid_shift(dis_from_cent=0, pixel_x=None, pixel_y=None,no_pixels=0, _id_map_tuple_=None):
    _id = _id_map_tuple_[0]
    map_ = _id_map_tuple_[1]
    EM = 10**map_
    EM_flat = EM.flatten()
    #convert shift to Mpc                                                                                                                                                                                                                    
    pixel_length = (2 * cc.R500c[int(float(_id)-1)]) / no_pixels
    #for a number of increasing apertures find the centroid                                                                                                                                                                                  
    N_ap = 8
    #take the mean of this                                                                                                                                                                                         
    aperture_radii = np.linspace(0.15, 1, N_ap) * cc.R500c[int(_id-1)]
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

def peak_shift(map_):
    #x and y are coords with origin at center = CoP
    #units issue, pdf should be around 0, not 1
    EM = 10**map_
    map_size = len(map_)
    idx = np.where(EM == EM.max())
    peak_dist_from_cent = np.sqrt(np.abs(idx[0] -  map_size/2)**2 + np.abs(idx[1] - map_size/2)**2)
    return peak_dist_from_cent / (len(map_)*0.5)


def plot(map_, halo_id, cc, centroid_x, centroid_y):
    index_cluster = int(halo_id - 1)
    mapRes = 512
    fig,axs = plt.subplots(figsize=(6,6))
    rChoice = cc.R500c[index_cluster]
    xChoice = cc.CoP[index_cluster,0]
    yChoice = cc.CoP[index_cluster,1]
    maxRegion = unyt.unyt_quantity(1*rChoice,'Mpc')
    mapMin=-6
    mapMax=0
    image=axs.pcolormesh(map_,cmap='magma',vmin=mapMin,vmax=mapMax)
    #plot cluster                                                                                                                                                                                                  
    axs.add_patch(Circle((0.5*mapRes,0.5*mapRes),rChoice*mapRes/(2.*maxRegion.value),fill=False,color='yellow',linewidth=2))
    #plot BCG                                                                                                                                                                                                      
    axs.add_patch(Circle((0.5*mapRes,0.5*mapRes),(unyt.unyt_quantity(50,'kpc')).to('Mpc')*mapRes/(2.*maxRegion.value),fill=False,color='white',linewidth=1))
    #plot X-ray centroid
    
    for i in range(len(centroid_x)):
        '''
        need to convert from -255 to 256 scale to 0 to 512 scale?
        '''
        if centroid_y[i] < 0: 
            x= centroid_y[i] + 255
        else:
            x= centroid_y[i] + 256
        if centroid_x[i] < 0:
            y= centroid_x[i] + 255
        else:
            y= centroid_x[i] + 256
        axs.add_patch(Circle((x,y),(unyt.unyt_quantity(15,'kpc')).to('Mpc')*mapRes/(2.*maxRegion.value),fill=False,color='red',linewidth=1))
    
    fig.tight_layout()
    plt.savefig('test_shift1.png',dpi=300)
    breakpoint()

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

def corner_plot(m14, c, gamma, offset):
    idx = np.argwhere((np.isfinite(c)) & (np.isfinite(gamma)) & (np.array(gamma) < 7))
    data = np.vstack([np.array(m14)[idx].flatten(), np.array(c)[idx].flatten(), np.array(gamma)[idx].flatten(), np.array(np.log10(offset))[idx].flatten()])
    figure = corner.corner(data.T,labels = ['M14','c',r'$\Gamma$',r'$log \langle w \rangle$'],show_titles = True,)
    plt.savefig('corner.png')

if __name__ == "__main__":                                                                                                                                                                                 
    z = 0
    catalogue_path, snapshot_path, ds = paths(res, z, group_path, catalogue, run, size, mass='200m',type_='mag', return_ds = True)
    cc = cat.catalogue(catalogue_path, apert='50')
    data = ds.data
    halo_ids = data['host_id']
    mag_gap = data['M14']

    #load pre-saved maps
    
    h5file=h5.File('/cosma8/data/dp004/dc-corr2/magnitude_gap/saved_data/L1000N/1800/Xray_maps_z0_R500c.hdf5')
    h5dset=h5file["Xray_maps"]
    maps = h5dset[...]
    
    X,Y = np.meshgrid(np.arange(-256,256,1),np.arange(-256,256,1))
    pixel_x = X.flatten()
    pixel_y = Y.flatten()
    dis_from_cent = np.sqrt(pixel_x**2 + pixel_y**2)


    #---------- peak shift and PDF ------------
    peak_shift(maps[0])                                                                                                                                                                                                                           
    pool = mp.Pool(processes=20)
    peak_offset = list(tqdm(pool.map(ft.partial(peak_shift), maps)))
    pool.close()
    pool.join()
    breakpoint()
    hist, bin_edges = np.histogram(peak_offset, bins=20)
    plt.plot(bin_edges[:-1],hist)
    plt.yscale('log')
    plt.ylabel('PDF')
    plt.xlabel('Distnace from centre [dimensionless]')
    plt.savefig('peak_pdf.png')
    #-------------------------------------
    '''
    #plot individual cluster 

    #larg_halos = np.where(data['M200m'] >= 1e15)[0]
    #centroid_shift,centroid_x, centroid_y = shift(512, halo_ids[larg_halos[30]], maps[larg_halos[30]], dis_from_cent, pixel_x, pixel_y)
    #plot(maps[larg_halos[30]], halo_ids[larg_halos[30]], cc, centroid_x, centroid_y)
    '''

    #calculate concentration                                                                                                                                                                                    
    pool = mp.Pool(processes=20)
    concent = list(tqdm(pool.map(ft.partial(concentration), maps)))
    pool.close()
    pool.join()
    
    #load accretion rates
    #DO NOT USE THE 1e14 FILE FOR ACC. RATES! 
    #FULL DATA IN THE 1e13
    filename = 'm9_acc_200m_z0_1e13_50kpc.h5'
    ds_acc = yt.load("saved_data/" + size + '/' + res + "/" + filename)
    acc = ds_acc.data['Gamma'][0]
    halo_id_acc = ds_acc.data['VR_ID'][0]
    
    #save corresponding quants here
    m14 = []
    gamma = []
    
    #shift((halo_ids[0],maps[0]), dis_from_cent, pixel_x, pixel_y,512)
    ind_list = np.arange(len(halo_ids))
    pool = mp.Pool(processes=20)
    outputs = list(tqdm(pool.map(ft.partial(centroid_shift, dis_from_cent, pixel_x, pixel_y, 512), list(zip(halo_ids,maps)))))
    pool.close()
    pool.join()

    centroid_shift = [result[0] for result in outputs]
    
    #---------- corner plot -----------
    #find corresponding m14 and gammas for corner plot
    pbar = tqdm(total=len(halo_ids))
    for i, _id in enumerate(halo_ids):
        if len(np.where(halo_id_acc == _id)[0]) == 0:
            breakpoint()
        pbar.update(1)
        m14.append(mag_gap[np.where(halo_ids == _id)[0][0]])
        gamma.append(acc[np.where(halo_id_acc == _id)[0][0]])
    
    
    #make corner plot
    corner_plot(m14, concent, gamma, centroid_shift)
    breakpoint() 
    
    
