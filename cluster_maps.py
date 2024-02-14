import numpy as np
import h5py as h5
import unyt
import yt
from matplotlib.patches import Circle
from utils import snapshot, data_bin, paths, locate_gals
import unyt
from map_maker import maps
import cat_reader as cat
import matplotlib.pyplot as plt
#from swiftsimio.visualisation.projection import scatter
from mpl_toolkits.axes_grid1 import AxesGrid
from yt.funcs import get_pbar

group_path = "/cosma8/data/dp004/flamingo/Runs/"
run = "HYDRO_FIDUCIAL"
catalogue = "/SOAP"
size = 'L1000N'
res = '3600'

if res == '1800':
    mass_ = 'm9'
else:
    mass_ = 'm8'

def halo_finder(gap, z, res,min_mass, ds): 
    #find halos according to mag gap and mass criteria                                                                                                                                                        
    #look through presaved dataset                                                                                                                                                                                                                                            
    #ds = yt.load("saved_data/" +size + '/' + res +"/" + mass_ + "_mag_200m_z"+str(z)+"_1e14_50kpc.h5")
    data = ds.data
    
    m14 = data['M14']
    halo_id = data['host_id']
    mass = data['M200m']
    
    '''
    Note this relation doesnt hold 
    for high resolution
    '''
    
    if gap > 1:
        mask = mass > min_mass
    else:
        mask = mass < min_mass
        
    hid = halo_id[mask]
    mag_gap = m14[mask] 
    m = mass[mask]
    print('finding halo id...')
    for i in range(len(hid)):
        if np.abs(mag_gap[i] - gap) <= 0.05:
            print(mag_gap[i])
            print(m[i]) 
            print(hid[i])
            return hid[i]

    raise Exception ('no magnitude gap near what has been input')

def locate_halo(center, CoP_list, hid_list, bool_, min_):
    pbar = yt.get_pbar('Comparing centers...', len(CoP_list[bool_]))
    #distance to corner of box                                                                                                                                                                          
    index = []
    for i,cop in enumerate(CoP_list[bool_]):
        pbar.update(i)
        if (np.abs(cop[0] - center[0]) < min_) & \
        (np.abs(cop[1] - center[1]) < min_) & \
        (np.abs(cop[2] - center[2]) < min_):
           index.append(i)

    return hid_list[bool_][index]

def plot_Xray(map, mapMin, mapMax):
    mapRes = 512
    fig,axs = plt.subplots(figsize=(6,6))
    image=axs.pcolormesh(map,cmap='jet',vmin=mapMin,vmax=mapMax)
    fig.tight_layout()
    plt.savefig('Xray.png',dpi=300)
    breakpoint()

def subhalo_density_map(map, mapMin, mapMax, cc, index_cluster):
    mapRes = 512
    fig,axs = plt.subplots(figsize=(6,6))
    rChoice = cc.R200m[index_cluster]
    xChoice = cc.CoP[index_cluster,0]
    yChoice = cc.CoP[index_cluster,1]
    maxRegion = unyt.unyt_quantity(1*rChoice,'Mpc')
    image=axs.pcolormesh(map,cmap='magma',vmin=mapMin,vmax=mapMax)
    print('halos mass %s' % cc.M200m[index_cluster])
    #plot cluster
    axs.add_patch(Circle((0.5*mapRes,0.5*mapRes),rChoice*mapRes/(2.*maxRegion.value),fill=False,color='yellow',linewidth=2))
    #plot BCG
    axs.add_patch(Circle((0.5*mapRes,0.5*mapRes),(unyt.unyt_quantity(50,'kpc')).to('Mpc')*mapRes/(2.*maxRegion.value),fill=False,color='white',linewidth=2))
    
    #VR ID is just the index+1 since the VR_ID list starts at 1                                                                                                                                                                               
    #cs = ['red', 'blue','green']
    idx_gals, mag_gals, gal_pos = locate_gals(cc, index_cluster, z=0, box=1000, radius='200m')
    breakpoint()
    for i in range(5):
        rsub= unyt.unyt_quantity(50,'kpc')
        xsub= gal_pos[i][0] - xChoice
        ysub= gal_pos[i][1] - yChoice
        xsub = (xsub +maxRegion.value)/(2.*maxRegion.value)
        ysub = (ysub +maxRegion.value)/(2.*maxRegion.value)
        axs.add_patch(Circle(( ysub*mapRes,  xsub*mapRes), rsub.to('Mpc')*mapRes/(2.*maxRegion.value),fill=False,linewidth=2, color='white'))
    
    fig.tight_layout()
    plt.savefig('lit_rev_maggap.png',dpi=300)
    breakpoint()

if __name__ == "__main__":
    z = 0
    catalogue_path, snapshot_path = paths(res, z, group_path, catalogue, run, size)
    cc = cat.catalogue(catalogue_path, apert='50')

    filename = 'saved_data/L1000N/' + res+ '/' + mass +'_mag_500c_z0_1e13_50kpc.h5'
    data = cat.h5_file(filename,store_type,data_type)
    
    #halo_ID = halo_finder(2, z, res, 1e15, ds)
    #breakpoint()
    halo_ID = 4
    #cop = [294.21158543, 203.14734543, 932.84095543]                                                                                                                                                                                         
    #fig, axes = plt.subplots(1,3,figsize=(30,10))                                                                                                                                                                                             
    #halo_ID = locate_halo(cop, cc.CoP, cc.VR_ID, bool_=cc.M200m > 1e14,min_=0.7) 
    map, mapMin, mapMax,_ = maps(int(halo_ID-1), cc.CoP, cc.R200m, snapshot_path, z, 8)
    subhalo_density_map(map, mapMin, mapMax, cc, int(halo_ID-1))
    
    #print(halo_ID)
    #print(cc.CoP[int(halo_ID - 1)])
    #plot_Xray(map,  mapMin, mapMax)
    #breakpoint()
    maptypes = [8,7]
    colors = ['magma', 'viridis']
    titles = ['star map', 'DM map']
    labels = [r'$\log_{10}(\Sigma_{\rm *}/\Sigma_{\rm *,max})$',
              r'$\log_{10}(\Sigma_{\rm DM}/\Sigma_{\rm DM,max})$']
              
    fig = plt.figure()
    grid = AxesGrid(
        fig,
        (0.075, 0.075, 0.85, 0.85),
        nrows_ncols = (1,2),
        axes_pad = 0.15,
        share_all = True,
        label_mode = 'L',
        cbar_location='bottom',
        cbar_mode = 'each',
        cbar_size='1%',
        cbar_pad='5%',
        )
    
    for i,map_type in enumerate(maptypes):
        map, mapMin, mapMax, _ = maps(int(halo_ID-1), cc.CoP, cc.R200m, snapshot_path, z, map_type)
        im = grid[i].imshow(map, cmap=colors[i], vmin=mapMin,vmax=mapMax)
        grid[i].axes.get_xaxis().set_visible(False)
        grid[i].axes.get_yaxis().set_visible(False)
        grid[i].set_title(titles[i])
        cbar = grid.cbar_axes[i].colorbar(im)
        cbar.set_label(labels[i])
        #grid.cbar_axes[i]
        #plt.colorbar(im, ax=grid[i].axes, location='bottom',orientation='horizontal')
        
    plt.savefig('test.png',dpi=300)
#Cosmadirac1999
