import unyt
from swiftsimio.visualisation.smoothing_length_generation import generate_smoothing_lengths
from swiftsimio.visualisation.projection import scatter
from swiftsimio import mask
from swiftsimio import load
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

'''
Script addapted from Scott's cluster_cat.py
to make any type of map from this list:

# choice of map                                                      
# 1 - density                                                        
# 2 - emission measure                                               
# 3 - y-weighted temperature                                         
# 4 - m-weighted temperature                                         
# 5 - sl-weighted temperature                                        
# 6 - y/sl temperature ratio                                         
# 7 - stellar mass                                                                                                                                                                  
# 8 - dark matter mass     
# 9 - ROSAT 0.5-2 keV X-ray luminosity   
'''

def maps(clust_idx,CoP, haloRadius, flamingoFile, z, mapChoice):

    if mapChoice not in np.arange(1,10,1):
        raise Exception('mapChoice has to be between 1 and 9!')

    rChoice=haloRadius[clust_idx]*(1.+z)
    xChoice=CoP[clust_idx,0]
    yChoice=CoP[clust_idx,1]
    zChoice=CoP[clust_idx,2]
    
    xCen = unyt.unyt_quantity(xChoice,'Mpc')
    yCen = unyt.unyt_quantity(yChoice,'Mpc')
    zCen = unyt.unyt_quantity(zChoice,'Mpc')
    maxRegion = unyt.unyt_quantity(1*rChoice,'Mpc')
    maskRegion = mask(flamingoFile)

    #spatially mask the snapshot data around the cluster                                                                                                                                                                                                                            
    region=[[xCen-maxRegion,xCen+maxRegion],
            [yCen-maxRegion,yCen+maxRegion],
            [zCen-maxRegion,zCen+maxRegion]]
    maskRegion.constrain_spatial(region)

    #load the data for only the masekd region                                                                                                                                                                                                                                       
    data = load(flamingoFile,mask=maskRegion)

    if mapChoice==7:
        from swiftsimio.visualisation.smoothing_length_generation import generate_smoothing_lengths
        data.stars.smoothing_lengths = generate_smoothing_lengths(
            data.stars.coordinates,
            data.metadata.boxsize,
            kernel_gamma=1.8,
            neighbours=57,
            speedup_fac=2,
            dimension=3
        )

    if mapChoice==8:
        from swiftsimio.visualisation.smoothing_length_generation import generate_smoothing_lengths
        data.dark_matter.smoothing_lengths = generate_smoothing_lengths(
            data.dark_matter.coordinates,
            data.metadata.boxsize,
            kernel_gamma=1.8,
            neighbours=57,
            speedup_fac=2,
            dimension=3
        )
                                                            
    dx=data.gas.coordinates.value[:,0]-xChoice
    dy=data.gas.coordinates.value[:,1]-yChoice
    dz=data.gas.coordinates.value[:,2]-zChoice
                                                         
    h=data.gas.smoothing_lengths.value
    m=data.gas.masses.value
    t=data.gas.temperatures.value
    d=data.gas.densities.value
    lx=data.gas.xray_luminosities.ROSAT.value

    if mapChoice==7:
        dxstar=data.stars.coordinates.value[:,0]-xChoice
        dystar=data.stars.coordinates.value[:,1]-yChoice
        dzstar=data.stars.coordinates.value[:,2]-zChoice
        hstar=data.stars.smoothing_lengths.value
        mstar=data.stars.masses.value

    if mapChoice==8:
        dxdm=data.dark_matter.coordinates.value[:,0]-xChoice
        dydm=data.dark_matter.coordinates.value[:,1]-yChoice
        dzdm=data.dark_matter.coordinates.value[:,2]-zChoice
        hdm=data.dark_matter.smoothing_lengths.value
        mdm=data.dark_matter.masses.value
                                             
    ind=np.where((dx>-maxRegion.value)&(dx<maxRegion.value)&
             (dy>-maxRegion.value)&(dz<maxRegion.value)&
             (dz>-maxRegion.value)&(dz<maxRegion.value)&(t>1.e6))[0]
                                                                 
    dx=(dx[ind]+maxRegion.value)/(2.*maxRegion.value)
    dy=(dy[ind]+maxRegion.value)/(2.*maxRegion.value)
    dz=(dz[ind]+maxRegion.value)/(2.*maxRegion.value)
    h=h[ind]/(2.*maxRegion.value)
    
    lx=lx[ind]
    m=m[ind]
    t=t[ind]
    d=d[ind]

    if mapChoice==7:
        ind=np.where((dxstar>-maxRegion.value)&(dxstar<maxRegion.value)&
                 (dystar>-maxRegion.value)&(dzstar<maxRegion.value)&
                 (dzstar>-maxRegion.value)&(dzstar<maxRegion.value))[0]
        dxstar=(dxstar[ind]+maxRegion.value)/(2.*maxRegion.value)
        dystar=(dystar[ind]+maxRegion.value)/(2.*maxRegion.value)
        dzstar=(dzstar[ind]+maxRegion.value)/(2.*maxRegion.value)
        hstar=hstar[ind]/(2.*maxRegion.value)
        mstar=mstar[ind]

    if mapChoice==8:
        ind=np.where((dxdm>-maxRegion.value)&(dxdm<maxRegion.value)&
                     (dydm>-maxRegion.value)&(dzdm<maxRegion.value)&
                     (dzdm>-maxRegion.value)&(dzdm<maxRegion.value))[0]
        dxdm=(dxdm[ind]+maxRegion.value)/(2.*maxRegion.value)
        dydm=(dydm[ind]+maxRegion.value)/(2.*maxRegion.value)
        dzdm=(dzdm[ind]+maxRegion.value)/(2.*maxRegion.value)
        hdm=hdm[ind]/(2.*maxRegion.value)
        mdm=mdm[ind]

    mapRes=512
    if mapChoice==1:
        map=scatter(x=dx,y=dy,h=h,m=m,res=mapRes)
        map+=1.e-10
        map/=map.max()
        mapMin=-4
        mapMax=0
    elif mapChoice==2:
        map=scatter(x=dx,y=dy,h=h,m=m*d,res=mapRes)
        map/=map.max()
        mapMin=-6.0
        mapMax=-0.1
        mapLabel=r'$\log_{10}(EM/EM_{\rm max})$'

    elif mapChoice==3:
        mapUp=scatter(x=dx,y=dy,h=h,m=m*t*t,res=mapRes)
        mapDo=scatter(x=dx,y=dy,h=h,m=m*t,res=mapRes)
        map=mapUp/mapDo
        map/=map.max()
        mapMin=-1#7.5                                                                                                                                                                                                                                                               
        mapMax=0#8.5                                                                                                                                                                                                                                                                
        mapLabel=r'$\log_{10}(T_{y}/{\rm K})$'
    elif mapChoice==4:
        mapUp=scatter(x=dx,y=dy,h=h,m=m*t,res=mapRes)
        mapDo=scatter(x=dx,y=dy,h=h,m=m,res=mapRes)
        map=mapUp/mapDo
    elif mapChoice==5:
        mapUp=scatter(x=dx,y=dy,h=h,m=m*d*(t**0.25),res=mapRes)
        mapDo=scatter(x=dx,y=dy,h=h,m=m*d*(t**-0.75),res=mapRes)
        map=mapUp/mapDo
        mapMin=7.0
        mapMax=8.5
        mapLabel=r'$\log_{10}(T_{\rm sl}/{\rm K})$'
    elif mapChoice==6:
        mapUp1=scatter(x=dx,y=dy,h=h,m=m*t*t,res=mapRes)
        mapDo1=scatter(x=dx,y=dy,h=h,m=m*t,res=mapRes)
        mapUp2=scatter(x=dx,y=dy,h=h,m=m*d*(t**0.25),res=mapRes)
        mapDo2=scatter(x=dx,y=dy,h=h,m=m*d*(t**-0.75),res=mapRes)
        map=(mapUp1/mapDo1)/(mapUp2/mapDo2)
        mapMin=0
        mapMax=1
        mapLabel=r'$\log_{10}(T_{y}/T_{\rm sl})$'
        
    elif mapChoice==7:
        map=scatter(x=dxstar,y=dystar,h=hstar,m=mstar,res=mapRes)
        map+=1.e-10
        map/=map.max()
        mapMin=-4
        mapMax=0
        mapLabel=r'$\log_{10}(\Sigma_{\rm *}/\Sigma_{\rm *,max})$'
        
    elif mapChoice==8:
        map=scatter(x=dxdm,y=dydm,h=hdm,m=mdm,res=mapRes)
        map+=1.e-10
        map/=map.max()
        mapMin=-3
        mapMax=0
        mapLabel=r'$\log_{10}(\Sigma_{\rm DM}/\Sigma_{\rm DM,max})$'
    elif mapChoice==9:
        map=scatter(x=dx,y=dy,h=h,m=lx,res=mapRes)
        map/=map.max()
        mapMin=-6
        mapMax=0
        mapLabel=r'$\log_{10}(L_{\rm X}/L_{\rm X,max})$'
    #comptonyMap=scatter(x=dx,y=dy,h=h,m=m*t,res=mapRes)
    #comptonyMap/=np.max(comptonyMap)
    #print('this method returns a log10 map, unlog to get the pixel values')
    return np.log10(map), mapMin,mapMax, mapRes
    #image=axs.pcolormesh(np.log10(map),cmap=color_map,vmin=mapMin,vmax=mapMax)
    
    #if showComptony:
    #    axs.contour(np.log10(comptonyMap),levels=[-2,-1.5,-1,-0.5,-0.1],colors='white',linestyles='solid')

    #axs.axis('off')
    #axs.set_box_aspect(1)
    #axs.add_patch(Circle((0.5*mapRes,0.5*mapRes),0.005*mapRes,fill=True,color='black',alpha=0.5))                                                                                                                                                                                  
    #axs.add_patch(Circle((0.5*mapRes,0.5*mapRes),rChoice*mapRes/(2.*maxRegion.value),fill=False,color='yellow',linewidth=2))
    
    
