
import numpy as np
import h5py as h5

class h5_file:
    def __init__(self,filename,store_type,data_type):
        if store_type == 'yt':
            if data_type == 'mag':
                h5file=h5.File(filename, 'r')
                h5dset = h5file['data']["host_id"]
                self.VR_ID = h5dset[...]
                
                h5dset = h5file['data']["lum_bgg"]
                self.bcg_lum = h5dset[...]
                
                h5dset = h5file['data']["lum_4thbgg"]
                self.sat_lum = h5dset[...]
                
                h5dset = h5file['data']["bgg4th_VR_id"]
                self.sat_VR_ID = h5dset[...]
                
                h5dset = h5file['data']["M14"]
                self.M14 = h5dset[...]
                
                try:
                    h5dset = h5file['data']["M500c"]
                except:
                    h5dset = h5file['data']["M200m"]
                self.mass = h5dset[...]
        else:
            #note for hdf5 files the -1 entries need to be manually removed
            h5file=h5.File(filename,'r')
            h5dset = h5file["VR_ID"]
            VR_ID = h5dset[...]
            h5dset = h5file["bgg_sm"]
            bcg_stellar_mass = h5dset[...]
            h5dset = h5file["sat_VR_ID"]
            sat_VR_ID = h5dset[...]
            h5dset = h5file["sat_sm"]
            sat_stellar_mass = h5dset[...]
            h5dset = h5file["M200m"]
            halo_mass = h5dset[...]
            
            idx = np.where(VR_ID != -1)[0]
            self.bcg_stellar_mass = bcg_stellar_mass[idx]
            self.sat_VR_ID = sat_VR_ID[idx]
            self.sat_stellar_mass = sat_stellar_mass[idx]
            self.VR_ID = VR_ID[idx]
            self.M200m = halo_mass[idx]

        h5file.close()

class catalogue:
    def __init__(self,filename, apert):
        h5file=h5.File(filename,'r')
        groupName="/SWIFT/Cosmology/"
        #choose the group, in this case Cosmology is the type of cosmology framework setup                                                                                                                 
        h5group=h5file[groupName]

        #setup the attributed of the simualtion --> LCDM parameters                                                                                                                                         
        attrName="Critical density [internal units]"
        self.rhoCrit=h5group.attrs[attrName]
        attrName="H [internal units]"
        self.Hz=h5group.attrs[attrName]
        attrName="Omega_lambda"
        self.OmegaLambda=h5group.attrs[attrName]
        attrName="H0 [internal units]"
        self.H0=h5group.attrs[attrName]
        self.Ez=self.Hz/self.H0
        attrName="Omega_b"
        self.OmegaBaryon=h5group.attrs[attrName]
        attrName="Omega_m"
        self.OmegaMatter=h5group.attrs[attrName]
        attrName="Redshift"
        self.redshift=h5group.attrs[attrName]
        attrName="h"
        self.h=h5group.attrs[attrName]

        #open the rest of the grouped simulations
        h5file = h5.File(filename, 'r')
        
        h5dset = h5file['/SO/500_crit/TotalMass']
        self.M500c = h5dset[...]
        
        h5dset = h5file['/SO/200_mean/TotalMass']
        self.M200m = h5dset[...] 
        
        h5dset = h5file['VR/ID']
        self.VR_ID = h5dset[...]
        
        h5dset = h5file['/SO/500_crit/SORadius']
        self.R500c = h5dset[...] #Mpc                                                                                                                                                                      
        h5dset = h5file['/SO/200_mean/SORadius']
        self.R200m = h5dset[...] #Mpc
        
        h5dset = h5file['InclusiveSphere/'+ apert +'kpc/StellarMass']
        self.stellar_mass = h5dset[...]
        
        h5dset = h5file['InclusiveSphere/' + apert + 'kpc/StellarLuminosity']
        self.subhalo_luminosities = h5dset[...]
        
        #all subhalos have the same host id, halos that are hosts/empty = -1                                                                                                                                    
        h5dset = h5file['VR/HostHaloID']
        self.host_ID = h5dset[...]
        
        h5dset = h5file['InclusiveSphere/'+ apert +'kpc/TotalMass']
        self.aperture_mass = h5dset[...]
        
        h5dset = h5file['/VR/CentreOfPotential']
        self.CoP = h5dset[...]
        
        h5dset = h5file['/SO/200_mean/CentreOfMass']
        self.CoM_200m = h5dset[...]
        
        h5dset = h5file['/FOFSubhaloProperties/TotalMass']
        self.fof_subhalo_mass = h5dset[...]
        
        h5dset = h5file['InclusiveSphere/'+ apert +'kpc/LuminosityWeightedMeanStellarAge']
        self.LW_st_age = h5dset[...] 
        
        h5dset = h5file['InclusiveSphere/'+ apert +'kpc/StellarMassFractionInMetals']
        self.SMF_metals = h5dset[...]

        h5dset = h5file['InclusiveSphere/'+ apert +'kpc/MassWeightedMeanStellarAge']
        self.MW_st_age = h5dset[...]
        #subhalo properties
        h5dset = h5file['/BoundSubhaloProperties/TotalMass']
        self.bound_subhalo_mass = h5dset[...]
        h5file.close()
