import h5py as h5
# Extract key properties from SOAP catalogues for groups and clusters.                                                                                                                                                                       
# This is to manage file sizes when processing objects 
# includes Xray maps of the halos, minimum halo mass associated with comp. test

# Group and cluster selection parameters                                                                                                                                                                                                     
delta="200_mean" # density contrast [200_mean, 500_crit etc]                                                                                                                                                                                
minHaloMass=1.e14    # minimum mass in Msun units      

flamingoSnap=77
flamingoRun="L1000N1800"
flamingoModel="HYDRO_FIDUCIAL"

# File details                                                                                                                                                                                                                                
projectPath="/cosma8/data/dp004/flamingo/Runs"
dataPath=projectPath+"/"+flamingoRun+"/"+flamingoModel
snapFile=dataPath+"/snapshots/flamingo_{:04d}/flamingo_{:04d}.hdf5".format(flamingoSnap,flamingoSnap)
soapFile=dataPath+"/SOAP/halo_properties_{:04d}.hdf5".format(flamingoSnap)
outputPath="/cosma8/data/dp004/dc-corr2/magnitude_gap/saved_data/L1000N/1800"
outputFile=outputPath+'RC_z0_' + delta  + '_.hdf5'

# Read in basic halo properties                                                                                                                                                                                                              
h5file=h5.File(soapFile,'r')

groupName="/SWIFT/Cosmology/"

print("Reading cosmological parameters")
h5group=h5file[groupName]
attrName="Critical density [internal units]"
rhoCrit=h5group.attrs[attrName]
attrName="H [internal units]"
Hz=h5group.attrs[attrName]
attrName="H0 [internal units]"
H0=h5group.attrs[attrName]
Ez=Hz/H0

attrName="Omega_b"
OmegaBaryon=h5group.attrs[attrName]
attrName="Omega_m"
OmegaMatter=h5group.attrs[attrName]
attrName="Redshift"
redshift=h5group.attrs[attrName]
attrName="h"
hubbleParameter=h5group.attrs[attrName]
print("rho_cr, E(z), Omega_b, Omega_m, z, h:",rhoCrit,Ez,OmegaBaryon,OmegaMatter,redshift,hubbleParameter)

groupName="/SO/"+delta+"/"
datasetName=groupName+"TotalMass"

print("Reading:",datasetName)
h5dset=h5file[datasetName]
MDelta=h5dset[...]

datasetName=groupName+"HotGasMass"
print("Reading:",datasetName)
h5dset=h5file[datasetName]
MhotDelta=h5dset[...]

# Extract group and cluster sample                                                                                                                                                                                                            
selection=(MDelta>minHaloMass)
MDelta=MDelta[selection]
MhotDelta=MhotDelta[selection]
numHaloes=MDelta.size
print("Number of selected haloes:",numHaloes)

datasetName=groupName+"SORadius"
print("Reading:",datasetName)
h5dset=h5file[datasetName]
dummy=h5dset[...]
RDelta=dummy[selection]

groupName="/VR/"

datasetName=groupName+"CentreOfPotential"
print("Reading:",datasetName)
h5dset=h5file[datasetName]
dummy=h5dset[...]
CoP=dummy[selection,:]

datasetName=groupName+"ID"
print("Reading:",datasetName)
h5dset=h5file[datasetName]
dummy=h5dset[...]
VR_ID=dummy[selection]

datasetName=groupName+'HostHaloID'
h5dset = h5file[datasetName]
dummy = h5dset[...]
host_ID = dummy[selection]

#get basic subhalos data
groupName="/FOFSubhaloProperties/"
datasetName=groupName+'TotalMass'
h5dset=h5file[datasetName]
dummy = h5dset[...]
subhalo_mass = dummy[selection]

h5file.close()

# Write out data                                                                                                                                                                                                                              
print("Writing to:",outputFile)
h5file=h5.File(outputFile,'w')

h5group=h5file.create_group("Cosmology")
attrName="Critical density [internal units]"
h5group.attrs[attrName]=rhoCrit
attrName="H [internal units]"
h5group.attrs[attrName]=Hz
attrName="H0 [internal units]"
h5group.attrs[attrName]=H0
attrName="Omega_b"
h5group.attrs[attrName]=OmegaBaryon
attrName="Omega_m"
h5group.attrs[attrName]=OmegaMatter
attrName="Redshift"
h5group.attrs[attrName]=redshift
attrName="h"
h5group.attrs[attrName]=hubbleParameter

h5dset=h5file.create_dataset("VR_ID",VR_ID.shape,dtype='u8',compression="gzip")
h5dset[:]=VR_ID

h5dset=h5file.create_dataset("host_ID",host_ID.shape,dtype='u8',compression="gzip")
h5dset[:]=host_ID

h5dset=h5file.create_dataset("CentreOfPotential",CoP.shape,dtype='f',compression="gzip")
h5dset[:,:]=CoP

h5dset=h5file.create_dataset("TotalMass",MDelta.shape,dtype='f',compression="gzip")
h5dset[:]=MDelta

h5dset=h5file.create_dataset("HotGasMass",MhotDelta.shape,dtype='f',compression="gzip")
h5dset[:]=MhotDelta

h5dset=h5file.create_dataset("R200m",RDelta.shape,dtype='f',compression="gzip")
h5dset[:]=RDelta

h5dset=h5file.create_dataset("FOFSubhaloMass",subhalo_mass.shape,dtype='f',compression="gzip")
h5dset[:]=subhalo_mass
h5file.close()

print("----------------------------------------------------------------")
print("----------------------------------------------------------------")
print()

