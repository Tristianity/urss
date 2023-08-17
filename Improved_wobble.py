from astropy.io import fits
from astropy.stats import sigma_clip
import os,sys,pdb,h5py
import wobble
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import Image
DATADIREC = 'AllData'
root=str(os.path.dirname(__file__))+'/'

def read_file(file_name,directory):
    '''
    args:
        file_name: string including file extention of file to read in
        directory: directory the file is stored at
    returns:
        spectra: original spectra of observation
        wave: wavelenght data of observation
        noise: noise data for observation
        jd: (mjd) julian date of observation
        berv: baryecntric relative velocity of observation
        airmass: airmass of observation
    '''
    fits_file = fits.open(directory+file_name)
    h=fits_file[0].header
    jd,berv,airmass=h['MJD-OBS'],h['HIERARCH CARACAL BERV'],h['AIRMASS'],
    spectra,wave,noise=fits_file[1].data,fits_file[4].data,fits_file[3].data
    fits_file.close()
    #print(np.shape(spectra))
    return spectra,wave,noise,jd,berv,airmass

# 1,3,5,7
##############################################################
#there are 8 transits (0-7)
#epochs per transit [18, 58, 24, 32, 5, 50, 7, 55]#vis
TRANSIT = 7#which of 0-7 transits to analyse vis
#!!! hardcoded on all data on line 79
VIS = True#if False will do nir instead of vis
LINE = False#draw midpoint line: wrong date for some
#orders = [2,3,4,5,6,7,14,16,23,24,25,26,27]#nir orders
#orders=  [2,3,4,5,6,24,25,26,27]#NIR SHORT
orders = [19,20,21,22,23]#vis orders 0-17 are busted: VIS SHORT
#orders = [19,20,21,22,23,24,25,26,27,28,29]#vis go to]
#orders = [25]
#orders = [19,20,21,22,23,24,25,26,27,28,29,48,49,50,51]#VIS MEDIUM
#orders = [51,50,49,48,47,43,41,38,37,33,30,29,28,27,26,25,24,23,22,21,20,19]#VIS LONG
sigma_clipper = True#4 sigma using astropy
sigma = 4
##############################################################
if VIS:
    n_orders = 61
    n_pixels = 4096
    alldata = [i for i in os.listdir(root+DATADIREC) if i[-10:]=='vis_A.fits']
else:
    n_orders = 28
    n_pixels = 4080
    alldata = [i for i in os.listdir(root+DATADIREC) if i[-10:]=='nir_A.fits']

#getting transits
transits = []
alldata.sort(key = lambda x: int(x.split('-')[1][:8]))# sort by date
sortval = int(alldata[0].split('-')[1][:8])
temptransit = []
for n,i in enumerate(alldata):
    dateval = int(i.split('-')[1][:8])
    if sortval-2<=dateval<=sortval+2:
        temptransit.append(i)
    else:
        transits.append(temptransit)
        temptransit = [i]
    sortval= dateval
transits.append(temptransit)
if not(VIS):
    transits.insert(2,None)
    transits.insert(3,None)
    assert TRANSIT not in [2,3], 'no nir data for this transit'

#midpoint of all transits for all 8 transits
transit_mids = [57589.05139000015, 57648.95290999999, 57884.12184000015, 57904.08901000023, 58003.92487999983, 58063.826400000136, 58665.06016999995, 58704.99452000018]
files = transits[TRANSIT]#!!!!!!!!HERE
n_epochs = len(files)

#reading in all date
flux=np.zeros((n_epochs,n_orders,n_pixels),dtype=np.float32)
wave = np.zeros((n_epochs,n_orders,n_pixels),dtype=np.float64)
noise = np.zeros((n_epochs,n_orders,n_pixels),dtype=np.float64)
date=np.zeros(n_epochs)
berv=np.zeros(n_epochs)
airmass = np.zeros(n_epochs)
for n,i in enumerate(files):
    flux[n,:,:],wave[n,:,:],noise[n,:,:],date[n],berv[n],airmass[n]=read_file(i,root+DATADIREC+'/')
#munging data
data = wobble.Data()
for n in range(n_epochs):
    xs = [wave[n,r,:] for r in orders]
    ys = [flux[n,r,:] for r in orders]
    ns = [noise[n,r,:] for r in orders]
    ns = np.ma.filled(np.ma.masked_array(ns,mask=np.isclose(ns,0)),np.nan)# sorting out zeros
    ivars = np.array([i**-2 for i in ns])
    if sigma_clipper:#sigma cipping using astropy
        sigma_mask = sigma_clip(ys, sigma, None, cenfunc = np.mean, copy = True).mask
        ivars = np.ma.filled(np.ma.masked_array(ivars, mask=sigma_mask),0)
    ivars = np.nan_to_num(ivars)#no sigma clipping, just sort out nans
    sp = wobble.Spectrum(xs, ys, ivars,
                        dates=date[n], bervs=berv[n]*1.e3, airms=airmass[n])
    sp.mask_low_pixels(min_flux=1.e-3, padding=0)
    sp.transform_log()
    sp.continuum_normalize()
    data.append(sp)

#calculaating RVs
results = wobble.Results(data=data)
for r in range(len(data.orders)):
    print('starting order {0} of {1}'.format(r+1, len(data.orders)))
    model = wobble.Model(data, results, r)
    model.add_star('star')
    model.add_telluric('tellurics', variable_bases=2)
    wobble.optimize_order(model)
#pdb.set_trace()
results.combine_orders('star')
results.apply_drifts('star') # instrumental drift corrections
results.apply_bervs('star') # barycentric corrections

#counter to increment file name each runthrough
with open(root+'improved_wobble_counter.txt', 'r') as file:
    c = file.read()
    file.close()
with open(root+'improved_wobble_counter.txt', 'w') as file:
    file.write(str(int(c)+1))
    file.close()

#writing results
writedirec = '/home/tris/Documents/URSS/URSS_Code/improved_wobble_results/'
file_name = 'rvs_'+str(c)
results.write_rvs('star', writedirec+file_name+'.txt')
results.write(writedirec+file_name+'.hdf5')
data.write(writedirec+'data_'+file_name+'.hdf5')
#plotting results
#results.plot_spectrum()
#results.plot_chromatic_rvs()
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['figure.figsize'] = (8.0, 4.0)
plt.errorbar(data.dates, results.star_time_rvs,#- np.mean(results.star_time_rvs) 
                results.star_time_sigmas,
                fmt='o', ms=5, elinewidth=1)
if LINE:
    plt.axvline(transit_mids[TRANSIT])
plt.xlabel('JD')
plt.ylabel(r'RV (m s$^{-1}$)')
plt.show()