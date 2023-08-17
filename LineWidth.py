# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 18:58:31 2023

@author: Trist
"""
import matplotlib.pyplot as plt
import os, sys
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy import constants as const
from scipy.signal import find_peaks_cwt

root=str(os.path.dirname(__file__))+'/'

def read_file(file_name,directory):
    fits_file = fits.open(directory+file_name)
    wave = [i for j in fits_file[4].data for i in j]#Wavelengths in Angstroms
    spec = [i for j in fits_file[1].data for i in j]#reduced Spectra in Units
    h=fits_file[0].header
    #note: a few NIR files dont have BERV, only BJD, in 2016-07-20
    #try:
    #I catch these errors in time_series()#its 2017 He observations that are unmarked
    params = {'time':h['MJD-OBS'],'BERV':h['HIERARCH CARACAL BERV'],'airmass':h['AIRMASS'],'BJD':h['HIERARCH CARACAL BJD'],'name':file_name}
    #except KeyError:
     #   params = {'time':h['MJD-OBS'],'berv':0,'airmass':h['AIRMASS'],'BJD':h['HIERARCH CARACAL BJD'],'name':file_name}#temp
    fits_file.close()
    return np.array([wave,spec]),params

def find_nearest_index(array,value,direction):# change from whatever to closeest
    #direction True for right False for Left
    if direction:#right
        i = 0
        while array[i]<value:
            i+=1
        return i
    else:#left
        i = len(array)-1
        while array[i]<value:
            i-=1
        return i

# time from name replaced with reading header
def time_from_name(s):
     #t0='car-20160720T02h36m02s-sci-amap'
     return 'T'.join(['-'.join([s[4:8],s[8:10],s[10:12]]),':'.join([s[13:15],s[16:18],s[19:21]])])

def time_series(data,airmass=False):
    He_peaks,H_peaks,He_wls,H_wls,H_time,He_time,H_am,He_am=[],[],[],[],[],[],[],[]
    for n,item in enumerate(data):
        try:
            if item[-10:]=='vis_A.fits':
                spectrum,params=read_file(item, root+'AllData/')
                peak_width,peak_wl=eq_width(spectrum,params, *loi['H-alpha'])
                H_peaks.append(peak_width),H_wls.append(peak_wl),H_time.append(params['time']),H_am.append(params['airmass'])
                print(n)
            if item[-10:]=='nir_A.fits':
                spectrum,params=read_file(item, root+'AllData/')
                peak_width,peak_wl=eq_width(spectrum,params, *loi['He10833'])
                He_peaks.append(peak_width),He_wls.append(peak_wl),He_time.append(params['time']),He_am.append(params['airmass'])
        except KeyError:#ignore data points without BERV recorded, ill change this later
            print(item)
    plt.scatter(H_time,H_peaks,marker = '.',s=5)
    plt.scatter(He_time,He_peaks,marker = '.',s=5)
    plt.xlabel('time (jd)')
    plt.ylabel('equivilant widths')
    plt.legend(['H-alpha','He10833'])
    plt.show()
    if airmass:
        plt.scatter(H_am,H_peaks,marker = '.',s=5)
        plt.show()
        plt.scatter(H_am,H_time,marker = '.',s=5)
        plt.show()

def normalise_spec(spec,clipped_spec):#normailises clipped spectrum
    #all/(start+end/2)
    norm=(np.median(clipped_spec[:5])+np.median(clipped_spec[-5:]))/2
    return [i/norm for i in spec]

def correction(wave,params):#corrects for wavelenght shift
    #berv negative is moving away
    #o/(v/c+1) = r  for v moving away observed is longer o>r (-v menas o<r, +v= o>r)
    c=const.c.to('km/s').value
    v=-1*params['BERV']
    return [i/((v/c)+1) for i in wave]

def eq_width(spectrum,params,line_wl,search_region,plot=True,lines=True,close_search=False):
    '''
    Parameters
    ----------
    spectrum : 2d numpy array
        in the form[[wavelenght],[reduced_spectra]]
    line_wl : float
        target line wavelenght in Å
    search_region : float
        region over which data is fitted in Å
    plot : Bool, optional
        Will a plot be produced. The default is True.
    lines : Bool, optional
        Will there be fitting construction lines. The default is True.
    close_search : Bool, optional
        True: fitts to deapest peak. False: fitts to closest peak to line_wl. The default is False.

    Returns
    -------
    equiv_width : float
        equivilant wavelenght of fitted peak
    peak_pos : float
        wavlenght of peak in Å
    '''
    spectrum = np.swapaxes(spectrum, 0, 1)
    data_pairs = [i for i in spectrum if line_wl-search_region<i[0]<line_wl+search_region]
    data = np.swapaxes(data_pairs,0, 1)
    wave,spec=data[0],data[1]


    #temp eqwidth clipping. #VERY TEMPORARY NEEDED FOR NORMALISATION YIKES CODING HERE
    clipped_spec,clipped_wave=[],[]
    for n,i in enumerate(spec):
        if (line_wl == 6565) and (6562.5<wave[n]<6565.6):
            clipped_spec.append(i),clipped_wave.append(wave[n])
        if (line_wl == 10833) and(10831.5<wave[n]<10834.2):
            clipped_spec.append(i),clipped_wave.append(wave[n])
    #noramise spec
    spec=normalise_spec(spec,clipped_spec)
    #berv correction
    wave=correction(wave,params)

    continuum = np.median(spec)
    peaks=find_peaks_cwt([-i for i in spec],2)
    if close_search:#picks the peak candidate closest to the line_wl
        peak_index = [i for i in peaks if abs(wave[i]-line_wl)==min([abs(wave[j]-line_wl) for j in peaks])][0]
    else:#picks the lowest peak
        peak_index= [i for i in peaks if spec[i]==min([spec[j] for j in peaks])][0]
    peak_height,peak_pos=spec[peak_index],wave[peak_index]
    gap = continuum-peak_height#used scaling the graph axis

    #fwhh calculatin
    midpoint = continuum-0.5*gap
    LH=wave[find_nearest_index(spec[:peak_index],midpoint,False)]
    RH=wave[find_nearest_index(spec[peak_index:],midpoint,True)+peak_index]
    # equiv_width = (RH-LH)*gap #old eqw

    #temp eqwidth clipping.  MOVED UP TEMPORARILY
    clipped_spec,clipped_wave=[],[]
    for n,i in enumerate(spec):
        if (line_wl == 6565) and (6562.5<wave[n]<6565.6):
            clipped_spec.append(i),clipped_wave.append(wave[n])
        if (line_wl == 10833) and(10831.5<wave[n]<10834.2):
            clipped_spec.append(i),clipped_wave.append(wave[n])
    equiv_width = np.trapz([(1-(i/continuum)) for i in clipped_spec],clipped_wave)
    if equiv_width>2 or equiv_width<0:
        print(params)
    #plotting
    if plot:
        plt.figure()
        plt.scatter(wave,spec, marker = '.',s=5)
        plt.xlim([line_wl-search_region*1.2, line_wl+search_region*1.2])
        plt.ylim([peak_height-gap*0.1, continuum+gap*0.2])
        if lines:
            plt.axhline(continuum,color='r',lw=0.5)
            plt.axhline(peak_height,color='r',lw=0.5)
            plt.axhline(midpoint,color='r',lw=0.5)
            plt.axvline(peak_pos,color='r',lw=0.5)
            plt.axvline(LH,color='r',lw=0.5)
            plt.axvline(RH,color='r',lw=0.5)
        [plt.axvline(wave[i],color='b',lw=0.25) for i in peaks]
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Flux')
        plt.title('equivilant width = '+str(str(equiv_width)[:5]))
        plt.show()
    return equiv_width,peak_pos

#########################
#creating example graphs
spectrum,params = read_file('old_data/StartA.fits', root)
#spectrum,params=read_file('AllData/car-20170512T03h34m57s-sci-alof-vis_A.fits',root)
#spectrum,params=read_file('AllData/car-20170512T03h38m49s-sci-alof-vis_A.fits',root)
eq_width(spectrum,params, 6565, 3)#H-alpha
print('H')
spectrum,params = read_file('old_data/StartNirA.fits', root)
eq_width(spectrum,params, 10833, 8,close_search=True)#He10833
print('He')
#sys.exit()
loi = {'H-alpha':[6565,5,False,False,False],'He10833':[10833,10,False,False,True]}

##########################
#creating single window graph
data_2019 = os.listdir(root+'Latesttransit/red')
time_series(data_2019,airmass=True)
##########################
#creating full time series
all_data = os.listdir(root+'AllData')#[first obs,...,lastobs]
all_data.remove('car-20170512T03h34m57s-sci-alof-vis_A.fits')#problem children
all_data.remove('car-20170512T03h38m49s-sci-alof-vis_A.fits')
all_data.remove('car-20160720T03h28m20s-sci-amap-nir_A.fits')
#new problem children
all_data.remove('car-20170512T03h42m41s-sci-alof-vis_A.fits')
all_data.remove('car-20170512T03h46m33s-sci-alof-vis_A.fits')
all_data.remove('car-20170907T21h21m15s-sci-cabJ-vis_A.fits')
all_data.remove('car-20160720T02h46m00s-sci-amap-nir_A.fits')
#different single window
times=Time([time_from_name(i) for i in all_data],format='fits',scale='utc')
times = times.to_value('mjd','long')
#print([float(i) for i in times])
#full time series
time_series(all_data)

# #Ca a,b and c
# plt.figure() 
# plt.scatter(wave,spec, marker = '.',s=5)
# plt.xlim([8490, 8670])
# plt.ylim([0, 0.2])


#{'time': 57885.1519560185, 'BERV': 21.467913, 'airmass': 1.0745, 'BJD': 57885.6540451, 'name': 'car-20170512T03h38m49s-sci-alof-vis_A.fits'}
