#plot
from astropy.io import fits
from astropy.stats import sigma_clip
import os,sys,pdb,h5py
import wobble
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import Image
from scipy.signal import find_peaks_cwt


direcr = '/home/tris/Documents/URSS/URSS_Code/Processed_data/rvs_9.hdf5'
direcd = '/home/tris/Documents/URSS/URSS_Code/Processed_data/data_rvs_9.hdf5'
direcw = '/home/tris/Documents/URSS/URSS_Code/Processed_data/9_plot.png'
results = wobble.Results(filename=direcr)
data = wobble.Data(direcd)
#orders [0 to 10] = [19,20,21,22,23,24,25,26,27,28,29]
#print([np.e**i[0][0] for i in data.xs])
#results.plot_spectrum(6,23,data,direcw)
lines = []
txt = '/home/tris/Documents/URSS/URSS_Code/Processed_data/rvs_9.txt'
with open(txt, 'r') as file:
    for line in file:
        lines.append(line[:-1])
jd = [float(j[0]) for j in [i.split(' ') for i in lines[4:]]]
errors = [float(j[2]) for j in [i.split(' ') for i in lines[4:]]]
rv = [float(j[1]) for j in [i.split(' ') for i in lines[4:]]]

line_wl,search_region,plot,lines,close_search=6563,5,False,False,False
xlim=None
ylim=[0., 1.3]
ylim_resids=[-0.1,0.1]
#print(rv)
eqws = []
n,r=1,6
fig, (ax, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[4, 1]}, figsize=(12,5))
xs = np.exp(data.xs[r][n])
ax.scatter(xs, np.exp(data.ys[r][n]), marker=".", alpha=0.5, c='k', label='data', s=40)
mask = data.ivars[r][n] <= 1.e-8
#ax.scatter(xs[mask], np.exp(data.ys[r][n][mask]), marker=".", alpha=1., c='white', s=20)
#blue = []
#for c in results.component_names:
    #ax.plot(xs, np.exp(getattr(results, "{0}_ys_predicted".format(c))[r][n]), alpha=0.8)
    #blue = blue + np.exp(getattr(results, "{0}_ys_predicted".format(c))[r][n])
c = results.component_names[0]
ax.plot(xs, np.exp(getattr(results, "{0}_ys_predicted".format(c))[r][n]), alpha=0.8)
ax2.scatter(xs, np.exp(data.ys[r][n]) - np.exp(results.ys_predicted[r][n]), 
            marker=".", alpha=0.5, c='k', label='data', s=40)
ax2.scatter(xs[mask], np.exp(data.ys[r][n][mask]) - np.exp(results.ys_predicted[r][n][mask]), 
            marker=".", alpha=1., c='white', s=20)
ax.set_ylim(ylim)
ax2.set_ylim(ylim_resids)
ax.set_xticklabels([])
fig.tight_layout()
fig.subplots_adjust(hspace=0.05)
plt.show()

xs2=np.exp(data.xs)
spec2 = np.exp(data.ys)
line = np.exp(getattr(results, "{0}_ys_predicted".format(c)))
print(np.shape(np.array(line)),np.shape(xs2),np.shape(spec2))
print(line)
wave_length = fits.PrimaryHDU(xs2)
black_dots = fits.PrimaryHDU(spec2)
blue_line = fits.PrimaryHDU(line)
wave_length.writeto('wave_length.fits')
black_dots.writeto('black_dots.fits')
blue_line.writeto('blue_line.fits')
# show = True
# depth =[]
# for n in data.epochs:
#     xlim=None
#     ylim=[0., 1.3]
#     ylim_resids=[-0.1,0.1]
#     r=6
#     ub = 6580
#     lb = 6550

#     mpl.rcParams['figure.figsize'] = (8.0, 4.0)
#     xs = np.exp(data.xs[r][n])
#     ys = np.exp(data.ys[r][n])
#     spec = [i for n,i in enumerate(ys) if lb<xs[n]<ub]
#     wave = [i for i in xs if lb<i<ub]
#     continuum = np.median(ys)
#     #print(continuum)
#     equiv_width = np.trapz([(1-(i/continuum)) for i in spec],wave)
#     depth.append(np.min(spec))
#     eqws.append(equiv_width)
#     if show:
#         plt.scatter(wave, spec, marker=".", c='k', label='data', s=5)
#         plt.ylim(ylim)
#         plt.show()
#         show = False
# plt.errorbar(depth,rv,errors,fmt='o',ms=5,elinewidth=1)
# plt.show()