import fsps
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib import rc
#rc.set_prop_cycle(cycler('color',['red','blue','green','black', 'magenta']))

#initialize Stellar Population with Kroupa IMF and Stellar metallicity
sp = fsps.StellarPopulation(imf_type = 2, zmet = 10)
filters = ['wise_w1', 'sdss_r', 'galex_fuv']
ages = sp.ssp_ages

#create mass and magnitude dictionaries
mass = {}
mags = {}

#initialize sp with constant star formation
sp.params['sfh'] = 1
sp.params['const'] = 1.0

mags[1] = sp.get_mags(bands = filters)
mass[1] = sp.stellar_mass

#initialize sp with const sf and specified star formation start and truncation times
sp.params['sf_start'] = 0.1
sp.params['sf_trunc'] = 5.0

mags[2] = sp.get_mags(bands = filters)
mass[2] = sp.stellar_mass

sp.params['sf_start'] = 0.0
sp.params['sf_trunc'] = 0.0
sp.params['const'] = 0.5
sp.params['tburst'] = 10.01
sp.params['fburst'] = 0.5

mags[3] = sp.get_mags(bands = filters)
mass[3] = sp.stellar_mass

sp.params['const'] = 0.0
sp.params['tburst'] = 0.0
sp.params['fburst'] = 0.0
sp.params['tau'] = 1.0

mags[4] = sp.get_mags(bands = filters)
mass[4] = sp.stellar_mass

sp.params['sfh'] = 4
sp.params['tau'] = 1.0

mags[5] = sp.get_mags(bands = filters)
mass[5] = sp.stellar_mass

xmin = 5.0
xmax = 10.3
ymin = 14.0
ymax = 4.0


#fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex='col', sharey = 'row')
alph = 0.8

filter_names = ['WISE W1', 'SDSS R', 'GALEX FUV']
#from the internet
#nrow = 2; ncol = 2;
#fig, axs = plt.subplots(nrows=nrow, ncols=nrow)

#for ax in axs.reshape(-1):
#  ax.set_ylabel(str(i))

fig, ax = plt.subplots(nrows=2, ncols=2)

for i in range(3):
	for j in range(5):
		#ax[i].set_prop_cycle(cycler('color',['red','blue','green','black', 'magenta']))
		ax[i].plot(ages, mags[j][:,i], '.', alpha = alph)
		ax[i].plot(ages, mags[i][:,i], '.', alpha = alph)
		ax[i].set_ylim([ymin, ymax])
		ax[i].set_xlim([xmin,xmax])

#ax1.set_xticks(size = 5)

	ax[i].set_ylabel(filter_names[i])
	ax[i].set_xlabel('log(Age[years])')
#ax1.set_xlabel('log(Age) [years]')
#ax1.text(left,top,'WISE W1', va = 'top', transform=ax1.transAxes)
#ax1.annotate('WISE W1',xy=(xmax,ymax),xycoords='data',fontsize=14)
"""
ax2.plot(ages, mags1[:,1],'.', color = 'red', alpha = alph)
ax2.plot(ages, mags2[:,1],'.', color = 'blue', alpha = alph)
ax2.set_ylim([ymin,ymax])
ax2.set_xlim([xmin,xmax])
ax2.set_ylabel('SDSS R')
#ax2.text(left,top,'SDSS R', va = 'top', transform=ax2.transAxes)


ax3.plot(ages, mags1[:,2], '.', color = 'red', alpha = alph)
ax3.plot(ages, mags2[:,2], '.', color = 'blue', alpha = alph)
ax3.set_ylim([ymin,ymax])
ax3.set_xlim([xmin,xmax])
ax3.set_xlabel('log(Age[years])')
ax3.set_ylabel('GALEX FUV')
#ax3.text(left,top,'GALEX FUV', va = 'top', transform=ax3.transAxes)
"""
for i in range(5):
	ax[4].set_prop_cycle(cycler('color',['red','blue','green','black', 'magenta']))
	ax[4].plot(ages, mass[i], '.', alpha = alph)
	#ax4.set_ylim([ymax,ymin])
ax[4].set_xlabel('log(Age[years])')
ax[4].set_ylabel('Stellar Mass')
ax[4].set_xlim([xmin,xmax])
	#1ax4.text(left,top,'Stellar Mass', va = 'top', transform=ax4.transAxes)

plt.show()


