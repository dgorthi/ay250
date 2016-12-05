# ## This code tests the SFH parameters- sfstart, sftrunc, tau, tburst,
# ## fburst and constant; by generating the magnitude of each CSP under
# ## three different filters- Galex-FUV, SDSS-r, WISE-1.  As a fourth
# ## comparision, the evolution of the stellar mass for each sfh is also
# ## shown.

import fsps
import matplotlib.pyplot as pl
import numpy as np
# from cycler import cycler
# from matplotlib import rc
# rc.set_prop_cycle(cycler('color',['red','blue','green','black', 'magenta']))

# Stellar Population initiliased with Kroupa IMF, solar metallicity
# without nebular emission.  They are the default parameters- the
# final program we have to integrate this into already defines this.
pop = fsps.StellarPopulation(zcontinuous=1)
default_params = dict([(k, pop.params[k]) for k in pop.params.all_params])
#libs = pop.libraries

def _reset_default_params():
        for k in pop.params.all_params:
                pop.params[k] = default_params[k]

filters = ['wise_w1', 'sdss_r', 'galex_fuv']
ages = pop.ssp_ages

#create mass and magnitude dictionaries
prop = {}

## 1. Constant SFH
_reset_default_params()
pop.params['sfh'] = 1
pop.params['const'] = 1.0
prop[1] = np.concatenate((pop.get_mags(bands = filters), (pop.stellar_mass).reshape(np.size(ages),1)),axis=1)

## 2. Constant SFH between 10Myr and 10Gyr
_reset_default_params()
pop.params['sfh'] = 1
pop.params['const'] = 1.0
pop.params['sf_start'] = 0.01
pop.params['sf_trunc'] = 10.0
prop[2] = np.concatenate((pop.get_mags(bands = filters), (pop.stellar_mass).reshape(np.size(ages),1)),axis=1)

## 3. Constant SFH + Burst at log-age 10.01
_reset_default_params()
pop.params['sfh'] = 1
pop.params['const'] = 0.5
pop.params['tburst'] = 10.01
pop.params['fburst'] = 0.5
prop[3] = np.concatenate((pop.get_mags(bands = filters), (pop.stellar_mass).reshape(np.size(ages),1)),axis=1)

## 4. Exponential SFH with tau= 1Gyr
_reset_default_params()
pop.params['sfh'] = 1
pop.params['tau'] = 1.0
prop[4] = np.concatenate((pop.get_mags(bands = filters), (pop.stellar_mass).reshape(np.size(ages),1)),axis=1)

## 5. Delayed Exponential SFH with tau= 1Gyr
_reset_default_params()
pop.params['sfh'] = 4
pop.params['tau'] = 1.0
prop[5] = np.concatenate((pop.get_mags(bands = filters), (pop.stellar_mass).reshape(np.size(ages),1)),axis=1)

## ---------------------------------------------------------------------------------
##                                   Plotting stuff
## ---------------------------------------------------------------------------------
color= ['r','m','g','cyan','b']
alph = 0.8

prop['name'] = ['WISE W1', 'SDSS R', 'GALEX FUV','MASS']

nrows=2;ncols=2
fig, ax = pl.subplots(nrows=nrows, ncols=ncols)
# for i in range(2):
#         for j in range(2):
#                 ax[i][j].set_ylim([14.0, 4.0])
#                 ax[i][j].set_xlim([5.0, 10.3])

for row in range(nrows):
        for col in range(ncols):
                ax[row][col].set_ylabel(prop['name'][(nrows*row+col)])
                ax[row][col].set_xlabel('log(Age[years])')
                for sfh in range(5):
                        ax[row][col].plot(ages, prop[sfh+1][:,(nrows*row+col)],'.',c=color[sfh])

pl.show()


