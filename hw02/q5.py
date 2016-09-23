import numpy as np
import matplotlib.pyplot as plt
from   scipy import integrate
import seaborn as sns
import matplotlib
from matplotlib import rc
import fsps

matplotlib.rcParams.update({'font.size': 26})
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
sns.set_style("whitegrid")
sns.set_context("notebook",font_scale=2)

# SSP of a cloud with constant metallicity (Z_sun) and no
# dust. Get spectra at 10Myr between 1500-10,000\AA.
print ("Computing the SSP of the three IMFs..")
ssp_08 = fsps.StellarPopulation(sfh=0,zmet=1,imf_type=2,imf3=0.8)
ssp_13 = fsps.StellarPopulation(sfh=0,zmet=1,imf_type=2,imf3=1.3)
ssp_18 = fsps.StellarPopulation(sfh=0,zmet=1,imf_type=2,imf3=1.8)

print("Obtaining spectra..")
w,s08 = ssp_08.get_spectrum(tage=0.01,peraa=1)
w,s13 = ssp_13.get_spectrum(tage=0.01,peraa=1)
w,s18 = ssp_18.get_spectrum(tage=0.01,peraa=1)

args = np.where((w>1500)&(w<10000))
w,s08,s13,s18 = w[args],s08[args],s13[args],s18[args]

f,ax = plt.subplots(1,1)
ax.semilogy(w,s08,'g',label=r'$\alpha=0.8$')
ax.semilogy(w,s13,'r-.',label=r'$\alpha=1.3$')
ax.semilogy(w,s18,'b--',label=r'$\alpha=1.8$')
ax.set_xlabel('Wavelength [$\AA$]')
ax.set_ylabel('Luminosity [$L_{\odot}$]')
ax.legend()

# SSP of a 10Gyr population with constant metallicity and no dust
# between 5,000-20,000\AA. Three different IMFs- Salpeter, Kroupa, van
# Dokkum.
print("Now changing the IMF..")
ssp_salpeter = fsps.StellarPopulation(sfh=0,zmet=1,dust_type=0,dust_index=0,imf_type=0)
ssp_kroupa = fsps.StellarPopulation(sfh=0,zmet=1,dust_type=0,dust_index=0,imf_type=2)
ssp_dokkum = fsps.StellarPopulation(sfh=0,zmet=1,dust_type=0,dust_index=0,imf_type=3)

print("Recomputing spectra..")
w,salp = ssp_salpeter.get_spectrum(tage=10,peraa=1)
w,krou = ssp_kroupa.get_spectrum(tage=10,peraa=1)
w,dokk = ssp_dokkum.get_spectrum(tage=10,peraa=1)

idx = np.where((w>5000)&(w<20000))
w,salp,krou,dokk = w[idx],salp[idx],krou[idx],dokk[idx]

f2,ax2 = plt.subplots(1,1)
ax2.semilogy(w,salp,'r',label=r'Salpeter',alpha=0.8)
ax2.semilogy(w,krou,'b',label=r'Kroupa',alpha=0.8)
ax2.semilogy(w,dokk,'g',label=r'van Dokkum',alpha=0.8)
ax2.set_xlabel('Wavelength [$\AA$]')
ax2.set_ylabel('Luminosity [$L_{\odot}$]')
ax2.legend()

plt.show()
