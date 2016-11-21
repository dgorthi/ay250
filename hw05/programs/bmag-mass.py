import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import fsps

# csp = fsps.StellarPopulation(imf_type=2,zmet=10,sfh=1,const=1)
# mag = csp.get_mags(bands='b')
# mass = np.zeros([np.size(csp.ssp_ages)])

# for i,age in enumerate(csp.ssp_ages):
#     mass[i] = integrate.trapz(csp.sfr[:i+1],(10**csp.ssp_ages[:i+1]))

# mass = np.log10(mass)

# # a0 = np.array([5,-1])
# # lsq = lambda x: np.sum((mass[40:]*x[1]+x[0]-mag[40:])**2)
# # res = least_squares(lsq,a0)
# # print("Slope from scipy.optimize fit is %f"%res.x[1])

# # line_fit = mass*res.x[1]+res.x[0]

# fig1,ax1 = plt.subplots(1,1)
# ax1.plot(mass,mag,'-b')
# ax1.plot(mass[40:],mag[40:],'ob')
# ax1.plot(mass,line_fit,'r')
# ax1.set_xlabel(r'$\log{M_*}$')
# ax1.set_ylabel(r'<B>')
# ax1.set_title('Constant SFR')
# ax1.invert_yaxis()


# ## Compare the B-band and Galex FUV filters. The Galex FUV filter is
# ## constant as expected, unlike the B-band filter.
# fig3,ax3 = plt.subplots(1,1)
# ax3.plot(csp.ssp_ages,csp.get_mags(bands='b'),'ob',label='B band')
# ax3.plot(csp.ssp_ages,csp.get_mags(bands=fsps.find_filter('galex_fuv')),'r',label='Galex FUV')
# ax3.set_xlabel(r'$\log{Age}$')
# ax3.set_ylabel('Magnitude')

# # ------------------------------------------------------------
# #            EXPONENTIALLY DECLINING SFR
# # ------------------------------------------------------------
# csp = fsps.StellarPopulation(imf_type=2,zmet=10,sfh=1,tau=0.1)
# mag = csp.get_mags(bands='b')
# mass = np.zeros([np.size(csp.ssp_ages)])

# for i,age in enumerate(csp.ssp_ages):
#     mass[i] = integrate.trapz(csp.sfr[:i+1],(10**csp.ssp_ages[:i+1]))
    
# fig2,ax2 = plt.subplots(1,1) 
# ax2.semilogx(mass,mag,'o-')
# ax2.set_xlabel(r'$\log{M_*}$')
# ax2.set_ylabel(r'<B>')
# ax2.set_title(r'Exponential SFR ($\tau = 0.1$)')
# ax2.invert_yaxis()


## Only the slope after 100Myr matters- before that, even with a
## constant star formation rate the B band magnitude is not influenced
## by the high mass stars. Hence, for a mass-Bmag relationship, fit
## only the part of the graph between 13Myr and 1Gyr (it peculiarly
## falls off after that).

## The B band magnitudes are actually dominated by stars with masses
## between 1-5M_sun, not high mass stars. The magnitude rises once the
## A type stars start to dominate.
## The take away from this exercise was that the B band magnitude does
## not scale with time- hence even though the mass is increasing
## constantly with time, using this method to determine a mass
## magnitude relationship would not be correct.

## Instead, you can assume the relationship is constant, and the
## constant of proportionality would just be the magnitude of the
## composite stellar population today. 

csp = fsps.StellarPopulation(imf_type=2,zmet=10,sfh=1,const=1)
mag = csp.get_mags(bands='b',tage=13.7)
mab0 = -2.5*np.log10(3631e-23)
fx = 10**-((mag+mab0)/2.5)
Lx = fx*4*np.pi*(10*3.086e18)**2

C = 1/Lx

bmag = np.linspace(-3,10,num=100)
flux = 10**-((bmag+mab0)/2.5)
Lumin = flux*4*np.pi*(10*3.086e18)**2
mass = C*Lumin

fig,ax = plt.subplots(1,1)
ax.plot(bmag,mass,'o')
ax.set_xlabel(r'B magnitude')
ax.set_ylabel(r'M_{*}')

## This is not a great proxy for a galaxy's stellar mass because the
## star formation history changes the B-band magnitude you observe
## even if the mass is the same for both galaxies.
print ('The B-band magnitude for a constant SFH is %f'%csp.get_mags(bands='b',tage=13.7))

expcsp = fsps.StellarPopulation(imf_type=2,zmet=10,sfh=1,tau=0.5)
print ('The B-band magnitude for an exponentially declining SFH with tau=0.1 is %f'%expcsp.get_mags(bands='b',tage=13.7))

burst = fsps.StellarPopulation(imf_type=2,zmet=10,sfh=1,fburst=1,tburst=3)
print ('The B-band magnitude for a burst of star formation at 3Gyr is %f'%burst.get_mags(bands='b',tage=13.7))
