import numpy as np
import matplotlib.pyplot as plt
import fsps
from scipy import integrate
import corner

# Generate a stellar population with imf_type=2 (Kroupa 2001 IMF),
# zmet=10 (Solar metallicity for MIST Isochrones) and sfh=1-- a five
# parameter star formation history with const=1 for all the stars
# being formed in the constant star formation history period.

csp = fsps.StellarPopulation(imf_type=2,zmet=10,sfh=1,const=1)

## Get the magnitude of the population through the Galex FUV filter
## today. Convert this magnitude into a flux and then into a
## luminosity.

mag = csp.get_mags(bands=fsps.find_filter('galex_fuv'),tage=13.7)
mab0 = -2.5*np.log10(3631e-23)
fx = 10**-((mag+mab0)/2.5)
Lx = fx*4*np.pi*(10*3.086e18)**2

fil = fsps.get_filter('galex_fuv')
nu_eff = 2.9979e18/fil.lambda_eff
## The SFR appears very small but an integral over the total SFR--
## integrate.trapz(csp.sfr,10**csp.ssp_ages) ~ 1

Cfuv = nu_eff*Lx/csp.sfr

print ("The C_FUV from magnitude calculations is %.2e"%Cfuv)

# ## Instead, if you do not correct for the absolute zero point
# fx = 10**(-mag/2.5)
# Lx = fx*4*np.pi*(10*3.086e18)**2
# Cfuv = Lx/csp.sfr

# print ("The C_FUV without correcting for the absolute zero point is %.2e"%Cfuv)
# ## This is 3 orders of magnitude higher.

## --------------------------------------------------------------
##                           METHOD 2
## --------------------------------------------------------------

## The previous estimate being too different from Kennicutt's
## estimate, this time I'm going to use the luminosity computed from
## get_spectrum and manually estimate the luminosity that you obtain
## using the galex fuv filter. For this: 1. Normalize the transmission
## curve 2. Multiply (since the time domain is convolved) the
## normalized transmission curve with the spectrum and integrate over
## the frequency range of filter. 3. Convert this luminosity to C_FUV
## using the SFR.

c = 2.9979e18 #AA/s
L_sun = 3.846e33 #erg/s

fil = fsps.get_filter('galex_fuv')
fwl, ftrans = fil.transmission
ftrans_norm = ftrans/(integrate.trapz(ftrans,fwl))

wl,spec = csp.get_spectrum(peraa=True,tage=13.7)
mask = np.where((wl>fwl.min())&(wl<fwl.max()))
wl = wl[mask]; spec =spec[mask]

# Trim both the arrays to pick the nearest values- did this by eye
# like a computer inefficient idiot :-/ This trimming results in only
# 3 values being off.

fwl = fwl[5::10] #50 values long
ftrans_norm = ftrans_norm[5::10]

wl = np.delete(wl,[13,20,37])
spec = np.delete(spec,[13,20,37])

L = L_sun*integrate.trapz(wl*spec*ftrans_norm,wl)
C_FUV = L/(csp.sfr)

print ("The C_FUV from spectral luminosity is %.2e"%C_FUV)

## FUCK! This is so close to Kennicutt's value! That's alarming
## because the oft cited and used relation is for a constant star
## formation history and no dust correction at all!! Please check, for
## restoring your sanity that this value is close even with dust
## correction and other star formation histories.
