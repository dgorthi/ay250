import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import fsps
import emcee
import corner

## SFR vs metallicity fit

#Define constants
mab0 = -2.5*np.log10(3631e-23) # Zero pt
C = 4.56785766e-19             # B band lumin to mass
C_fuv = 43.35                  # FUV lumin to SFR

def metallicity(bmag,dist,N):
    M = bmag+5-5*np.log10(dist*1e6)
    Merr = np.random.normal(loc=0,scale=0.1,size=np.size(dist))
    Mb = M+Merr

    # Z = np.random.normal(loc=6.27,scale=0.21,size=np.size(dist))
    # +(np.random.normal(loc=-0.11,scale=0.01,size=np.size(dist)))*Mb
    Z = 6.27 -0.11*Mb
    Zerr = np.random.normal(loc=0,scale=0.15,size=np.size(dist))

    return Z,Zerr

def sfr(fuv,fuv_err,dist,N):
    mfuv = fuv+5-5*np.log10(dist*1e6)
    flux = 10**-((mfuv+mab0)/2.5)
    flux_err = np.log(10)*flux*fuv_err/2.5
    Lfuv = flux*4*np.pi*(10*3.086e18)**2
    Lfuv_err = flux_err*4*np.pi*(10*3.086e18)**2
    fil = fsps.get_filter('galex_fuv')
    nu_eff = 2.9979e18/fil.lambda_eff
    sfr = nu_eff*Lfuv/(10**C_fuv)
    sfr_err = nu_eff*Lfuv_err/(10**C_fuv)
    logsfr_err = sfr_err/(np.log(10)*sfr)
    return np.log10(sfr),logsfr_err

def lnP(theta,met,met_err,sfr,sfr_err):
    chi2 = 0.5*(sfr-theta[1]-theta[0]*met)**2/(sfr_err**2 + (theta[0]*met_err)**2)
    return -np.sum(chi2)


dist,bmag,fuv,fuv_err = np.loadtxt('ps5.data',comments='#',usecols=(7,8,11,12)).T
mask = np.where((bmag!=99.99)&(fuv!=99.99))
N = np.size(mask)
dist, bmag, fuv, fuv_err = dist[mask], bmag[mask], fuv[mask], fuv_err[mask]

err_mask = np.where(fuv_err==99.99)
fuv_err[err_mask]= 0.5

met,merr = metallicity(bmag,dist,N)
sfr,serr = sfr(fuv,fuv_err,dist,N)

## Fit a straight line to this data with it's uncertainities.
## TODO: ACCOUNT FOR INTRINSIC SCATTER IN THE DATA

ndim,nwalkers = 2,100
theta0 = np.array([np.random.rand(ndim) for i in range(nwalkers)])
# theta0[:,1] = theta0[:,1]-8
# theta0[:,0] = theta0[:,0]

sampler = emcee.EnsembleSampler(nwalkers,ndim,lnP,args=(met,merr,sfr,serr))
sampler.run_mcmc(theta0,50)

## Diagnostic plots
f1,(ax1,ax2)=plt.subplots(2,1)
ax1.plot(sampler.chain[:,:,0])#,'.-')
ax1.set_title('Slope--walker')
ax2.plot(sampler.chain[:,:,1])#,'.-')
ax2.set_title('Intercept--Walker')

f2,(ax3,ax4)=plt.subplots(2,1)
ax3.plot(sampler.lnprobability[:,0],'.-')
ax3.set_title('Slope- probability')
ax4.plot(sampler.lnprobability[:,1],'.-')
ax4.set_title('Intercept- probability')

## Corner plot
corner.corner(sampler.flatchain,labels=('Slope','Intercept'),show_titles=True)

slope = np.percentile(sampler.flatchain[:,0],50)
intercept = np.percentile(sampler.flatchain[:,1],50)

fakemet = np.linspace(6,9.5,100)
fakesfr = slope*fakemet + intercept

fig,ax = plt.subplots(1,1)
ax.errorbar(met,sfr,xerr=merr,yerr=serr,fmt='o')
ax.set_xlabel(r'Log [Metallicity]')
ax.set_ylabel(r'Log [SFR]')
#ax.plot(fakemet,fakesfr,'r')
