import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
import matplotlib
import emcee
import corner

matplotlib.rcParams.update({'font.size': 26})
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
sns.set_style("whitegrid")
sns.set_context("notebook",font_scale=2)

def data_salpeter():
    """From Table 2 of Salpeter (1955)"""
    mass = np.array([2.23, 2.08, 1.93, 1.78, 1.63, 1.48, 1.33, 1.20, 1.09,
                     1.00, 0.93, 0.86, 0.80, 0.74, 0.68, 0.62, 0.56, 0.50])-1
    num = np.array([6.63, 7.10, 7.36, 7.52, 7.72, 8.00, 7.98, 7.98, 8.32,
                    8.50, 8.60, 8.70, 8.83, 8.97, 9.04, 9.13, 9.20, 9.22])-10
    return mass,num

def imf(theta,mass,num):
    """Data is in log-space making alpha the slope of the fit."""
    num_pred = mass*theta[1] + theta[0]
    return -np.sum((num_pred- num)**2)

def optimum(mass,num):
    """Find the optimum using scipy least squares fitter."""
    a0 = np.array([-1.,0.1])
    lsq = lambda *args: -imf(*args)
    res = least_squares(lsq,a0,args=(mass,num))
    print("alpha from scipy.optimize fit is %f"%res.x[1])
    return res.x

def mcmc(ndim,nwalkers,mass,num):
    """Use emcee to fit a line to the Salpeter IMF data in log space."""
    sampler = emcee.EnsembleSampler(nwalkers, ndim, imf, args=(mass,num))
    theta0 = [np.random.ranf(ndim) for i in range(nwalkers)]
    sampler.run_mcmc(theta0,1000)
    c,alpha = np.percentile(sampler.chain[:,:,0],50), np.percentile(sampler.chain[:,:,1],50)
    print ("alpha from mcmc optimization is %f"%alpha)
    return c,alpha
    
# Actual data from '55 paper
mass,num = data_salpeter()
xx=np.linspace(mass.min(),mass.max(),50)
yy_salp = xx*(-1.35) + np.log10(0.03)

# Scipy optimization
theta_lsq = optimum(mass,num)
yy_lsq=xx*theta_lsq[1] + theta_lsq[0]

# Emcee optimization
theta_mcmc = mcmc(2,100,mass,num)
yy_mcmc = xx*theta_mcmc[1] + theta_mcmc[0]

# Plot all results
f,ax = plt.subplots(1,1)
ax.plot(mass,num,'r.',label='Data: Salpeter(1955)')
ax.plot(xx,yy_mcmc,'m',label='Salpeter(\'55) Result',alpha=0.5)
ax.plot(xx,yy_lsq,'g',label='Scipy Optimization')
ax.plot(xx,yy_mcmc,'k--',label='Emcee Optimization')
ax.set_xlabel(r'Log(M/$M_{\odot}$)')
ax.set_ylabel(r'Log($\xi$)')
ax.legend()

plt.show()
