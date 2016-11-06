import numpy as np
import matplotlib.pyplot as plt
from   sedpy import attenuation
import fsps
from   cycler import cycler
import matplotlib

wl = np.arange(1000,10000,0.5)
wlinv = (wl*1e-4)**-1
cardelli = attenuation.cardelli(wl)
calzetti = attenuation.calzetti(wl,R_v=3.1)
powerlaw = attenuation.powerlaw(wl)
lmc      = attenuation.lmc(wl)
smc      = attenuation.smc(wl)

# f,ax = plt.subplots(1,1)
# ax.plot(wlinv,cardelli,'k-' ,label='Cardelli Law',linewidth=1)
# ax.plot(wlinv,calzetti,'k--',label='Calzetti law',linewidth=3)
# ax.plot(wlinv,powerlaw,'k:',label='Power law',linewidth=2)
# line1, = ax.plot(wlinv,lmc,'k-',label='Large MC',linewidth=2)
# line1.set_dashes([5,4,2,4])
# line2, = ax.plot(wlinv,smc,'k-', label='Small MC',linewidth=2)
# line2.set_dashes([8,4,2,4,2,4])
# ax.legend(loc=2)

# ## Overplot filters
# sdss = fsps.find_filter('sdss')
# galex = fsps.find_filter('galex')

# ax.set_prop_cycle(cycler('color',['r','g','b','y','c','m','darkorange']))
# for i in range(np.size(sdss)):
#     f = fsps.get_filter(sdss[i])
#     ax.plot((f.transmission[0]*1e-4)**-1,10*f.transmission[1],label=f.name,alpha=0.5)
#     ax.annotate(f.name,xy=((f.lambda_eff*1e-4)**-1,10*f.transmission[1].max()),textcoords='offset points',xytext=(-20,10))

    
# for i in range(np.size(galex)):
#     f = fsps.get_filter(galex[i])
#     ax.plot((f.transmission[0]*1e-4)**-1,f.transmission[1]/10,label=f.name)
#     ax.annotate(f.name,xy=((f.lambda_eff*1e-4)**-1,f.transmission[1].max()/10),textcoords='offset points',xytext=(-20,10))

# ax.set_xlabel(r'1/$\lambda$ [$\mu m^{-1}$]')

# #------------------------------------------------------------------------------------
# #                           FROM A COOL EXAMPLE ONLINE
# #------------------------------------------------------------------------------------
# # an array of parameters, each of our curves depend on a specific
# # value of parameters
# bump = np.arange(0,1.1,0.1)

# # norm is a class which, when called, can normalize data into the
# # [0.0, 1.0] interval.
# norm = matplotlib.colors.Normalize(vmin=np.min(bump),vmax=np.max(bump))

# # choose a colormap
# c_m = matplotlib.cm.winter

# # create a ScalarMappable and initialize a data structure
# s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
# s_m.set_array([])

# plt.figure(2)
# for i in bump:
#     plt.plot(wlinv,attenuation.conroy(wl,f_bump=i),color=s_m.to_rgba(i))

# plt.colorbar(s_m,label='Bump Fraction')
# plt.plot(wlinv,cardelli,'k--' ,label='Cardelli Law',linewidth=1,alpha=0.7)

# plt.legend()

## To select the UV filters plot each one to pick the ones you
## like. Picking manually.

uv = fsps.find_filter('uv')

# for fil in uv:
#     plt.figure()
#     plt.plot(wlinv,10*attenuation.conroy(wl,f_bump=0.5),'-')
#     f = fsps.get_filter(fil)
#     plt.plot((f.transmission[0]*1e-4)**-1,f.transmission[1],label=f.name)
#     plt.legend()

## Amplify all the HST (WFC3) filters:

wfc3 = fsps.find_filter('wfc3_uvis')

for fil in wfc3:
    plt.figure()
    plt.plot(wlinv,attenuation.conroy(wl,f_bump=0.5)/10,'-')
    f = fsps.get_filter(fil)
    plt.plot((f.transmission[0]*1e-4)**-1,f.transmission[1],label=f.name)
    plt.legend()

plt.plot(wlinv,attenuation.conroy(wl,f_bump=0.5),'-');
f=fsps.get_filter('wfc3_uvis_f218w');
plt.plot((f.transmission[0]*1e-4)**-1,100*f.transmission[1],label=f.name);
f=fsps.get_filter('wfc3_uvis_f225w');
plt.plot((f.transmission[0]*1e-4)**-1,100*f.transmission[1],label=f.name);
plt.legend();
