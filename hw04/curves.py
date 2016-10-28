import numpy as np
import matplotlib.pyplot as plt
from   sedpy import attenuation

wl = np.arange(1000,10000,0.5)
wlinv = (wl*1e-4)**-1
cardelli = attenuation.cardelli(wl)
calzetti = attenuation.calzetti(wl,R_v=3.1)
powerlaw = attenuation.powerlaw(wl)
lmc      = attenuation.lmc(wl)
smc      = attenuation.smc(wl)


f,ax = plt.subplots(1,1)
ax.plot(wlinv,cardelli,'-',label='Cardelli Law')
ax.plot(wlinv,calzetti,'-',label='Calzetti law')
ax.plot(wlinv,powerlaw,'-',label='Power law')
ax.plot(wlinv,  lmc,   '-',label='Large MC')
ax.plot(wlinv,  smc,   '-',label='Small MC')

ax.set_xlabel(r'1/$\lambda$ [$\mu m^{-1}$]')
ax.legend(loc=2)

plt.show()
