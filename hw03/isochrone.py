import numpy as np
import matplotlib.pyplot as plt
from   matplotlib import rc
import matplotlib
import seaborn as sns
import read_mist_models

matplotlib.rcParams.update({'font.size': 26})
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
sns.set_style("whitegrid")
sns.set_context("notebook",font_scale=2)

data = np.loadtxt('NGC6341.M92/final/REAL/NGC6341R.RDVIQ.cal.adj.zpt',usecols=(3,7),skiprows=1).T
R = data[0]
I = data[1]

f,ax = plt.subplots()
ax.plot((data[0]-data[1]),data[1],'.')
ax.set_xlabel(r'F606W-F814W')
ax.set_ylabel(r'F814W')
ax.set_title('NGC 6341')

## Fix the metallicity and fit for age and distance. Then set through
## metallicity files till you reach the right value. Also remember
## that these metallicities are not in 0.05 dex, and don't have any
## extinction.

#Choose the right metallicity file
iso = read_mist_models.ISOCMD('MIST_v1.0_HST_ACSWF/MIST_v1.0_feh_m0.50_afe_p0.0_vvcrit0.4_HST_ACSWF.iso.cmd')
age_idx = iso.age_index[9]
f606 = iso.isocmds[age_idx]['ACS_WFC_F606W']
f814 = iso.isocmds[age_idx]['ACS_WFC_F814W']
# Recalibrate the magnitudes to the distance of the cluster (9kpc from NED)
f814 = f814 -5 + 5*np.log10(9e3)
f606 = f606 -5 + 5*np.log10(9e3)
