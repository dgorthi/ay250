import read_mist_models
import numpy as np
import glob

isochrones = {}
i = 0

for files in np.sort(glob.glob('MIST_iso_*')):
    z = 2 + i*0.05
    i+=1
    isochrones[z] = {}
    iso = read_mist_models.ISOCMD(files)

    for age in iso.ages:
        isochrones[z][age] = np.vstack((iso.isocmds[iso.age_index(age)]['ACS_WFC_F606W'], iso.isocmds[iso.age_index(age)]['ACS_WFC_F814W']))

