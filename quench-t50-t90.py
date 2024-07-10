'''
Given MB and no MB dict files, 
will iterate through each galaxy dictionary and plot different 
galaxy attributes as scatterplots/histograms. 

t95, t50, t95-t50 scatterplots

python quench-t50-t90.py -- automatically detects z1, z2, z3 dictionary
'''
import numpy as np
from matplotlib import pyplot as plt, ticker as ticker; plt.interactive(True)
from matplotlib.ticker import FormatStrFormatter
import sys
import os
import pandas as pd
import glob

import matplotlib.colors as colors
from matplotlib import cm

class Output:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
        
fig, ax = plt.subplots(3,3,figsize=(12,15))

# assign directory #DONT DO IN SHERLOCK - NO SEABORN
if len(sys.argv) > 0:
    mb_directory = str(sys.argv[1]) # example: '/Users/michpark/JWST_Programs/mockgalaxies/final/z3mb/'
    nomb_directory = str(sys.argv[2]) # example: '/Users/michpark/JWST_Programs/mockgalaxies/final/z3nomb/'
  
plotdir = '/Users/michpark/JWST_Programs/mockgalaxies/scatterplots-mb-nomb/'


zcounter_array = [0,1,2]
for zcounter in zcounter_array:
    directory_array = ['/Users/michpark/JWST_Programs/mockgalaxies/final-dicts/z'+str(zcounter+1)+'mb', '/Users/michpark/JWST_Programs/mockgalaxies/final-dicts/z'+str(zcounter+1)+'nomb/']

    for directory_index, directory in enumerate(directory_array):
        print("Current directory: " + str(directory)) # prints out directory we're currently iterating over
        #clearing variables
        dust2_array = []
        zred_array = []
        x_i = []
        x_o = []
    
        first_iteration = True # sets up this boolean for labels
    
        # Iterate through mcmc files in the directory
        for mcmcfile in os.listdir(directory): # NOW DICTIONARY!!!!!
                mcmcfile = os.path.join(directory, mcmcfile)
                #print('Making plots for '+str(mcmcfile))

                res = np.load(mcmcfile, allow_pickle=True)['res'][()]

                print('----- Object ID: ' + str(res.objname) + ' -----')

                divnorm = colors.TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.2)
                if(directory_index == 0): # MB
                    zred_dif = res.percentiles['zred'][1]-res.spsdict['zred']             
                    ax[1, zcounter].scatter(res.input_t50, res.output_t50, c=zred_dif, ec='k', norm=divnorm, cmap='bwr')
                    
                
                if(directory_index == 1): # No MB
                    ax[1, zcounter].scatter(res.input_t50, res.output_t50, c=zred_dif, ec='k', norm=divnorm, cmap='bwr')
                    
                    
                ax[1,zcounter].axline((0, 0), slope=1., ls='--', color='black', lw=2)
                ax[1,zcounter].set_ylabel(r'Recovered $t50$ [Gyr]')
                ax[1,zcounter].set_xlabel(r'Input $t50$ [Gyr]')
                first_iteration= False
                mcmc_counter += 1
            


# input vs. recovered t95
ax[0,1].scatter(results_mb[:,2], results_mb[:,3], c=zred_array[0], ec='k', norm=divnorm, cmap='bwr')
ax[0,1].axline((0, 0), slope=1., ls='--', color='black', lw=2)
ax[0,1].set_ylabel(r'Recovered $t95$ [Gyr]')
ax[0,1].set_xlabel(r'Input $t95$ [Gyr]')

# input vs. recovered t95-t50
ax[0,2].scatter(results_mb[:,2]-results_mb[:,0], results_mb[:,3]-results_mb[:,1], c=zred_array[0], ec='k', norm=divnorm, cmap='bwr')
ax[0,2].axline((0, 0), slope=1., ls='--', color='black', lw=2)
ax[0,2].set_ylabel(r'Recovered $t95 - t50$ [Gyr]')
ax[0,2].set_xlabel(r'Input $t95 - t50$ [Gyr]')

# BROAD+ONLY
# input vs. recovered t50
ax[1,0].scatter(results_nomb[:,0], results_nomb[:,1], c=zred_array[1], ec='k', norm=divnorm, cmap='bwr')
ax[1,0].axline((0, 0), slope=1., ls='--', color='black', lw=2)
ax[1,0].set_ylabel(r'Recovered $t50$ [Gyr]')
ax[1,0].set_xlabel(r'Input $t50$ [Gyr]')

# input vs. recovered t95
ax[1,1].scatter(results_nomb[:,2], results_nomb[:,3], c=zred_array[1], ec='k', norm=divnorm, cmap='bwr')
ax[1,1].axline((0,0), slope=1., ls='--', color='black', lw=2)
ax[1,1].set_ylabel(r'Recovered $t95$ [Gyr]')
ax[1,1].set_xlabel(r'Input $t95$ [Gyr]')
#ax[1,1].axline((0, percentiles['tlast_fraction'][1]*cosmo.age(obs['zred']).value), slope=0., ls='--', color='black', lw=2)

# input vs. recovered t95-t50
ax[1,2].scatter(results_nomb[:,2]-results_nomb[:,0], results_mb[:,3]-results_mb[:,1], c=zred_array[1], ec='k', norm=divnorm, cmap='bwr')
ax[1,2].axline((0, 0), slope=1., ls='--', color='black', lw=2)
ax[1,2].set_ylabel(r'Recovered $t95 - t50$ [Gyr]')
ax[1,2].set_xlabel(r'Input $t95 - t50$ [Gyr]')

plt.tight_layout()

plt.figtext(0.5,0.92, "Broad+MB", ha="center", va="top", fontsize=14, color="maroon")
plt.figtext(0.5,0.05, "Broad only", ha="center", va="top", fontsize=14, color="navy")
plt.subplots_adjust(top=0.87, bottom = 0.13)

fig.subplots_adjust(right=0.87)
cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
fig.colorbar(mappable=cm.ScalarMappable(norm=divnorm, cmap='bwr'), cax=cbar_ax, label="Difference in redshift", orientation="vertical") 
cbar_ax.set_yscale("linear")


plt.show()

# make sure plot directory exists
if not os.path.exists(plotdir):
    os.mkdir(plotdir)

counter=0
filename = '{}_z1_quenchtest.pdf' #defines filename for all objects
while os.path.isfile(plotdir+filename.format(counter)):
    counter += 1
filename = filename.format(counter) #iterate until a unique file is made
fig.savefig(plotdir+filename, bbox_inches='tight')
  
print('saved quench tests to '+plotdir+filename) 

#plt.close(fig)


        

        



             
