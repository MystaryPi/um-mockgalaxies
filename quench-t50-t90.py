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
        
fig, ax = plt.subplots(3,1,figsize=(6, 8))

# assign directory #DONT DO IN SHERLOCK - NO SEABORN
'''
if len(sys.argv) > 1:
    mb_directory = str(sys.argv[1]) # example: '/Users/michpark/JWST_Programs/mockgalaxies/final/z3mb/'
    nomb_directory = str(sys.argv[2]) # example: '/Users/michpark/JWST_Programs/mockgalaxies/final/z3nomb/'
'''

plotdir = '/Users/michpark/JWST_Programs/mockgalaxies/scatterplots-mb-nomb/'

zcounter_array = [0,1,2]
colors = ['#7FCDBB', '#1D91C0','#0C2C84']

for zcounter in zcounter_array:
    #directory_array = ['/Users/michpark/JWST_Programs/mockgalaxies/final-dicts/z'+str(zcounter+1)+'mb', '/Users/michpark/JWST_Programs/mockgalaxies/final-dicts/z'+str(zcounter+1)+'nomb/']
    directory_array = ['/Users/michpark/JWST_Programs/mockgalaxies/final-dicts/z'+str(zcounter+1)+'mb']

    for directory_index, directory in enumerate(directory_array):
        print("Current directory: " + str(directory)) # prints out directory we're currently iterating over
        #clearing variables
        t50_array = np.array([], dtype=np.int64).reshape(0,2)
        t95_array = np.array([], dtype=np.int64).reshape(0,2)
        tdif_array = np.array([], dtype=np.int64).reshape(0,2)
    
        first_iteration = True # sets up this boolean for labels
        
        # Iterate through mcmc files in the directory, gathering values
        for mcmcfile in os.listdir(directory): # NOW DICTIONARY!!!!!
            mcmcfile = os.path.join(directory, mcmcfile)
            #print('Making plots for '+str(mcmcfile))

            res = np.load(mcmcfile, allow_pickle=True)['res'][()]

            print('----- Object ID: ' + str(res.objname) + ' -----')
            
            # oops i accidentally calculated t95 wrong (for output b/c nan values present)
            res.output_t50 = res.output_lbt[np.argmin(np.abs(0.5 - res.cmf[:,2]))]
            res.output_t95 = res.output_lbt[np.argmin(np.abs(0.95 - res.cmf[:,2]))]
            
            # append to arrays
            t50_array = np.vstack((t50_array, [res.input_t50, res.output_t50]))
            t95_array = np.vstack((t95_array, [res.input_t95, res.output_t95]))
            tdif_array = np.vstack((tdif_array, [np.abs(res.input_t95 - res.input_t50), np.abs(res.output_t95 - res.output_t50)]))

        ax[0].scatter(t95_array[:,0], t95_array[:,1], color=colors[zcounter], label=zcounter+1 if first_iteration else "")
        ax[1].scatter(t50_array[:,0], t50_array[:,1], color=colors[zcounter])
        ax[2].scatter(tdif_array[:,0], tdif_array[:,1], color=colors[zcounter])
        
        first_iteration= False

# being a control freak
plt.rc('font', size=11)          # controls default text sizes
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=11)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=12)  # fontsize of the figure title

# ideal recovery lines
ax[0].axline((0, 0), slope=1., ls='--', color='black', lw=2)
ax[1].axline((0, 0), slope=1., ls='--', color='black', lw=2)
ax[2].axline((0, 0), slope=1., ls='--', color='black', lw=2)

# axes labels
ax[0].set_ylabel(r'Recovered $t95$ [Gyr]')
ax[0].set_xlabel(r'Input $t95$ [Gyr]')
ax[1].set_ylabel(r'Recovered $t50$ [Gyr]')
ax[1].set_xlabel(r'Input $t50$ [Gyr]')
ax[2].set_ylabel(r'Recovered $| t95 - t50 |$ [Gyr]')
ax[2].set_xlabel(r'Input $| t95 - t50 |$ [Gyr]')

# major ticks (all .1)
for zcounter in zcounter_array:
    ax[zcounter].yaxis.set_major_formatter('{x:.1f}') 
    ax[zcounter].xaxis.set_major_formatter('{x:.1f}') 

ax[0].legend(title='Redshift')

plt.tight_layout()
plt.show()

# make sure plot directory exists
if not os.path.exists(plotdir):
    os.mkdir(plotdir)

counter=0
filename = 'quench.pdf' #defines filename for all objects
while os.path.isfile(plotdir+filename.format(counter)):
    counter += 1
filename = filename.format(counter) #iterate until a unique file is made
#fig.savefig(plotdir+filename, bbox_inches='tight')
#fig.savefig("/Users/michpark/Sync/Documents/JWST RESEARCH/Interesting Plots/PAPER PLOTS/"+filename, bbox_inches='tight')
  
print('saved quench tests to '+plotdir+filename) 

#plt.close(fig)


        

        



             
