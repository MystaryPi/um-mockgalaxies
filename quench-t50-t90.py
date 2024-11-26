'''
Given MB and no MB dict files, 
will iterate through each galaxy dictionary and plot different 
galaxy attributes as scatterplots/histograms. 

t95, t50, t95-t50 scatterplots

python quench-t50-t90.py -- automatically detects z1, z2, z3 dictionary
'''
import numpy as np
import matplotlib as mpl
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
        
fig, ax = plt.subplots(3,2,figsize=(10, 11))
# being a control freak
plt.rc('font', size=11)          # controls default text sizes
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=11)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=12)  # fontsize of the figure title

# assign directory #DONT DO IN SHERLOCK - NO SEABORN
'''
if len(sys.argv) > 1:
    mb_directory = str(sys.argv[1]) # example: '/Users/michpark/JWST_Programs/mockgalaxies/final/z3mb/'
    nomb_directory = str(sys.argv[2]) # example: '/Users/michpark/JWST_Programs/mockgalaxies/final/z3nomb/'
'''

plotdir = '/Users/michpark/JWST_Programs/mockgalaxies/scatterplots-mb-nomb/'

z_array = [1, 2, 3, 4.5, 5]
cmap = mpl.colormaps['viridis']
colors = [cmap(0), cmap(0.25), cmap(0.5), cmap(0.875), cmap(0.99)]

t95_all_mb = np.array([], dtype=np.int64).reshape(0,2)
t50_all_mb = np.array([], dtype=np.int64).reshape(0,2)
tdif_all_mb = np.array([], dtype=np.int64).reshape(0,2)
t95_all_nomb = np.array([], dtype=np.int64).reshape(0,2)
t50_all_nomb = np.array([], dtype=np.int64).reshape(0,2)
tdif_all_nomb = np.array([], dtype=np.int64).reshape(0,2)

for zcounter, z in enumerate(z_array):
    zstring = str(z).replace('.', 'p') 
    directory_array = ['/Users/michpark/JWST_Programs/mockgalaxies/final-dicts/z'+zstring+'mb', '/Users/michpark/JWST_Programs/mockgalaxies/final-dicts/z'+zstring+'nomb']

    for directory_index, directory in enumerate(directory_array):
        print("Current directory: " + str(directory)) # prints out directory we're currently iterating over
        print(directory_index)

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
            res.output_t50 = res.output_lbt[np.nanargmin(np.abs(0.5 - res.cmf[:,2]))]
            res.output_t95 = res.output_lbt[np.nanargmin(np.abs(0.95 - res.cmf[:,2]))]
            
            if(res.output_t95 > 2.5):
                print(res.obs['objid'])
            
            # append to arrays
            t50_array = np.vstack((t50_array, [res.input_t50, res.output_t50]))
            t95_array = np.vstack((t95_array, [res.input_t95, res.output_t95]))
            tdif_array = np.vstack((tdif_array, [np.abs(res.input_t95 - res.input_t50), np.abs(res.output_t95 - res.output_t50)]))

        ax[0,directory_index].scatter(t95_array[:,0], t95_array[:,1], c=colors[zcounter], zorder=zcounter)
        ax[1,directory_index].scatter(t50_array[:,0], t50_array[:,1], c=colors[zcounter], zorder=zcounter)
        ax[2,directory_index].scatter(tdif_array[:,0], tdif_array[:,1], c=colors[zcounter], zorder=zcounter)
        
        # add to a big array for bias/scatter
        if directory_index == 0:
            t95_all_mb = np.vstack((t95_all_mb, t95_array))
            t50_all_mb = np.vstack((t50_all_mb, t50_array))
            tdif_all_mb = np.vstack((tdif_all_mb, tdif_array))
        else:
            t95_all_nomb = np.vstack((t95_all_nomb, t95_array))
            t50_all_nomb = np.vstack((t50_all_nomb, t50_array))
            tdif_all_nomb = np.vstack((tdif_all_nomb, tdif_array)) 
        
## add the scatter + bias
t95_bias_mb = np.median(t95_all_mb[:,1] - t95_all_mb[:,0])
t95_scatter_mb = np.std(np.abs(t95_all_mb[:,1] - t95_all_mb[:,0]))
t50_bias_mb = np.median(t50_all_mb[:,1] - t50_all_mb[:,0])
t50_scatter_mb = np.std(np.abs(t50_all_mb[:,1] - t50_all_mb[:,0]))
tdif_bias_mb = np.median(tdif_all_mb[:,1] - tdif_all_mb[:,0])
tdif_scatter_mb = np.std(np.abs(tdif_all_mb[:,1] - tdif_all_mb[:,0]))

t95_bias_nomb = np.median(t95_all_nomb[:,1] - t95_all_nomb[:,0])
t95_scatter_nomb = np.std(np.abs(t95_all_nomb[:,1] - t95_all_nomb[:,0]))
t50_bias_nomb = np.median(t50_all_nomb[:,1] - t50_all_nomb[:,0])
t50_scatter_nomb = np.std(np.abs(t50_all_nomb[:,1] - t50_all_nomb[:,0]))
tdif_bias_nomb = np.median(tdif_all_nomb[:,1] - tdif_all_nomb[:,0])
tdif_scatter_nomb = np.std(np.abs(tdif_all_nomb[:,1] - tdif_all_nomb[:,0]))
        
# ADD TEXT
ax[0,0].text(x=0.7, y=0.14, s=r'$\mu$ = {:.3f}'.format(t95_bias_mb), transform=ax[0,0].transAxes, color='maroon')
ax[0,0].text(x=0.7, y=0.07, s=r'$\sigma$ = {:.3f}'.format(t95_scatter_mb), transform=ax[0,0].transAxes, color='maroon')
ax[1,0].text(x=0.7, y=0.14, s=r'$\mu$ = {:.3f}'.format(t50_bias_mb), transform=ax[1,0].transAxes, color='maroon')
ax[1,0].text(x=0.7, y=0.07, s=r'$\sigma$ = {:.3f}'.format(t50_scatter_mb), transform=ax[1,0].transAxes, color='maroon')
ax[2,0].text(x=0.7, y=0.14, s=r'$\mu$ = {:.3f}'.format(tdif_bias_mb), transform=ax[2,0].transAxes, color='maroon')
ax[2,0].text(x=0.7, y=0.07, s=r'$\sigma$ = {:.3f}'.format(tdif_scatter_mb), transform=ax[2,0].transAxes, color='maroon')

ax[0,1].text(x=0.07, y=0.9, s=r'$\mu$ = {:.3f}'.format(t95_bias_nomb), transform=ax[0,1].transAxes, color='navy')
ax[0,1].text(x=0.07, y=0.83, s=r'$\sigma$ = {:.3f}'.format(t95_scatter_nomb), transform=ax[0,1].transAxes, color='navy')
ax[1,1].text(x=0.7, y=0.14, s=r'$\mu$ = {:.3f}'.format(t50_bias_nomb), transform=ax[1,1].transAxes, color='navy')
ax[1,1].text(x=0.7, y=0.07, s=r'$\sigma$ = {:.3f}'.format(t50_scatter_nomb), transform=ax[1,1].transAxes, color='navy')
ax[2,1].text(x=0.7, y=0.14, s=r'$\mu$ = {:.3f}'.format(tdif_bias_nomb), transform=ax[2,1].transAxes, color='navy')
ax[2,1].text(x=0.7, y=0.07, s=r'$\sigma$ = {:.3f}'.format(tdif_scatter_nomb), transform=ax[2,1].transAxes, color='navy')

# ideal recovery lines
for axes in ax.reshape(-1):
    axes.axline((0, 0), slope=1., ls='--', color='black', lw=2) 
    axes.yaxis.set_major_formatter('{x:.1f}') #set major ticks
    axes.xaxis.set_major_formatter('{x:.1f}') 

# axes labels
for directory_index, _ in enumerate(directory_array):
    ax[0,directory_index].set_ylabel(r'Recovered $t95$ [Gyr]')
    ax[0,directory_index].set_xlabel(r'Input $t95$ [Gyr]')
    ax[1,directory_index].set_ylabel(r'Recovered $t50$ [Gyr]')
    ax[1,directory_index].set_xlabel(r'Input $t50$ [Gyr]')
    ax[2,directory_index].set_ylabel(r'Recovered $| t95 - t50 |$ [Gyr]')
    ax[2,directory_index].set_xlabel(r'Input $| t95 - t50 |$ [Gyr]')

#plt.tight_layout()
ax[0,0].set_title("UNCOVER+MB")
ax[0,1].set_title("UNCOVER only")
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
norm = mpl.colors.Normalize(vmin=1, vmax=5)
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, label='Redshift')

plt.show()

# make sure plot directory exists
if not os.path.exists(plotdir):
    os.mkdir(plotdir)

counter=0
filename = 'quench_stretch.pdf' #defines filename for all objects
while os.path.isfile(plotdir+filename.format(counter)):
    counter += 1
filename = filename.format(counter) #iterate until a unique file is made
#fig.savefig(plotdir+filename, bbox_inches='tight')
#fig.savefig("/Users/michpark/Sync/Documents/JWST RESEARCH/Interesting Plots/PAPER PLOTS/"+filename, bbox_inches='tight')
  
print('saved quench tests to '+plotdir+filename) 

#plt.close(fig)


        

        



             
