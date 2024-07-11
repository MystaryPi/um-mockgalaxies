'''
Given MB and no MB dict files, 
will iterate through each galaxy dictionary and plot different 
galaxy attributes as scatterplots/histograms. 

python mb-nomb-fit_test.py -- automatically detects z1, z2, z3 dictionary
'''
import numpy as np
from matplotlib import pyplot as plt, ticker as ticker; plt.interactive(True)
from matplotlib.ticker import FormatStrFormatter
import sys
import os
import pandas as pd
import glob

# set some params
tflex=2
nflex=5
nfixed=3

# assign directory #DONT DO IN SHERLOCK - NO SEABORN
'''
if len(sys.argv) > 0:
    mb_directory = str(sys.argv[1]) # example: '/Users/michpark/JWST_Programs/mockgalaxies/final-dicts/z3mb/'
    nomb_directory = str(sys.argv[2]) # example: '/Users/michpark/JWST_Programs/mockgalaxies/final-dicts/z3nomb/'
'''
plotdir = '/Users/michpark/JWST_Programs/mockgalaxies/scatterplots-mb-nomb/'

class Output:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
 
 # being a control freak
plt.rc('font', size=11)          # controls default text sizes
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=11)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=12)  # fontsize of the figure title
      
fig, ax = plt.subplots(4,3,figsize=(12,15))

#directory_array = [mb_directory, nomb_directory]

zcounter_array = [0,1,2]
for zcounter in zcounter_array:
    directory_array = ['/Users/michpark/JWST_Programs/mockgalaxies/final-dicts/z'+str(zcounter+1)+'mb', '/Users/michpark/JWST_Programs/mockgalaxies/final-dicts/z'+str(zcounter+1)+'nomb/']

    for directory_index, directory in enumerate(directory_array):
        print("Current directory: " + str(directory)) # prints out directory we're currently iterating over
        #clearing variables
        dust2_array = []
        zred_array = []
        input_mass_array = []
        output_mass_array = np.array([], dtype=np.int64).reshape(0,3)
        input_SFR = []
        output_SFR = np.array([], dtype=np.int64).reshape(0,3)
        
        x_i = []
        x_o = []
    
        first_iteration = True # sets up this boolean for labels
    
        # Iterate through mcmc files in the directory
        for mcmcfile in os.listdir(directory): # NOW DICTIONARY!!!!!
                mcmcfile = os.path.join(directory, mcmcfile)
                #print('Making plots for '+str(mcmcfile))

                res = np.load(mcmcfile, allow_pickle=True)['res'][()]

                print('----- Object ID: ' + str(res.objname) + ' -----')

                ##### SFR over meaningful timescale #####
                '''
                - 100 Myr - timescale that direct Halpha measurements are sensitive to
                - Built function to take in SFH and an averaging timescale (default 100 Myr) 
                - adds up the total mass formed in that timescale / timescale = average SFR

                timescale: most recent timescale (in Gyr)
                lbt_interp: lookback time of FULL range
                sfh: takes in SFH of FULL range
                '''
                # Square interpolation - SFR(t1) and SFR(t2) are two snapshots, then for t<(t1+t2)/2 you assume SFR=SFR(t1) and t>(t1+t2)/2 you assume SFR=SFR(t2)
            
                def averageSFR(lbt, sfh, timescale):  
                    # SFH * time / total time
                    limited_lbt = np.array([])
                    limited_sfh = np.array([])
                
                    # check if reversed array
                    if(lbt[0] > lbt[1]): 
                        lbt = lbt[::-1]
                        sfh = sfh[::-1]
                
                    for m in range(len(lbt)):
                        if(lbt[m] <= timescale):
                            limited_lbt = np.append(limited_lbt, lbt[m])
                            limited_sfh = np.append(limited_sfh, sfh[m])
                    area_under = 0
                    for n in range(len(limited_lbt)-1):
                        area_under += ((limited_lbt[n+1]+limited_lbt[n])*0.5 - limited_lbt[n]) * (limited_sfh[n] + limited_sfh[n+1])
                
                    # add the last value, ends up being 0 if all the way up to 
                    area_under += limited_sfh[-1] * (timescale - limited_lbt[-1])
                
                    return area_under/timescale
                
                inputAverageSFR = averageSFR(res.input_lbt, res.input_sfh, timescale=0.1)
                outputAverageSFR_LE = averageSFR(res.output_lbt, res.output_sfh[:,1], timescale=0.1)
                outputAverageSFR = averageSFR(res.output_lbt, res.output_sfh[:,2], timescale=0.1)
                outputAverageSFR_UE = averageSFR(res.output_lbt, res.output_sfh[:,3], timescale=0.1)

                dust2_array.append(res.percentiles['dust2'][1])
                zred_array.append(res.percentiles['zred'][1]) 
                
                input_mass_array = np.append(input_mass_array, res.obs['logM'])
                output_mass_array = np.vstack((output_mass_array, res.percentiles['logmass']))
                
                input_SFR = np.append(input_SFR, inputAverageSFR)
                output_SFR = np.vstack((output_SFR, [outputAverageSFR_LE, outputAverageSFR, outputAverageSFR]))
            
                first_iteration = False
         
        # PLOT THE VIOLIN PLOTS (zred, dust2, dust_index - INPUT is same!!!)
        # FIND BIAS + SCATTER
        mass_bias = np.median(output_mass_array[:,1] - input_mass_array)
        mass_scatter = np.std(np.abs(output_mass_array[:,1] - input_mass_array))
        SFR_bias = np.median(output_SFR[:,1] - input_SFR)
        SFR_scatter = np.std(np.abs(output_SFR[:,1] - input_SFR))
        dust2_bias = np.median(dust2_array - np.full(len(dust2_array), res.spsdict['dust2']))
        dust2_scatter = np.std(np.abs(dust2_array - np.full(len(dust2_array), res.spsdict['dust2'])))
        zred_bias = np.median(zred_array - np.full(len(zred_array), res.spsdict['zred']))
        zred_scatter = np.std(np.abs(zred_array - np.full(len(zred_array), res.spsdict['zred'])))
        
        zred_lim = [0.1, 0.6, 0.6] # describes the max scatter for binning
        
        if(directory_index == 0): # medium bands
            # plot results
            zred_bins = (res.spsdict['zred']-zred_lim[zcounter], res.spsdict['zred']+zred_lim[zcounter])
            _, bins_zred, _ = ax[0,zcounter].hist(zred_array, bins=np.linspace(zred_bins[0], zred_bins[1], 10), color='maroon', alpha=0.5)
            _, bins_dust2, _ = ax[3,zcounter].hist(dust2_array, bins=np.linspace(0.0, 1.0, 10), range=[0,1], color='maroon', alpha=0.5)
        
            ax[1,zcounter].errorbar(input_mass_array,output_mass_array[:,1],yerr=np.vstack((output_mass_array[:,1]-output_mass_array[:,0],output_mass_array[:,2]-output_mass_array[:,1])),marker='.', markersize=10, ls='', lw=2, 
                markerfacecolor='maroon',markeredgecolor='maroon',ecolor='maroon',elinewidth=1.4, alpha=0.7,label="Broad+MB" if first_iteration else "")
            ax[2,zcounter].errorbar(input_SFR,output_SFR[:,1], yerr=np.vstack((output_SFR[:,1]-output_SFR[:,0], output_SFR[:,2]-output_SFR[:,1])), marker='.', markersize=10, ls='', lw=2, markerfacecolor='maroon', markeredgecolor='maroon', ecolor='maroon',elinewidth=1.4, alpha=0.7)
            
            # add bias + scatter
            ax[0,zcounter].text(x=0.07, y=0.9, s=r'$\mu$ = {:.3f}'.format(zred_bias), transform=ax[0,zcounter].transAxes, color='maroon')
            ax[0,zcounter].text(x=0.07, y=0.83, s=r'$\sigma$ = {:.3f}'.format(zred_scatter), transform=ax[0,zcounter].transAxes, color='maroon')
            ax[3,zcounter].text(x=0.7, y=0.9, s=r'$\mu$ = {:.3f}'.format(dust2_bias), transform=ax[3,zcounter].transAxes, color='maroon')
            ax[3,zcounter].text(x=0.7, y=0.83, s=r'$\sigma$ = {:.3f}'.format(dust2_scatter), transform=ax[3,zcounter].transAxes, color='maroon')
            
            ax[1,zcounter].text(x=0.7, y=0.3, s=r'$\mu$ = {:.3f}'.format(mass_bias), transform=ax[1,zcounter].transAxes, color='maroon')
            ax[1,zcounter].text(x=0.7, y=0.23, s=r'$\sigma$ = {:.3f}'.format(mass_scatter), transform=ax[1,zcounter].transAxes, color='maroon')
            ax[2,zcounter].text(x=0.7, y=0.3, s=r'$\mu$ = {:.3f}'.format(SFR_bias), transform=ax[2,zcounter].transAxes, color='maroon')
            ax[2,zcounter].text(x=0.7, y=0.23, s=r'$\sigma$ = {:.3f}'.format(SFR_scatter), transform=ax[2,zcounter].transAxes, color='maroon')
            
        if(directory_index == 1): # no medium bands
            # plot results
            ax[0,zcounter].hist(zred_array, bins=bins_zred, color='navy', alpha=0.5)
            ax[3,zcounter].hist(dust2_array, bins=bins_dust2, color='navy', alpha=0.5)
            ax[1,zcounter].errorbar(input_mass_array,output_mass_array[:,1],yerr=np.vstack((output_mass_array[:,1]-output_mass_array[:,0],output_mass_array[:,2]-output_mass_array[:,1])),marker='.', markersize=10, ls='', lw=2, 
                            markerfacecolor='navy',markeredgecolor='navy',ecolor='navy',elinewidth=1.4, alpha=0.7,label="Broad+MB" if first_iteration else "")
            ax[2,zcounter].errorbar(input_SFR,output_SFR[:,1], yerr=np.vstack((output_SFR[:,1]-output_SFR[:,0], output_SFR[:,2]-output_SFR[:,1])), marker='.', markersize=10, ls='', lw=2, markerfacecolor='navy', markeredgecolor='navy', ecolor='navy',elinewidth=1.4, alpha=0.7)
            
            # add bias + scatter
            ax[0,zcounter].text(x=0.07, y=0.74, s=r'$\mu$ = {:.3f}'.format(zred_bias), transform=ax[0,zcounter].transAxes, color='navy')
            ax[0,zcounter].text(x=0.07, y=0.67, s=r'$\sigma$ = {:.3f}'.format(zred_scatter), transform=ax[0,zcounter].transAxes, color='navy')
            ax[3,zcounter].text(x=0.7, y=0.74, s=r'$\mu$ = {:.3f}'.format(dust2_bias), transform=ax[3,zcounter].transAxes, color='navy')
            ax[3,zcounter].text(x=0.7, y=0.67, s=r'$\sigma$ = {:.3f}'.format(dust2_scatter), transform=ax[3,zcounter].transAxes, color='navy')
            
            ax[1,zcounter].text(x=0.7, y=0.14, s=r'$\mu$ = {:.3f}'.format(mass_bias), transform=ax[1,zcounter].transAxes, color='navy')
            ax[1,zcounter].text(x=0.7, y=0.07, s=r'$\sigma$ = {:.3f}'.format(mass_scatter), transform=ax[1,zcounter].transAxes, color='navy')
            ax[2,zcounter].text(x=0.7, y=0.14, s=r'$\mu$ = {:.3f}'.format(SFR_bias), transform=ax[2,zcounter].transAxes, color='navy')
            ax[2,zcounter].text(x=0.7, y=0.07, s=r'$\sigma$ = {:.3f}'.format(SFR_scatter), transform=ax[2,zcounter].transAxes, color='navy')
        
            print(str(zcounter) + " " + str(zred_array))
            ax[0,zcounter].set_xlim(zred_bins[0]-0.05, zred_bins[1]+0.05)
            
                
        ax[0,zcounter].yaxis.set_major_formatter('{x:.0f}') # frequency with no decimals
        ax[0,zcounter].xaxis.set_major_formatter('{x:.1f}') # 1 decimal after for zred
        ax[0,0].set_xticks(np.arange(0.8, 1.2, 0.1)) # special for scaling
        
        ax[1,zcounter].yaxis.set_major_formatter('{x:.1f}') # 1 decimal after for mass
        ax[1,zcounter].xaxis.set_major_formatter('{x:.1f}') # 1 decimal after for mass
        
        ax[3,zcounter].yaxis.set_major_formatter('{x:.0f}') # frequency with no decimals
        ax[3,zcounter].xaxis.set_major_formatter('{x:.1f}')
        
    # Below this point is just scaling
    #ZRED - hist
    ax[0,zcounter].set_xlabel("Recovered redshift")
    ax[0,zcounter].axvline(res.spsdict['zred'], ls='--',color='black', lw=2, label='Input')

    # DUST2 - violin
    ax[3, zcounter].set_xlabel("Recovered dust2")
    ax[3, zcounter].axvline(res.spsdict['dust2'], ls='--',color='black', lw=2)
    ax[3, zcounter].set_xlim(-0.1,1.1)

    # LOGMASS - scatter
    ax[1, zcounter].axline((10.5, 10.5), slope=1, ls='--', color='black', lw=2)
    ax[1, zcounter].set_xlabel(r'Input $\log{\rm{M}_* / \rm{M}_\odot}$')
    ax[1, zcounter].set_ylabel(r'Recovered $\log{\rm{M}_* / \rm{M}_\odot}$')

    # SFR - over meaningful timescale (100 Myr)
    ax[2, zcounter].axline((0, 0), slope=1., ls='--', color='black', lw=2)
    ax[2, zcounter].set_ylabel(r'Recovered $\log{\rm{SFR}} [\rm{M}_\odot yr^{-1}]$')
    ax[2, zcounter].set_xlabel(r'Input $\log{\rm{SFR}} [\rm{M}_\odot yr^{-1}]$')
    ax[2, zcounter].set_xscale('log')
    ax[2, zcounter].set_yscale('log')

for zvalue in zcounter_array:
    ax[0, zvalue].set_title("z = " + str(zvalue+1))
    #ax[0, zvalue].legend()
plt.tight_layout()
plt.show()

# make sure plot directory exists
if not os.path.exists(plotdir):
    os.mkdir(plotdir)

counter=0
filename = '{}_z_all.pdf' #defines filename for all objects
while os.path.isfile(plotdir+filename.format(counter)):
    counter += 1
filename = filename.format(counter) #iterate until a unique file is made
#fig.savefig(plotdir+filename, bbox_inches='tight')
#fig.savefig("/Users/michpark/Sync/Documents/JWST RESEARCH/Interesting Plots/PAPER PLOTS/allzscatter.pdf", bbox_inches='tight')
print('saved mb vs. nomb scatterplot to '+plotdir+filename) 

#plt.close(fig)


        

        



             
