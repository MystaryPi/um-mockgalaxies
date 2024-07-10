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

def stepInterp(ab, val, ts):
    '''ab: agebins vector
    val: the original value (sfr, etc) that we want to interpolate
    ts: new values we want to interpolate to '''
    newval = np.zeros_like(ts) + np.nan
    for i in range(0,len(ab)):
        newval[(ts >= ab[i,0]) & (ts < ab[i,1])] = val[i]  
    return newval 

def quantile(data, percents, weights=None):
    ''' percents in units of 1%
    weights specifies the frequency (count) of data.
    '''
    if weights is None:
        return np.percentile(data, percents)
    ind = np.argsort(data)
    d = data[ind]
    w = weights[ind]
    p = 1.*w.cumsum()/w.sum()*100
    y = np.interp(percents, p, d)
    return y

def trap(x, y):
        return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]))/2. 

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
            
                if(directory_index == 0): # MB
                    #LOGMASS
                    ax[1,zcounter].errorbar(res.obs['logM'],res.percentiles['logmass'][1],yerr=np.vstack((res.percentiles['logmass'][1]-res.percentiles['logmass'][0],res.percentiles['logmass'][2]-res.percentiles['logmass'][1])),marker='.', markersize=10, ls='', lw=2, 
                        markerfacecolor='maroon',markeredgecolor='maroon',ecolor='maroon',elinewidth=1.4, alpha=0.7,label="Broad+MB" if first_iteration else "")
        
                    #SFR over last 100 Myr
                    ax[2,zcounter].errorbar(inputAverageSFR,outputAverageSFR, yerr=np.vstack((outputAverageSFR-outputAverageSFR_LE, outputAverageSFR_UE-outputAverageSFR)), marker='.', markersize=10, ls='', lw=2, markerfacecolor='maroon', markeredgecolor='maroon', ecolor='maroon',elinewidth=1.4, alpha=0.7)
                
                if(directory_index == 1): # No MB
                    #LOGMASS 
                    ax[1, zcounter].errorbar(res.obs['logM'],res.percentiles['logmass'][1],yerr=np.vstack((res.percentiles['logmass'][1]-res.percentiles['logmass'][0],res.percentiles['logmass'][2]-res.percentiles['logmass'][1])), marker='.', markersize=10, ls='', lw=2, 
                        c='navy',markeredgecolor='navy',ecolor='navy',elinewidth=1.4, alpha=0.7,label="Broad only" if first_iteration else "")
                
                    #SFR over last 100 Myr
                    ax[2, zcounter].errorbar(inputAverageSFR,outputAverageSFR, yerr=np.vstack((outputAverageSFR-outputAverageSFR_LE, outputAverageSFR_UE-outputAverageSFR)), marker='.', markersize=10, ls='', lw=2, markerfacecolor='navy', markeredgecolor='navy', ecolor='navy',elinewidth=1.4, alpha=0.7)
                
            
                dust2_array.append(res.percentiles['dust2'][1])
                zred_array.append(res.percentiles['zred'][1]) 
            
                first_iteration = False
         
        # PLOT THE VIOLIN PLOTS (zred, dust2, dust_index - INPUT is same!!!)
        if(directory_index == 0): # medium bands
            _, bins_zred, _ = ax[0,zcounter].hist(zred_array, bins=np.linspace(res.spsdict['zred']-0.35, res.spsdict['zred']+0.35, 30), range=[res.spsdict['zred']-0.35,res.spsdict['zred']+0.35], color='maroon', alpha=0.5)
            _, bins_dust2, _ = ax[3,zcounter].hist(dust2_array, bins=np.linspace(0.0, 1.0, 25), range=[0.0,1.0], color='maroon', alpha=0.5)
        if(directory_index == 1): # no medium bands
            ax[0,zcounter].hist(zred_array, bins=bins_zred, range=[res.spsdict['zred']-0.35,res.spsdict['zred']+0.35], color='navy', alpha=0.5)
            ax[3,zcounter].hist(dust2_array, bins=bins_dust2, range=[0.0,1.0], color='navy', alpha=0.5)
        
    # Below this point is just scaling
    #ZRED - hist
    ax[0,zcounter].set_ylabel("Recovered redshift")
    ax[0,zcounter].set_xlabel("Input redshift")
    ax[0,zcounter].axvline(res.spsdict['zred'], ls='--',color='black', lw=2, label='Input redshift: {0:.3f}'.format(res.spsdict['zred']))
    ax[0,zcounter].set_xlim(res.spsdict['zred']-0.35,res.spsdict['zred']+0.35)

    # DUST2 - violin
    ax[3,zcounter].set_xlabel("Input dust2")
    ax[3, zcounter].set_ylabel("Recovered dust2")
    ax[3, zcounter].axvline(0.2, ls='--',color='black', lw=2, label='Input dust2: 0.2')
    ax[3, zcounter].set_xlim(0,1.0)

    # LOGMASS - scatter
    ax[1, zcounter].axline((10.5, 10.5), slope=1, ls='--', color='black', lw=2)
    ax[1, zcounter].set_xlabel(r'Input $log M_{stellar}$ (log $M_{sun}$)')
    ax[1, zcounter].set_ylabel(r'Recovered $log M_{stellar}$ (log $M_{sun}$)')
    ax[1, zcounter].legend(fontsize=10) # the one legend

    # SFR - over meaningful timescale (100 Myr)
    ax[2, zcounter].axline((0, 0), slope=1., ls='--', color='black', lw=2)
    ax[2, zcounter].set_ylabel(r'Recovered $log SFR_{ave, 100 Myr}$ (log $M_{sun}$ / yr)')
    ax[2, zcounter].set_xlabel(r'Input $log SFR_{ave, 100 Myr}$ (log $M_{sun}$ / yr)')
    ax[2, zcounter].set_xscale('log')
    ax[2, zcounter].set_yscale('log')

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
fig.savefig("/Users/michpark/Sync/Documents/JWST RESEARCH/Interesting Plots/PAPER PLOTS/allzscatter.pdf", bbox_inches='tight')
print('saved mb vs. nomb scatterplot to '+plotdir+filename) 

#plt.close(fig)


        

        



             
