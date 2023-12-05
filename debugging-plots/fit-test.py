import numpy as np
from prospect.models import priors, SedModel
from prospect.models.sedmodel import PolySedModel
from prospect.models.templates import TemplateLibrary
from prospect.sources import CSPSpecBasis
from sedpy.observate import load_filters
import sedpy
from astropy.io import fits
from scipy import signal
from scipy import interpolate
import dynesty
import h5py
from matplotlib import pyplot as plt, ticker as ticker; plt.interactive(True)
from matplotlib.ticker import FormatStrFormatter
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
import um_cornerplot
from prospect.models.transforms import logsfr_ratios_to_masses
import sys
from scipy.interpolate import interp1d
import os
from prospect.io.read_results import results_from, get_sps
from prospect.io.read_results import traceplot, subcorner
import fsps
import seaborn as sns
import pandas as pd
import glob

from prospect.models.transforms import logsfr_ratios_to_masses_psb, psb_logsfr_ratios_to_agebins

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

def get_percentiles(res,mod, ptile=[16, 50, 84], start=0.0, thin=1, **extras):
    """Get get percentiles of the marginalized posterior for each parameter.

    :param res:
        A results dictionary, containing a "chain" and "theta_labels" keys.

    :param ptile: (optional, default: [16, 50, 84])
       A list of percentiles (integers 0 to 100) to return for each parameter.

    :param start: (optional, default: 0.5)
       How much of the beginning of chains to throw away before calculating
       percentiles, expressed as a fraction of the total number of iterations.

    :param thin: (optional, default: 10.0)
       Only use every ``thin`` iteration when calculating percentiles.

    :returns pcts:
       Dictionary with keys giving the parameter names and values giving the
       requested percentiles for that parameter.
    """

    parnames = np.array(res.get('theta_labels', mod.theta_labels()))
    niter = res['chain'].shape[-2]
    start_index = np.floor(start * (niter-1)).astype(int)
    if res["chain"].ndim > 2:
        flatchain = res['chain'][:, start_index::thin, :]
        dims = flatchain.shape
        flatchain = flatchain.reshape(dims[0]*dims[1], dims[2])
    elif res["chain"].ndim == 2:
        flatchain = res["chain"][start_index::thin, :]
    pct = np.array([quantile(p, ptile, weights=res.get("weights", None)) for p in flatchain.T])
    return dict(zip(parnames, pct)) 
 
def intersection_function(x, y1, y2):
    """Find the intercept of two curves, given by the same x data"""

    def intercept(point1, point2, point3, point4):
        """find the intersection between two lines
        the first line is defined by the line between point1 and point2
        the first line is defined by the line between point3 and point4
        each point is an (x,y) tuple.

        So, for example, you can find the intersection between
        intercept((0,0), (1,1), (0,1), (1,0)) = (0.5, 0.5)

        Returns: the intercept, in (x,y) format
        """    

        def line(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0]*p2[1] - p2[0]*p1[1])
            return A, B, -C

        def intersection(L1, L2):
            D  = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]

            x = Dx / D
            y = Dy / D
            return x,y

        L1 = line([point1[0],point1[1]], [point2[0],point2[1]])
        L2 = line([point3[0],point3[1]], [point4[0],point4[1]])

        R = intersection(L1, L2)

        return R

    idx = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)
    xc, yc = intercept((x[idx], y1[idx]),((x[idx+1], y1[idx+1])), ((x[idx], y2[idx])), ((x[idx+1], y2[idx+1])))
    return xc,yc

def trap(x, y):
        return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]))/2. 

# set some params
tflex=2
nflex=5
nfixed=3

# assign directory #DONT DO IN SHERLOCK - NO SEABORN
directory = '/Users/michpark/JWST_Programs/mockgalaxies/nov_z3/' #nov_z3 vs. tlast_z3fixed
plotdir = '/Users/michpark/JWST_Programs/mockgalaxies/big_plots/'
cosmo = FlatLambdaCDM(H0=70, Om0=.3)

#command line argument (whole thing or just one)
#if len(sys.argv) > 0:
#    dictfile = sys.argv[1]  
#else: can test whole thing later ig
    
 
# iterate over files in that directory
# YES - DICT FILES ARE UNIQUE

# violin vs scatterplot
# the sketchiest method i cant
#mcmcCounter = len(glob.glob1(directory,"*.h5")

dust2_array = []
dust_index_array = []
zred_array = []

fig, ax = plt.subplots(4,2,figsize=(9,9))

for mcmcfile in os.listdir(directory):
        #SPECIFICALLY 121104703 - new_mcmc_29_121104703_1688582046_mcmc.h5 (nan example)
        #SPECIFICALLY 121072637 - new_mcmc_6_121072637_1688582046_mcmc.h5 (input sfh beyond output sfh)
        mcmcfile = os.path.join(directory, mcmcfile)
        #print('Making plots for '+str(mcmcfile))

        res, obs, mod = results_from("{}".format(mcmcfile), dangerous=True)
        gal = (np.load('obs/umobs_'+str(obs['objid'])+'.npz', allow_pickle=True))['gal']
        spsdict = (np.load('obs/umobs_'+str(obs['objid'])+'.npz', allow_pickle=True))['params'][()]

        sps = get_sps(res)

        print('----- Object ID: ' + str(obs['objid']) + ' -----')
        
        # obtain sfh from universemachine
        um_sfh = gal['sfh'][:,1]
        #error bars (quantiles) + values for zred, logzsol, dust2, logmass, dust_index
        
        # SFH THINGS
        # actual sfh percentiles
        flatchain = res["chain"]
        niter = res['chain'].shape[-2]
        if 'zred' in mod.theta_index:
            tmax = cosmo.age(np.min(flatchain[:,mod.theta_index['zred']])).value #matches scales
        else: 
            tmax = cosmo.age(obs['zred']).value

        if tmax > 2:
            age_interp = np.concatenate((np.arange(0,2,.001),np.arange(2,tmax,.01),[tmax])) #np.append(np.arange(0, tuniv, 0.01), tuniv)
        else:
            age_interp = np.arange(0,tmax+.005, .001)    
        age_interp[0] = 1e-9      

        # actual sfh percentiles
        allsfrs = np.zeros((flatchain.shape[0], len(mod.params['agebins'])))
        allagebins = np.zeros((flatchain.shape[0], len(mod.params['agebins']), 2))
        for iteration in range(flatchain.shape[0]):
            logr = np.clip(flatchain[iteration, mod.theta_index["logsfr_ratios"]], -7,7)
            tlast = flatchain[iteration, mod.theta_index['tlast']]
            logr_young = np.clip(flatchain[iteration, mod.theta_index['logsfr_ratio_young']], -7, 7)
            logr_old = np.clip(flatchain[iteration, mod.theta_index['logsfr_ratio_old']], -7, 7)
            try:
                logmass = flatchain[iteration, mod.theta_index['massmet']][0] # not what we are using
            except:
                logmass = flatchain[iteration, mod.theta_index["logmass"]]      
            agebins = psb_logsfr_ratios_to_agebins(logsfr_ratios=logr, agebins=mod.params['agebins'], 
                tlast=tlast, tflex=mod.params['tflex'], nflex=mod.params['nflex'], nfixed=mod.params['nfixed'])
            allagebins[iteration, :] = agebins
            dt = 10**agebins[:, 1] - 10**agebins[:, 0]
            masses = logsfr_ratios_to_masses_psb(logsfr_ratios=logr, logmass=logmass, agebins=agebins,
                logsfr_ratio_young=logr_young, logsfr_ratio_old=logr_old,
                tlast=tlast, tflex=mod.params['tflex'], nflex=mod.params['nflex'], nfixed=mod.params['nfixed'])
            allsfrs[iteration,:] = (masses  / dt)

        # calculate interpolated SFR and cumulative mass  
        # with each likelihood draw you can convert the agebins from units of lookback time to units of age 
        # using the redshift at that likelihood draw, and put it on your fixed grid of ages
        allagebins_ago = 10**allagebins/1e9  
        allsfrs_interp = np.zeros((flatchain.shape[0], len(age_interp)))
        masscum_interp = np.zeros_like(allsfrs_interp)
        totmasscum_interp = np.zeros_like(allsfrs_interp)
        dt = (age_interp - np.insert(age_interp,0,0)[:-1]) * 1e9
        for i in range(flatchain.shape[0]):
            allsfrs_interp[i,:] = stepInterp(allagebins_ago[i,:], allsfrs[i,:], age_interp)
            allsfrs_interp[i,-1] = 0
            masscum_interp[i,:] = 1 - (np.cumsum(allsfrs_interp[i,:] * dt) / np.sum(allsfrs_interp[i,:] * dt))
            totmasscum_interp[i,:] = np.sum(allsfrs_interp[i,:] * dt) - (np.cumsum(allsfrs_interp[i,:] * dt))

        # sfr and cumulative mass percentiles 
        sfrPercent = np.array([quantile(allsfrs_interp[:,i], [16,50,84], weights=res.get('weights', None)) 
            for i in range(allsfrs_interp.shape[1])])
        sfrPercent = np.concatenate((age_interp[:,np.newaxis], sfrPercent), axis=1) # add time  
        massPercent = np.array([quantile(masscum_interp[:,i], [16,50,84], weights=res.get('weights', None)) 
            for i in range(masscum_interp.shape[1])])    
        massPercent = np.concatenate((age_interp[:,np.newaxis], massPercent), axis=1) # add time          
        totmassPercent = np.array([quantile(totmasscum_interp[:,i], [16,50,84], weights=res.get('weights', None)) 
            for i in range(totmasscum_interp.shape[1])])    
        totmassPercent = np.concatenate((age_interp[:,np.newaxis], totmassPercent), axis=1) # add time

        # all percentiles...
        percentiles = get_percentiles(res, mod) # stores 16 50 84 percentiles for dif parameters
        massFrac = 1 - massPercent[age_interp==1, 1:].flatten()[::-1]  

        #print("obs logM: " + str(obs['logM']))
        #print("um sfh integral mass: " + str(np.log10(trap(cosmo.age(gal['sfh'][:,0]).value*1e9, um_sfh))))
        #print("output original:" + str(logmass_array[1]))
        #print("output integral mass: " + str(np.log10(trap(cosmo.age(gal['sfh'][:,0]).value*1e9, sfhadjusted))))
        #outputIntegralMass = np.log10(trap(cosmo.age(gal['sfh'][:,0]).value*1e9, sfhadjusted))

        ##### AREA UNDER THE CURVE COMPARISON #####
        # Full range vs recent bit (0.6 Gyr) vs old bit
        inputFullLBT = cosmo.age(obs['zred']).value - cosmo.age(gal['sfh'][:,0]).value
        outputFullLBT = age_interp
        inputFull = np.log10(np.abs(trap(inputFullLBT * 1e9, um_sfh)))
        outputFull = np.log10(np.abs(trap(outputFullLBT * 1e9, sfrPercent[:,2])))

        # Split into recent and old LBTs for both input + output LBTs
        # multiplied by 1e9 in this step due to type errors
        inputRecentLBT = [inputFullLBT[i] * 1e9 for i in range(len(inputFullLBT)) if inputFullLBT[i] <= 0.6]
        inputOldLBT = [inputFullLBT[i] * 1e9 for i in range(len(inputFullLBT)) if inputFullLBT[i] > 0.6]
        outputRecentLBT = [outputFullLBT[i] * 1e9 for i in range(len(outputFullLBT)) if outputFullLBT[i] <= 0.6]
        outputOldLBT = [outputFullLBT[i] * 1e9 for i in range(len(outputFullLBT)) if outputFullLBT[i] > 0.6]

        inputRecent = np.log10(np.abs(trap(np.array(inputRecentLBT), um_sfh[len(inputOldLBT):len(inputFullLBT)])))
        inputOld = np.log10(np.abs(trap(np.array(inputOldLBT), um_sfh[0:len(inputOldLBT)])))

        outputRecent = np.log10(np.abs(trap(np.array(outputRecentLBT), sfrPercent[:,2][0:len(outputRecentLBT)])))
        outputOld = np.log10(np.abs(trap(np.array(outputOldLBT), sfrPercent[:,2][0:len(outputOldLBT)]))) 
        ax[3,1].plot(inputFull, outputFull, marker='.', markersize=10, ls='', lw=2, color = 'black')
        ax[3,1].plot(inputRecent, outputRecent, marker='.', markersize=10, ls='', lw=2, color = 'skyblue')
        ax[3,1].plot(inputOld, outputOld, marker='.', markersize=10, ls='', lw=2, color = 'maroon')

        #LOGMASS
        ax[0,0].errorbar(obs['logM'],percentiles['logmass'][1],yerr=np.vstack((percentiles['logmass'][1]-percentiles['logmass'][0],percentiles['logmass'][2]-percentiles['logmass'][1])),marker='.', markersize=10, ls='', lw=2, 
            markerfacecolor='navy',markeredgecolor='navy',markeredgewidth=3,ecolor='navy',elinewidth=1.4) 
        
        #SFRs - get last value of um_sfh + 0th value of sfrpercent (both most recent values)
        ax[0,1].errorbar(um_sfh[-1], sfrPercent[:,2][0], yerr=np.vstack((sfrPercent[:,2][0] - sfrPercent[:,1][0], sfrPercent[:,3][0]-sfrPercent[:,2][0])),marker='.', markersize=10, ls='', lw=2, 
            markerfacecolor='dodgerblue',markeredgewidth=3,markeredgecolor='dodgerblue', ecolor='dodgerblue',elinewidth=1.4) 
        print("#### SFH: " + str(um_sfh[-1]) + ", " + str(sfrPercent[:,2][0]) + " ####")
        #LOGZSOL
        ax[1,0].errorbar(spsdict['logzsol'],percentiles['logzsol'][1],yerr=np.vstack((percentiles['logzsol'][1]-percentiles['logzsol'][0],percentiles['logzsol'][2]-percentiles['logzsol'][1])),marker='.', markersize=10, ls='', lw=2, 
            markerfacecolor='darkorange',markeredgecolor='darkorange',markeredgewidth=3,ecolor='darkorange',elinewidth=1.4) 
        #ax[1,0].errorbar(spsdict['logzsol'],logzsol[1],marker='.', markersize=10, ls='', lw=2, markerfacecolor='darkorange',markeredgecolor='darkorange',markeredgewidth=3) 

        dust2_array.append(percentiles['dust2'][1])
        dust_index_array.append(percentiles['dust_index'][1])
        #zred_array.append(zred[0])

        # QUENCH TIME 
        input_mask = [i for i in enumerate(um_sfh) if i == 0]
        output_mask = [i for i, n in enumerate(sfrPercent[:,2]) if n == 0]

        input_sfh = um_sfh[::-1]
        input_lbt = (cosmo.age(obs['zred']).value - cosmo.age(gal['sfh'][:,0]).value)[::-1]
        output_sfh = sfrPercent[:,2]
        output_lbt = age_interp

        for i in sorted(input_mask, reverse=True):
            input_sfh = np.delete(input_sfh, i)
            input_lbt = np.delete(input_lbt, i)
        for i in sorted(output_mask, reverse=True):
            output_sfh = np.delete(output_sfh, i)
            output_lbt = np.delete(output_lbt, i)

        lbtLimit = (cosmo.age(obs['zred']).value - cosmo.age(gal['sfh'][:,0]).value)[0]
        output_lbt_mask = [i for i, n in enumerate(output_lbt) if n > lbtLimit]
        for i in sorted(output_lbt_mask, reverse=True): # go in reverse order to prevent indexing error
            output_sfh = np.delete(output_sfh, i)
            output_lbt = np.delete(output_lbt, i)

        # Find derivatives of input + output SFH, age is adjusted b/c now difference between points
        y_d_input = -np.diff(input_sfh) / np.diff(input_lbt) 
        y_d_output = -np.diff(output_sfh) / np.diff(output_lbt)
        x_d_input = (np.array(input_lbt)[:-1] + np.array(input_lbt)[1:]) / 2
        x_d_output = (np.array(output_lbt)[:-1] + np.array(output_lbt)[1:]) / 2

        # Use intersect package to determine where derivatives intersect the quenching threshold
        x_i, y_i = intersection_function(x_d_input, np.full(len(x_d_input), -500), y_d_input)
        x_o, y_o = intersection_function(x_d_output, np.full(len(x_d_output), -500), y_d_output)

        # both quench times must be present
        if len(x_i) != 0 and len(x_o) != 0:
            ax[3,0].plot(x_i[0], x_o[0], marker='.', markersize=10, ls='', lw=2, 
            markerfacecolor='c',markeredgecolor='c',markeredgewidth=3)

            



         
# PLOT THE VIOLIN PLOTS (zred, dust2, dust_index - INPUT is same!!!)
#ZRED - hist
'''
ax[1,1].hist(zred_array, bins=20, range=[1.9,3.1], color='lightcoral')
ax[1,1].set_xlabel("Recovered redshift")
ax[1,1].axvline(spsdict['zred'], ls='--',color='black', lw=2, label='Input redshift: {0:.3f}'.format(spsdict['zred']))
ax[1,1].set_xlim(1.9,3.1)
ax[1,1].set_ylim(0,40)
'''

# DUST2 - violin
ax[2,0].hist(dust2_array, bins=20, color='silver')
ax[2,0].set_xlabel("Recovered dust2")
ax[2,0].axvline(0.2, ls='--',color='black', lw=2, label='Input dust2: 0.0')
ax[2,0].set_xlim(-0.2,2.5)
ax[2,0].set_ylim(0,40)

#DUST_INDEX - violin
ax[2,1].hist(dust_index_array, bins=20, color='gray')
ax[2,1].set_xlabel("Recovered dust_index")
ax[2,1].axvline(0, ls='--',color='black', lw=2, label='Input dust index: 0.0')
ax[2,1].set_xlim(-1.2,0.6)
ax[2,1].set_ylim(0,40) 

# LOGMASS - scatter
ax[0,0].axline((10.5, 10.5), slope=1., ls='--', color='black', lw=2)
ax[0,0].set_xlabel(r'Input $log M_{stellar}$ (log $M_{sun}$)')
ax[0,0].set_ylabel(r'Recovered $log M_{stellar}$ (log $M_{sun}$)')

# SFR - scatter - TBD
ax[0,1].axline((0, 0), slope=1., ls='--', color='black', lw=2)
ax[0,1].set_xlabel(r'Input SFR ($M_{sun}/yr$)')
ax[0,1].set_ylabel(r'Recovered SFR ($M_{sun}/yr$)')
ax[0,1].set_xscale('log')
ax[0,1].set_yscale('log')

# LOGZSOL - scatter
ax[1,0].axline((0, 0), slope=1., ls='--', color='black', lw=2)
ax[1,0].set_xlim(-0.35,0.5)
ax[1,0].set_ylim(-2,0.3)
ax[1,0].set_xlabel(r'Input $log(Z/Z_{\odot})$')
ax[1,0].set_ylabel(r'Recovered $log(Z/Z_{\odot})$')

# QUENCHTIME - scatter
ax[3,0].axline((0, 0), slope=1., ls='--', color='black', lw=2)
ax[3,0].set_ylabel(r'Recovered quench time [Gyr]')
ax[3,0].set_xlabel(r'Input quench time [Gyr]')

# Different mass integrals - scatter
# Blue = recent, maroon = old, black = combined/full
ax[3,1].axline((0, 0), slope=1., ls='--', color='black', lw=2)
ax[3,1].set_ylabel(r'Recovered mass integral')
ax[3,1].set_xlabel(r'Input mass integral')
ax[3,1].set_ylim(6, 12)
ax[3,1].set_xlim(8.5,12)
ax[3,1].set_xlim(left=5)
ax[3,1].set_ylim(bottom=5)

plt.tight_layout()
plt.show()

# save plot 
counter=0
filename = 'bigplot_tlast_z3_{}.pdf' #defines filename for all objects
while os.path.isfile(plotdir+filename.format(counter)):
    counter += 1
filename = filename.format(counter) #iterate until a unique file is made
fig.savefig(plotdir+filename, bbox_inches='tight')
  
print('saved big plot to '+plotdir+filename) 

#plt.close(fig)
