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
        newval[(ts <= ab[i,0]) & (ts > ab[i,1])] = val[i] #previously >= and <
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
    
def stepInterp(ab, val, ts):
    '''ab: agebins vector
    val: the original value (sfr, etc) that we want to interpolate
    ts: new values we want to interpolate to '''
    newval = np.zeros_like(ts) + np.nan
    for i in range(0,len(ab)):
        newval[(ts <= ab[i,0]) & (ts > ab[i,1])] = val[i] #previously >= and <
    return newval    

def trap(x, y):
        return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]))/2. 

# assign directory #DONT DO IN SHERLOCK - NO SEABORN
directory = '/Users/michpark/JWST_Programs/mockgalaxies/z3_fixedzred/' #FOR THE UPDATED PARAM 7/6 BATCH
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
#mcmcCounter = len(glob.glob1(directory,"*.h5"))

logmass_array = [[0]*3] #scatter
sfr = [[0]*3] #scatter
#zred_df = pd.DataFrame() #violin, set at 0.9743 something
logzsol_array = [[0]*3] #scatter
#dust2_df = pd.DataFrame() #violin, set at 0
#dust_index_df = pd.DataFrame() #violin, set at 0

dust2_array = []
dust_index_array = []
zred_array = []

fig, ax = plt.subplots(3,2,figsize=(9,9))

for mcmcfile in os.listdir(directory):
    if mcmcfile.startswith('z3_fixedzred'): 
        #SPECIFICALLY 121104703 - new_mcmc_29_121104703_1688582046_mcmc.h5 (nan example)
        #SPECIFICALLY 121072637 - new_mcmc_6_121072637_1688582046_mcmc.h5 (input sfh beyond output sfh)
        mcmcfile = os.path.join(directory, mcmcfile)
        #print('Making plots for '+str(mcmcfile))

        res, obs, mod = results_from("{}".format(mcmcfile), dangerous=True)
        gal = (np.load('obs-z3/umobs_'+str(obs['objid'])+'.npz', allow_pickle=True))['gal']
        spsdict = (np.load('obs-z3/umobs_'+str(obs['objid'])+'.npz', allow_pickle=True))['params'][()]

        sps = get_sps(res)

        print('----- Object ID: ' + str(obs['objid']) + ' -----')
        # obtain sfh from universemachine
        um_sfh = gal['sfh'][:,1]

        # CORNERPLOT TYPE BEAT
        #['zred','logzsol','dust2','logmass','tlast','logsfr_ratio_young','logsfr_ratio_old_1','logsfr_ratio_old_2',
        # 'logsfr_ratio_old_3','logsfr_ratios_1','logsfr_ratios_2','logsfr_ratios_3','logsfr_ratios_4','dust_index']

        #error bars (quantiles) + values for zred, logzsol, dust2, logmass, dust_index
        #logmass_array = quantile(res['chain'][i, mod.theta_index['logmass']], [16,50,84],weights=res.get("weights", None))
        
        # SFH THINGS
        # actual sfh percentiles
        flatchain = res["chain"]
        niter = res['chain'].shape[-2]
        #tmax = cosmo.age(np.min(flatchain[:,mod.theta_index['zred']])).value
        tmax = 13.3 

        age_interp = np.arange(0,tmax+.005, .001)  #in age
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
                logmass = flatchain[iteration, mod.theta_index['massmet']][0] 
            except:
                logmass = flatchain[iteration, mod.theta_index["logmass"]]   # MASS COLLECTED HERE 
            agebins = psb_logsfr_ratios_to_agebins(logsfr_ratios=logr, agebins=mod.params['agebins'], tlast=tlast, 
                tflex=mod.params['tflex'], nflex=mod.params['nflex'], nfixed=mod.params['nfixed'])
            allagebins[iteration, :] = agebins
            dt = 10**agebins[:, 1] - 10**agebins[:, 0]
            masses = logsfr_ratios_to_masses_psb(logsfr_ratios=logr, logmass=logmass, agebins=agebins,
                logsfr_ratio_young=logr_young, logsfr_ratio_old=logr_old,
                tlast=tlast, tflex=mod.params['tflex'], nflex=mod.params['nflex'], nfixed=mod.params['nfixed']) #issues
            allsfrs[iteration,:] = (masses  / dt)

            logzsol = flatchain[iteration, mod.theta_index["logzsol"]] # LOGZSOL COLLECTED HERE
            dust2 = flatchain[iteration, mod.theta_index["dust2"]] # DUST2 COLLECTED HERE
            dust_index = flatchain[iteration, mod.theta_index["dust_index"]] #DUST INDEX COLLECTED HERE
            #zred = flatchain[iteration,mod.theta_index['zred']] # ZRED COLLECTED HERE

        print("###############################################")
        print("logmass: " + str(logmass))
        print("logzsol: " + str(logzsol))
        print("dust2: " + str(dust2))
        print("dust_index: " + str(dust_index))
        
        
        allagebins_ago = 10**allagebins/(1e9) 
        allsfrs_interp = np.zeros((flatchain.shape[0], len(age_interp)))
        masscum_interp = np.zeros_like(allsfrs_interp)
        totmasscum_interp = np.zeros_like(allsfrs_interp)
        dt = (age_interp - np.insert(age_interp,0,0)[:-1]) * 1e9
        for i in range(flatchain.shape[0]):
            #tuniv = cosmo.age(flatchain[i,mod.theta_index['zred']]).value
            tuniv = cosmo.age(3.0015846275124947).value
            allsfrs_interp[i,:] = stepInterp(tuniv - allagebins_ago[i,:], allsfrs[i,:], age_interp) #age_interp in age now, allagebins_ago in age #age = t_univ(z) - lookback time
            allsfrs_interp[i,-1] = 0
            masscum_interp[i,:] = 1 - (np.cumsum(allsfrs_interp[i,:] * dt) / np.sum(allsfrs_interp[i,:] * dt))
            totmasscum_interp[i,:] = np.sum(allsfrs_interp[i,:] * dt) - (np.cumsum(allsfrs_interp[i,:] * dt))
        
        # sfr and cumulative mass percentiles 
        sfrPercent = np.array([quantile(allsfrs_interp[:,i], [16,50,84], weights=res.get('weights', None)) 
            for i in range(allsfrs_interp.shape[1])])
        sfrPercent = np.concatenate((age_interp[:,np.newaxis], sfrPercent), axis=1) # add time # NEEDS REDSHIFT IN AGE_INTERP

        # I doubt this is accurate...
        logzsol_array = quantile(logzsol, [16,50,84], weights=res.get("weights", None))
        logmass_array = quantile(logmass, [16,50,84], weights=res.get("weights", None))

        # SFR - error bars + last bin 
        sfhadjusted = np.interp(cosmo.age(gal['sfh'][:,0]).value, age_interp, sfrPercent[:,2])
        sfhadjusted_lower = np.interp(cosmo.age(gal['sfh'][:,0]).value, age_interp, sfrPercent[:,1])
        sfhadjusted_upper = np.interp(cosmo.age(gal['sfh'][:,0]).value, age_interp, sfrPercent[:,3])
        x = -1
        while np.isnan(sfhadjusted[x]): #some of the last values for the output may be nan, in this case, take the last not-nan bin
            x -= 1
        
        print("obs logM: " + str(obs['logM']))
        print("um sfh integral mass: " + str(np.log10(trap(cosmo.age(gal['sfh'][:,0]).value*1e9, um_sfh))))
        print("output original:" + str(logmass_array[1]))
        print("output integral mass: " + str(np.log10(trap(cosmo.age(gal['sfh'][:,0]).value*1e9, sfhadjusted))))
        outputIntegralMass = np.log10(trap(cosmo.age(gal['sfh'][:,0]).value*1e9, sfhadjusted))

        #LOGMASS
        ax[0,0].errorbar(obs['logM'],outputIntegralMass,yerr=np.vstack((logmass_array[1]-logmass_array[0],logmass_array[2]-logmass_array[1])),marker='.', markersize=10, ls='', lw=2, 
            markerfacecolor='navy',markeredgecolor='navy',markeredgewidth=3,ecolor='navy',elinewidth=1.4) 
        #ax[0,0].plot(obs['massforgal'],logmass_array[1],marker='.', markersize=10, ls='', lw=2, markerfacecolor='navy',markeredgecolor='navy',markeredgewidth=3)

        #SFRs
        ax[0,1].errorbar(um_sfh[x], sfhadjusted[x], yerr=np.vstack((sfhadjusted[x] - sfhadjusted_lower[x], sfhadjusted_upper[x]-sfhadjusted[x])),marker='.', markersize=10, ls='', lw=2, 
            markerfacecolor='dodgerblue',markeredgewidth=3,markeredgecolor='dodgerblue', ecolor='dodgerblue',elinewidth=1.4) 
        #cosmo.age(gal['sfh'][:,0]).value - plots with sfhadjusted
       
        #LOGZSOL
        ax[1,0].errorbar(spsdict['logzsol'],logzsol_array[1],yerr=np.vstack((logzsol_array[1]-logzsol_array[0],logzsol_array[2]-logzsol_array[1])),marker='.', markersize=10, ls='', lw=2, 
            markerfacecolor='darkorange',markeredgecolor='darkorange',markeredgewidth=3,ecolor='darkorange',elinewidth=1.4) 
        #ax[1,0].errorbar(spsdict['logzsol'],logzsol[1],marker='.', markersize=10, ls='', lw=2, markerfacecolor='darkorange',markeredgecolor='darkorange',markeredgewidth=3) 

        '''
        #ZRED
        zred_df = pd.concat([zred_df, pd.DataFrame([{'data':zred[0]}])])
        
        #DUST2
        dust2_df = pd.concat([dust2_df,pd.DataFrame([{'data':dust2[0]}])])

        #DUST_INDEX
        dust_index_df = pd.concat([dust_index_df,pd.DataFrame([{'data':dust_index[0]}])])
        '''
        dust2_array.append(dust2[0])
        dust_index_array.append(dust_index[0])
        #zred_array.append(zred[0])


         
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
ax[2,0].axvline(0, ls='--',color='black', lw=2, label='Input dust2: 0.0')
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
ax[0,1].set_xscale('log')
ax[0,1].set_yscale('log')
ax[0,1].set_xlabel(r'Input SFR ($M_{sun}/yr$)')
ax[0,1].set_ylabel(r'Recovered SFR ($M_{sun}/yr$)')

# LOGZSOL - scatter
ax[1,0].axline((0, 0), slope=1., ls='--', color='black', lw=2)
ax[1,0].set_xlim(-0.35,0.5)
ax[1,0].set_ylim(-2,0.3)
ax[1,0].set_xlabel(r'Input $log(Z/Z_{\odot})$')
ax[1,0].set_ylabel(r'Recovered $log(Z/Z_{\odot})$')

plt.tight_layout()
plt.show()

# save plot 
counter=0
filename = 'bigplot_z3_fixedzred_{}.pdf' #defines filename for all objects
while os.path.isfile(plotdir+filename.format(counter)):
    counter += 1
filename = filename.format(counter) #iterate until a unique file is made
fig.savefig(plotdir+filename, bbox_inches='tight')
  
print('saved big plot to '+plotdir+filename) 

#plt.close(fig)


