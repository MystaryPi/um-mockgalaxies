'''
Given two directories of MB mcmc files and no MB mcmc files, 
will iterate through each galaxy fit and plot different 
galaxy attributes as scatterplots/histograms. 

python mb-nomb-fit_test.py /path/to/mb/directory/ /path/to/nomb/directory/
'/Users/michpark/JWST_Programs/mockgalaxies/final/z2mb/' '/Users/michpark/JWST_Programs/mockgalaxies/final/z2nomb/'
'''
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
if len(sys.argv) > 0:
    mb_directory = str(sys.argv[1]) # example: '/Users/michpark/JWST_Programs/mockgalaxies/final/z3mb/'
    nomb_directory = str(sys.argv[2]) # example: '/Users/michpark/JWST_Programs/mockgalaxies/final/z3nomb/'
  
plotdir = '/Users/michpark/JWST_Programs/mockgalaxies/debug-april/colored-ave-sfr/'
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

from um_prospector_param_file import updated_logsfr_ratios_to_masses_psb, updated_psb_logsfr_ratios_to_agebins

directory_array = [mb_directory, nomb_directory]
zred_array = np.empty(shape=(2, len(os.listdir(mb_directory)))) 

for directory_index, directory in enumerate(directory_array):
    print("Current directory: " + str(directory)) # prints out directory we're currently iterating over
    
    first_iteration = True # sets up this boolean for labels
    mcmc_counter = 0
    # Iterate through mcmc files in the directory
    for mcmcfile in os.listdir(directory):
            mcmcfile = os.path.join(directory, mcmcfile)
            #print('Making plots for '+str(mcmcfile))

            res, obs, mod = results_from("{}".format(mcmcfile), dangerous=True)
            gal = (np.load('/Users/michpark/JWST_Programs/mockgalaxies/obs-z5/umobs_'+str(obs['objid'])+'.npz', allow_pickle=True))['gal']
            spsdict = (np.load('/Users/michpark/JWST_Programs/mockgalaxies/obs-z5/umobs_'+str(obs['objid'])+'.npz', allow_pickle=True))['params'][()]

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
                lbt_interp = np.concatenate((np.arange(0,2,.001),np.arange(2,tmax,.01),[tmax])) 
            else:
                lbt_interp = np.arange(0,tmax+.005, .001)    
            lbt_interp[0] = 1e-9    
            age_interp = tmax - lbt_interp
        
            nflex = mod.params['nflex']
            nfixed = mod.params['nfixed']
            tflex_frac = mod.params['tflex_frac']  

            # actual sfh percentiles
            allsfrs = np.zeros((flatchain.shape[0], len(mod.params['agebins'])))
            allagebins = np.zeros((flatchain.shape[0], len(mod.params['agebins']), 2))
            for iteration in range(flatchain.shape[0]):
                zred = flatchain[iteration, mod.theta_index['zred']]
                tuniv_thisdraw = cosmo.age(zred).value
                logr = np.clip(flatchain[iteration, mod.theta_index["logsfr_ratios"]], -7,7)
                tlast_fraction = flatchain[iteration, mod.theta_index['tlast_fraction']]
                logr_young = np.clip(flatchain[iteration, mod.theta_index['logsfr_ratio_young']], -7, 7)
                logr_old = np.clip(flatchain[iteration, mod.theta_index['logsfr_ratio_old']], -7, 7)
                try:
                    logmass = flatchain[iteration, mod.theta_index['massmet']][0] # not what we are using
                except:
                    logmass = flatchain[iteration, mod.theta_index["logmass"]]      
                agebins = updated_psb_logsfr_ratios_to_agebins(logsfr_ratios=logr, agebins=mod.params['agebins'], 
                    tlast_fraction=tlast_fraction, tflex_frac=tflex_frac, nflex=nflex, nfixed=nfixed, zred=zred)
                allagebins[iteration, :] = agebins
                dt = 10**agebins[:, 1] - 10**agebins[:, 0]
                masses = updated_logsfr_ratios_to_masses_psb(logsfr_ratios=logr, logmass=logmass, agebins=agebins,
                    logsfr_ratio_young=logr_young, logsfr_ratio_old=logr_old,
                    tlast_fraction=tlast_fraction, tflex_frac=tflex_frac, nflex=nflex, nfixed=nfixed, zred=zred)
                allsfrs[iteration,:] = (masses  / dt)

            # calculate interpolated SFR and cumulative mass  
            # with each likelihood draw you can convert the agebins from units of lookback time to units of age 
            # using the redshift at that likelihood draw, and put it on your fixed grid of ages
            allagebins_lbt = 10**allagebins/1e9  
            allagebins_age = np.zeros_like(allagebins_lbt) + np.nan
            allsfrs_interp = np.zeros((flatchain.shape[0], len(lbt_interp))) # this one is in LBT (x-axis = lbt_interp)
            allsfrs_interp_age = np.zeros((flatchain.shape[0], len(lbt_interp))) # this one is in age of universe (x-axis = age_interp)
            masscum_interp = np.zeros_like(allsfrs_interp)
            totmasscum_interp = np.zeros_like(allsfrs_interp)
            dt = (lbt_interp - np.insert(lbt_interp,0,0)[:-1]) * 1e9
            for i in range(flatchain.shape[0]):
                allsfrs_interp[i,:] = stepInterp(allagebins_lbt[i,:], allsfrs[i,:], lbt_interp)
                allsfrs_interp[i,-1] = 0
                masscum_interp[i,:] = 1 - (np.cumsum(allsfrs_interp[i,:] * dt) / np.nansum(allsfrs_interp[i,:] * dt))
                totmasscum_interp[i,:] = np.nansum(allsfrs_interp[i,:] * dt) - (np.cumsum(allsfrs_interp[i,:] * dt))

                # now: let's also calculate this in terms of age of universe, not just LBT
                tuniv_thisdraw = cosmo.age(flatchain[i,mod.theta_index['zred']][0]).value
                allagebins_age[i,:] = tuniv_thisdraw - allagebins_lbt[i,:]; allagebins_age[i, -1,-1] = 0
                # swap the order so that interp can deal with it
                allagebins_age[i,:] = allagebins_age[i,:][:,::-1]
                allsfrs_interp_age[i,:] = stepInterp(allagebins_age[i,:], allsfrs[i,:], age_interp)

            # sfr and cumulative mass percentiles 
            sfrPercent = np.array([quantile(allsfrs_interp[:,i], [16,50,84], weights=res.get('weights', None)) 
                    for i in range(allsfrs_interp.shape[1])])
            sfrPercent = np.concatenate((lbt_interp[:,np.newaxis], sfrPercent), axis=1) # add time  
            massPercent = np.array([quantile(masscum_interp[:,i], [16,50,84], weights=res.get('weights', None)) 
                for i in range(masscum_interp.shape[1])])    
            massPercent = np.concatenate((lbt_interp[:,np.newaxis], massPercent), axis=1) # add time          
            totmassPercent = np.array([quantile(totmasscum_interp[:,i], [16,50,84], weights=res.get('weights', None)) 
                for i in range(totmasscum_interp.shape[1])])    
            totmassPercent = np.concatenate((lbt_interp[:,np.newaxis], totmassPercent), axis=1) # add time
            # now SFR in terms of age as well
            sfrPercent_age = np.array([quantile(allsfrs_interp_age[:,i], [16,50,84], weights=res.get('weights', None)) 
                for i in range(allsfrs_interp_age.shape[1])])
            sfrPercent_age = np.concatenate((age_interp[:,np.newaxis], sfrPercent_age), axis=1) # add time  

            # all percentiles...
            percentiles = get_percentiles(res, mod) # stores 16 50 84 percentiles for dif parameters
            massFrac = 1 - massPercent[lbt_interp==1, 1:].flatten()[::-1]  

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
                
            inputAverageSFR = averageSFR(cosmo.age(obs['zred']).value - cosmo.age(gal['sfh'][:,0]).value, um_sfh, timescale=0.1)
            outputAverageSFR_LE = averageSFR(lbt_interp, sfrPercent[:,1], timescale=0.1)
            outputAverageSFR = averageSFR(lbt_interp, sfrPercent[:,2], timescale=0.1)
            outputAverageSFR_UE = averageSFR(lbt_interp, sfrPercent[:,3], timescale=0.1)
            
            print("For " + str(obs['objid']) + ", we have input INST: " + str(um_sfh[-1]) + ", input AVE: " + str(inputAverageSFR) + ", outputINST: " + str(sfrPercent[:,2][0]) + ", output AVE: " + str(outputAverageSFR) + " with errors: " + str(outputAverageSFR_LE) + " and " + str(outputAverageSFR_UE))
            
            # input ave, output ave le, output ave, output ave ue
            zred_array[directory_index][mcmc_counter] = percentiles['zred'][1]-spsdict['zred']             
            if(directory_index == 0): # MB
                if(first_iteration):
                    results_mb = np.array([inputAverageSFR, outputAverageSFR_LE, outputAverageSFR, outputAverageSFR_UE]).flatten()
                else:
                    results_mb = np.vstack([results_mb, np.array([inputAverageSFR, outputAverageSFR_LE, outputAverageSFR, outputAverageSFR_UE]).flatten()])  
            if(directory_index == 1): # No MB
                if(first_iteration):
                    results_nomb = np.array([inputAverageSFR, outputAverageSFR_LE, outputAverageSFR, outputAverageSFR_UE]).flatten()
                else:
                    results_nomb = np.vstack([results_nomb, np.array([inputAverageSFR, outputAverageSFR_LE, outputAverageSFR, outputAverageSFR_UE]).flatten()])  
            first_iteration= False
            mcmc_counter += 1
            
# input vs. recovered t50
# input vs. recovered t95
# input vs. recovered t95-t50
# input my quench vs t95
# recovered my quench vs. t95
fig,ax = plt.subplots(2, 1, figsize=(8,8))
import matplotlib.colors as colors
from matplotlib import cm
divnorm = colors.TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.2)
counter=0
while counter < 2:
    if(counter == 0): 
        ax[counter].scatter(results_mb[:,0], results_mb[:,2], c=zred_array[0], norm=divnorm, cmap='bwr',ec='k')
        ax[counter].errorbar(results_mb[:,0], results_mb[:,2], yerr=np.vstack((results_mb[:,2]-results_mb[:,1], results_mb[:,3]-results_mb[:,2])), ecolor = 'gray', zorder=0, ls='none',elinewidth=.7)
    else: 
        ax[counter].scatter(results_nomb[:,0], results_nomb[:,2], c=zred_array[1], norm=divnorm, cmap='bwr',ec='k')
        ax[counter].errorbar(results_nomb[:,0], results_nomb[:,2], yerr=np.vstack((results_nomb[:,2]-results_nomb[:,1], results_nomb[:,3]-results_nomb[:,2])), ecolor = 'gray', zorder=0,ls='none',elinewidth=.7)
    
    ax[counter].axline((0, 0), slope=1., ls='--', color='black', lw=2,zorder=0)
    ax[counter].set_ylabel(r'Recovered $log SFR_{ave, 100 Myr}$ (log $M_{sun}$ / yr)')
    ax[counter].set_xlabel(r'Input $log SFR_{ave, 100 Myr}$ (log $M_{sun}$ / yr)')
    ax[counter].set_xscale('log')
    ax[counter].set_yscale('log')
    counter += 1
    
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
filename = '{}_z1_colored-ave-sfr.pdf' #defines filename for all objects
while os.path.isfile(plotdir+filename.format(counter)):
    counter += 1
filename = filename.format(counter) #iterate until a unique file is made
fig.savefig(plotdir+filename, bbox_inches='tight')
  
print('saved colored average SFRs to '+plotdir+filename) 

#plt.close(fig)


        

        



             
