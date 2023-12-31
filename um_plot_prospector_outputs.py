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
from scipy.optimize import minimize
from scipy.stats import chisquare
from scipy.optimize import fsolve
import dynesty
import h5py
from matplotlib import pyplot as plt, ticker as ticker
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
import um_cornerplot
from prospect.models.transforms import logsfr_ratios_to_masses
import sys
from scipy.interpolate import interp1d
from scipy.misc import derivative
import os
from prospect.models import priors, SedModel
from prospect.io.read_results import results_from, get_sps
from prospect.io.read_results import traceplot, subcorner
from wren_functions import modified_logsfr_ratios_to_agebins, modified_logsfr_ratios_to_masses_flex
import fsps
import math

# set up cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=.3)

# read in a command-line argument for which output to plot where
if len(sys.argv) > 0:
    outroot = sys.argv[1]
else:
    outroot = 'squiggle_1680728723_mcmc.h5'
plotdir = 'plots/'
print('Making plots for '+outroot)

# functions to make sure we interpret the results correctly....
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
        newval[(ts >= ab[i,0]) & (ts < ab[i,1])] = val[i]  
    return newval    

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
#Edited for new PSB model: youngest bin is 'tquench' long, and it is preceded by 'nflex' young flexible bins, then 'nfixed' older fixed bins
'''
def priors(tflex=tflex, nflex=nflex, nfixed=nfixed, tquench=0.2):
        #set smarter priors on the logSFR ratios. given a 
        redshift zred, use the closest-z universe machine SFH
        to set a better guess than a constant SFR. returns
        agebins, logmass, and set of logSFR ratios that are
        self-consistent. 

        # define default agebins
        agelims = np.array([1, tquench*1e9] + \
            np.linspace((tquench+.1)*1e9, (tflex)*1e9, nflex).tolist() \
            + np.linspace(tflex*1e9, tuniv[0]*1e9, nfixed+1)[1:].tolist())
        
        agebins_ml = np.array([np.log10(agelims[:-1]), np.log10(agelims[1:])]).T 

        # subsample to make sure we'll have umachine bin for everything
        newages = np.arange(0, tuniv, 0.001)
        sfh = np.ones_like(newages) / len(newages)
        mass = sfh*0.001*1e9 # mass formed in each little bin    

        # get default agebins in same units
        abins_age = 10**agebins_ml/1e9 # in Gyr

        # get mass in each bin
        myoung = np.sum(mass[(newages >= abins_age[0,0]) & (newages <= abins_age[0,1])])
        mold = []
        for i in range(nflex+1, len(abins_age)):
            mold.append(np.sum(mass[(newages >= abins_age[i,0]) & (newages <= abins_age[i,1])]))
        mflex = np.sum(mass[(newages >= abins_age[1,0]) & (newages <= abins_age[nflex,1])])    

        # adjust agebins according to mflex
        # each of the nflex flex bins should have (mflex / nflex) mass
        idx = (newages >= abins_age[1,0]) & (newages <= abins_age[nflex,1]) # part of flex range

        agelims_flex = []
        for i in range(nflex):
            agelims_flex.append(np.interp(mflex/nflex*i, np.cumsum(mass[idx]), newages[idx]))
        abins_age[1:nflex, 1] = agelims_flex[1:]
        abins_age[2:nflex+1, 0] = agelims_flex[1:]

        # remake agebins
        agebins_ml = np.log10(1e9*abins_age)

        # now get the sfr in each bin
        sfrs = np.zeros((len(abins_age)))
        for i in range(len(sfrs)):
            # relevant umachine ages
            idx = (newages >= abins_age[i,0]) & (newages <= abins_age[i,1])
            sfrs[i] = trap(newages[idx], sfh[idx]) / (abins_age[i,1] - abins_age[i,0])

        # young is easy
        logsfr_ratio_young = np.log10(sfrs[0] / sfrs[1])

        # old is w ref to the oldest flex bin
        logsfr_ratio_old = np.ones(nfixed)
        for i in range(nfixed):
            logsfr_ratio_old[i] = sfrs[nflex+i] / sfrs[nflex+i+1]
        logsfr_ratio_old = np.log10(logsfr_ratio_old)    

        # and finally the flex bins
        logsfr_ratios = np.ones(nflex-1)
        for i in range(nflex-1):
            logsfr_ratios[i] = sfrs[i+1] / sfrs[i+2] 
        logsfr_ratios = np.log10(logsfr_ratios)
        
        return logsfr_ratio_young, logsfr_ratios, logsfr_ratio_old

def psb(ages, logsfr_ratios, tquench, logsfr_ratio_young, logsfr_ratio_old, mtot, tflex=tflex, nflex=nflex, nfixed=nfixed):

    # define initial agebins
    agelims = np.array([1, tquench*1e9] + \
        np.linspace((tquench+.1)*1e9, (tflex)*1e9, nflex).tolist() \
        + np.linspace(tflex*1e9, tuniv[0]*1e9, nfixed+1)[1:].tolist())
    agebins = np.array([np.log10(agelims[:-1]), np.log10(agelims[1:])]).T 

    agebins = modified_logsfr_ratios_to_agebins(logsfr_ratios=logsfr_ratios, agebins=agebins, 
        tquench=tquench, tflex=tflex, nflex=nflex, nfixed=nfixed) 
    
    dt = 10**agebins[:, 1] - 10**agebins[:, 0]
    masses = modified_logsfr_ratios_to_masses_flex(logsfr_ratios=logsfr_ratios, logmass=np.log10(mtot), agebins=agebins,
        logsfr_ratio_young=logsfr_ratio_young, logsfr_ratio_old=logsfr_ratio_old,
        tquench=tquench, tflex=tflex, nflex=nflex, nfixed=nfixed) 
    
    sfrs = (masses  / dt)

    # interpolate SFR to time array
    agebins_ago = 10**agebins/1e9 
    age_interp = ages
    dt = (age_interp - np.insert(age_interp,0,0)[:-1]) * 1e9

    #print('test', age_interp, agebins_ago)
    sfh = opp_stepInterp(agebins_ago, sfrs, age_interp) #CHANGED FIRST PARAM #age_interp in age now, allagebins_ago in age #age = t_univ(z) - lookback time
    sfh[-1] = 0.
    sfh[np.isnan(sfh)] = 0.
    
    # and normalize it so that the total stellar mass formed is mtot
    sfh = sfh * (mtot / trap(ages*1e9, sfh)) # ok, now we're in Msun/yr units #pretty much multiply by 1?

    return sfh, agebins_ago

def log_likelihood(theta, x, y, yerr):
        logr0, logr1, logr2, logr3, tquench, logsfr_ratio_young, logrold0, logrold1, logrold2, mtot = theta

        # group some of the variables
        logsfr_ratios = np.array([logr0, logr1, logr2, logr3])
        logsfr_ratio_old = np.array([logrold0, logrold1, logrold2])

        # run model
        model, _ = psb(x, logsfr_ratios, tquench, logsfr_ratio_young, logsfr_ratio_old, mtot)
        
        sigma2 = yerr**2
        test = -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

        return test

'''
  
##############################################################################  
    
# grab results (dictionary), the obs dictionary, and our corresponding models
res, obs, mod = results_from("{}".format(outroot), dangerous=True) # This displays the model parameters too
print("{}".format(outroot))
sps = get_sps(res)

#obs = (np.load('obs-z3/umobs_'+str(obs_mcmc['objid'])+'.npz', allow_pickle=True))['obs']
gal = (np.load('obs/umobs_'+str(obs['objid'])+'.npz', allow_pickle=True))['gal']
spsdict = (np.load('obs/umobs_'+str(obs['objid'])+'.npz', allow_pickle=True))['params'][()]


print('Object ID: ' + str(obs['objid']))

print('Loaded results')

# make sure plot directory exits
if not os.path.exists(plotdir):
    os.mkdir(plotdir)

##############################################################################  

# traceplot
tracefig = traceplot(res, figsize=(10,5))
# make sure directory exists
if not os.path.exists(plotdir+'trace'):
    os.mkdir(plotdir+'trace')

#check to see if duplicates exist
counter=0
filename = str(int(obs['objid'])) + '_{}.pdf' #defines filename for all objects
while os.path.isfile(plotdir+'trace/'+filename.format(counter)):
    counter += 1
filename = filename.format(counter) #iterate until a unique file is made

'''
plt.savefig(plotdir+'trace/'+filename, bbox_inches='tight')
print('saved tracefig to '+plotdir+'trace/'+filename)
plt.close()
'''

##############################################################################  
# corner plot
# maximum a posteriori (of the locations visited by the MCMC sampler)
# also put “truth” = input values for e.g. stellar mass, dust, metallicity, 
# all of the values we set when we made the mock galaxy, on that cornerplot so we can see how right/wrong the fit is 

# obtain gal from obs dictionary file
#load_values = np.load(res)
#gal = (np.load(res))['gal'] 

# Truth values
#['zred','logzsol','dust2','logmass','tlast','logsfr_ratio_young','logsfr_ratio_old_1','logsfr_ratio_old_2',
# 'logsfr_ratio_old_3','logsfr_ratios_1','logsfr_ratios_2','logsfr_ratios_3','logsfr_ratios_4','dust_index']

truth_array = [gal['z'], spsdict['logzsol'], spsdict['dust2'], obs['logM'], 0, 0, 0, 0, 0, 0, 0, 0, 0, spsdict['dust_index']]
imax = np.argmax(res['lnprobability'])
theta_max = res['chain'][imax, :].copy()

print('MAP value: {}'.format(theta_max))
fig, axes = plt.subplots(len(theta_max), len(theta_max), figsize=(15,15))
axes = um_cornerplot.allcorner(res['chain'].T, mod.theta_labels(), axes, show_titles=True, 
    span=[0.997]*len(mod.theta_labels()), weights=res.get("weights", None), 
    label_kwargs={"fontsize": 8}, tick_kwargs={"labelsize": 6}, title_kwargs={'fontsize':11}, truths=truth_array)

if not os.path.exists(plotdir+'corner'):
    os.mkdir(plotdir+'corner')    
fig.savefig(plotdir+'corner/'+filename, bbox_inches='tight')  
print('saved cornerplot to '+plotdir+filename)  
plt.close(fig)    
print('Made cornerplot')

##############################################################################  

# look at sed & residuals
# generate model at MAP value
mspec_map, mphot_map, _ = mod.mean_model(theta_max, obs, sps=sps) #usually in restframe, in maggies
# wavelength vectors
#a = 1.0 + mod.params.get('zred', 0.0) # cosmological redshifting # FIXED ZRED
a = 1.0 + spsdict['zred']

# photometric effective wavelengths
wphot = np.array(obs["wave_effective"]) #is this correct?
# spectroscopic wavelengths
if obs["wavelength"] is None:
    # *restframe* spectral wavelengths, since obs["wavelength"] is None
    wspec = sps.wavelengths.copy() #rest frame
    wspec *= a #redshift them to be observed frame
else:
    wspec = obs["wavelength"]
    
# get real 16/50/84% spectra
# only calculate from 1000 highest-weight samples
print('Starting to calculate spectra...')
weights = res.get('weights',None)
idx = np.argsort(weights)[-1000:]
allspec = np.zeros((2, len(mspec_map), len(idx)))
allphot = np.zeros((len(mphot_map), len(idx)))
allmfrac = np.zeros((len(idx)))

for ii, i in enumerate(idx):
    # wavelength obs = wavelength rest * (1+z), so this is observed wavelength
    if 'zred' in mod.theta_index:
        allspec[0,:,ii] = sps.wavelengths.copy() * (1+res['chain'][i, mod.theta_index['zred']]) # array of zred
    else:
        allspec[0,:,ii] = sps.wavelengths.copy() * (1+obs['zred']) # array of zred
    
    allspec[1,:,ii], allphot[:,ii], allmfrac[ii] = mod.mean_model(res['chain'][i,:], obs, sps=sps)
phot16 = np.array([quantile(allphot[i,:], 16, weights = weights[idx]) for i in range(allphot.shape[0])])
phot50 = np.array([quantile(allphot[i,:], 50, weights = weights[idx]) for i in range(allphot.shape[0])])
phot84 = np.array([quantile(allphot[i,:], 84, weights = weights[idx]) for i in range(allphot.shape[0])])
print('Done calculating spectra')

# Make plot of data and model
c = 2.99792458e18

# NORMAL PLOTTING
fig, ax = plt.subplots(3,1,figsize=(8,12))

# wphot + wspec both observed
'''
ax[0].errorbar(wphot, (phot50*c/(wphot/(1+obs['zred']))**2.), label='Model photometry',
               yerr = (phot84-phot16), marker='s', markersize=10, alpha=0.8, ls='', lw=3, 
               markerfacecolor='none', markeredgecolor='green', markeredgewidth=3)
ax[0].errorbar(wphot, obs['maggies'] * c/(wphot/(1+obs['zred']))**2, yerr=obs['maggies_unc']*c/wphot**2, 
         label='Observed photometry', ecolor='red', 
         marker='o', markersize=10, ls='', lw=3, alpha=0.8, 
         markerfacecolor='none', markeredgecolor='black', 
         markeredgewidth=3)
# MAP Model spectrum#
ax[0].plot(wspec, mspec_map * c/(wspec/(1+obs['zred']))**2., label='Model spectrum', lw=1.5, color='grey', alpha=0.7, zorder=10)
'''

###### PLOTS IN FLAM ###### 
def convertMaggiesToFlam(w, maggies):
    # converts maggies to f_lambda units
    # For OBS DICT photometries - use w as obs['wave_effective'] - observed wavelengths 
    c = 2.99792458e18 #AA/s
    flux_fnu = maggies * 10**-23 * 3631 # maggies to cgs fnu
    flux_flambda = flux_fnu * c/w**2 # v Fnu = lambda Flambda
    return flux_flambda

# wphot = wave_effective
ax[0].plot(wspec, convertMaggiesToFlam(wspec, mspec_map), label='MAP Model spectrum',
       lw=1.5, color='grey', alpha=0.7, zorder=10)    
ax[0].errorbar(wphot, convertMaggiesToFlam(wphot, phot50), label='Model photometry',
         yerr = convertMaggiesToFlam(wphot, phot84) - convertMaggiesToFlam(wphot,phot16),
         marker='s', markersize=10, alpha=0.8, ls='', lw=3, 
         markerfacecolor='none', markeredgecolor='green', 
         markeredgewidth=3)
ax[0].errorbar(wphot, convertMaggiesToFlam(wphot, obs['maggies']), yerr=convertMaggiesToFlam(wphot, obs['maggies_unc']), 
         label='Observed photometry', ecolor='red', 
         marker='o', markersize=10, ls='', lw=3, alpha=0.8, 
         markerfacecolor='none', markeredgecolor='black', 
         markeredgewidth=3)            
     
ax[0].set_xlim((1e3, 1e5))
ax[0].set_xlabel('Observed Wavelength (' + r'$\AA$' + ')', fontsize=10)
ax[0].set_ylabel(r"F$_\lambda$ in ergs/s/cm$^2$/AA", fontsize=10) # in flam units
ax[0].set_xscale('log')
ax[0].legend(loc='best', fontsize=9)
ax[0].set_title(str(int(obs['objid'])))
ax[0].tick_params(axis='both', which='major', labelsize=10)
print('Made spectrum plot')

######################## SFH for FLEXIBLE continuity model ########################
from prospect.models.transforms import logsfr_ratios_to_masses_psb, psb_logsfr_ratios_to_agebins

# from squiggle_flex_continuity import modified_logsfr_ratios_to_masses_flex, modified_logsfr_ratios_to_agebins

# obtain sfh from universemachine
um_sfh = gal['sfh'][:,1]

# actual sfh percentiles
flatchain = res["chain"]
niter = res['chain'].shape[-2]
if 'zred' in mod.theta_index:
    tmax = cosmo.age(np.min(flatchain[:,mod.theta_index['zred']])).value #matches scales
else: 
    tmax = cosmo.age(obs['zred']).value

# will need to interpolate to get everything on same time scale
# make sure this matches b/t two model types!
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

'''
print("generating histogram....")
plt.hist(temp_logmass_array, bins=10)
plt.xlabel("Recovered logmass")
plt.axvline(obs['logM'], ls='--',color='black', lw=2, label='Input logmass: {0:.3f}'.format(obs['logM']))
plt.show()
'''

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
print(percentiles) # prints percentiles

# mass fraction in the last Gyr
massFrac = 1 - massPercent[age_interp==1, 1:].flatten()[::-1]  

############## MAXIMUM LIKELIHOOD ESTIMATE ##################
'''
# Collect values for MLE, straight from UMachine...
sfr_obs = gal['sfh'][:,1]
sfr_obs_err = np.ones_like(sfr_obs) * 1e-3 * max(sfr_obs) 
t_obs = cosmo.age(gal['sfh'][:,0]).value
t_obs = tuniv - t_obs # convert t_obs (Gyr) to lookback time (log10(yr))

# remove any points that occur after obs redshift of galaxy
badidx = np.where(t_obs < 0.)[0]
t_obs = np.delete(t_obs, badidx)
sfr_obs = np.delete(sfr_obs, badidx)
sfr_obs_err = np.delete(sfr_obs_err, badidx)
t_obs = t_obs[::-1]
sfr_obs = sfr_obs[::-1]
sfr_obs_err = sfr_obs_err[::-1]

mtot_init = trap(t_obs*1e9, sfr_obs) 

nll = lambda *args: -log_likelihood(*args)
tquench_init = 0.2
logsfr_ratio_young_init, logsfr_ratios_init, logsfr_ratio_old_init = priors(tquench=tquench_init)
logr0_init, logr1_init, logr2_init, logr3_init = logsfr_ratios_init
logrold0_init, logrold1_init, logrold2_init = logsfr_ratio_old_init

initial = np.array([logr0_init, logr1_init, logr2_init, logr3_init, tquench_init, logsfr_ratio_young_init, logrold0_init, logrold1_init, logrold2_init, mtot_init])

soln = minimize(nll, initial, args=(t_obs, sfr_obs, sfr_obs_err))
sfr_ml, agebins_ml = psb(t_obs, soln.x[0:4], soln.x[4], soln.x[5], soln.x[6:9], mtot=soln.x[9]) 

t_plot = tuniv - t_obs
t_plot = t_plot[::-1]
'''

###### SFH ADJUSTED PLOTS ##############
# One-dimensional linear interpolation.
# age_interp is in LOOK BACK TIME 
# tage =  tmax - LBT 
# LBT = tmax - tage

# First convert age_interp to age
#sfhadjusted_lower = np.interp(cosmo.age(gal['sfh'][:,0]).value, age_interp, sfrPercent[:,1])
#sfhadjusted = np.interp(cosmo.age(gal['sfh'][:,0]).value, age_interp, sfrPercent[:,2])
#sfhadjusted_upper = np.interp(cosmo.age(gal['sfh'][:,0]).value, tmax - age_interp, sfrPercent[:,3])

######## SFH PLOTTING in LBT ##########
ax[1].fill_between(age_interp, sfrPercent[:,1], sfrPercent[:,3], color='grey', alpha=.5)
ax[1].plot(age_interp, sfrPercent[:,2], color='black', lw=1.5, label='Output SFH') #age_interp in Gyr

#ax[1].fill_between(cosmo.age(gal['sfh'][:,0]).value, sfhadjusted_lower, sfhadjusted_upper, color='grey', alpha=.5)
#ax[1].plot(cosmo.age(gal['sfh'][:,0]).value, sfhadjusted, '-o', label = 'Output SFH', color='black', lw=1) # OUTPUT SFH
# Convert x-axis from age to LBT
ax[1].plot(cosmo.age(obs['zred']).value - cosmo.age(gal['sfh'][:,0]).value, um_sfh, label='Input SFH', color='blue', lw=1, marker="o") # INPUT SFH

#label='Input SFH (z = {0:.3f})'.format(spsdict['zred'])
#label='Output SFH (z = {0:.3f})'.format(mod.params['zred'][0])
#ax[1].plot(t_plot, sfr_ml[::-1], 'g--', lw=2, label='Maximum Likelihood SFH') # MLE SFH
#ax[1].plot([], [], ' ', label='Input mass: {0:.3f}, MLE output mass: {1:.3f}'.format(np.log10(mtot_init), np.log10(soln.x[9]))) # MLE MASS ESTIMATE

ax[1].set_xlim(cosmo.age(gal['sfh'][:,0]).value[-1], 0)
ax[1].set_yscale('log')
ax[1].legend(loc='best', fontsize=9)
ax[1].tick_params(axis='both', which='major', labelsize=10)
ax[1].set_ylabel('SFR [' + r'$M_{\odot} /yr$' + ']', fontsize = 10)
#ax[1].set_xlabel('years before observation [Gyr]')
ax[1].set_xlabel('Lookback Time [Gyr]', fontsize = 10)

print('Finished SFH')

# plot sfh and percentiles
#ax[1].fill_between(age_interp, sfrPercent[:,1], sfrPercent[:,3], color='grey', alpha=.5)
#ax[1].plot(age_interp, sfrPercent[:,2], color='black', lw=1.5)
#ax[1].set_ylabel('SFR [Msun/yr]')
#ax[1].set_xlabel('years before observation [Gyr]')


# add secondary axis for redshift
#x_formatter = [1, 2, 3, 5, 7, 10, 15]
#x_locator = [5.75164694, 3.22662706, 2.11252719, 1.15475791, 0.75081398, 0.46588724, 0.26562898]
#secax = ax[1].twiny()
#secax.set_xticks(x_locator)
#secax.set_xticklabels(x_formatter)
#secax.tick_params(axis='x',which='both')
#secax.set_xlabel('Redshift [z]') 
 
######## Derivative for SFH ###########
# Eliminates 0 values from the SFHs, which can skew the derivative; limits quenchtime search for output
# SFH to only be within input SFH's range
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

# Plot derivative for input + output SFH, + quenching threshold from Wren's paper
# Plot vertical lines for the quench time on the SFH plot
if len(x_i) != 0:
    ax[2].plot(x_d_input, y_d_input, '-o', color='blue', lw=1.5, label='Input SFH time derivative (quench time: ' + str(list(map('{0:.3f}'.format, x_i[0])))[2:-2] + ' Gyr)')
    ax[1].axvline(x_i[0], linestyle='--', lw=1, color='blue')
    ax[2].axvline(x_i[0], linestyle='--', lw=1, color='blue')
else:
    ax[2].plot(x_d_input, y_d_input, '-o', color='blue', lw=1.5, label='Input SFH time derivative (does not pass quenching threshold)')

if len(x_o != 0):
    ax[2].plot(x_d_output, y_d_output, '-o', color='black', lw=1.5, label='Output SFH time derivative (quench time: ' + str(list(map('{0:.3f}'.format, x_o[0])))[2:-2] + ' Gyr)')
    ax[1].axvline(x_o[0], linestyle='--', lw=1, color='black')
    ax[2].axvline(x_o[0], linestyle='--', lw=1, color='black')
else: 
    ax[2].plot(x_d_output, y_d_output, '-o', color='black', lw=1.5, label='Output SFH time derivative (does not pass quenching threshold)')
ax[2].axhline(-500, linestyle='--', color='maroon', label='-500 ' + r'$M_{\odot} yr^{-2}$' + ' quenching threshold') # Quench threshold

ax[2].set_ylabel("SFH Time Derivative " + r'$[M_{\odot} yr^{-2}]$', fontsize=10)
ax[2].set_xlabel('Lookback Time [Gyr]', fontsize=10)
ax[2].set_xlim(cosmo.age(gal['sfh'][:,0]).value[-1], 0)
ax[2].set_ylim(-1000, 1000)
ax[2].legend(loc='best', fontsize=9)
ax[2].tick_params(axis='both', which='major', labelsize=10)

print('Finished derivative plot')

# cumulative mass fraction plot
#ax[2].fill_between(age_interp, massPercent[:,1], massPercent[:,3], color='grey', alpha=.5)
#ax[2].plot(age_interp, massPercent[:,2], color='black', lw=1.5)
#ax[2].set_xlim((tmax+.1,-.1))
#ax[2].set_ylabel('Cumulative mass fraction')
#ax[2].set_xlabel('years before observation [Gyr]')
plt.show()
# save plot
fig.tight_layout()
if not os.path.exists(plotdir+'sfh'):
    os.mkdir(plotdir+'sfh')    
fig.savefig(plotdir+'sfh/' + filename, bbox_inches='tight')
  
print('saved sfh to '+plotdir+'sfh/'+filename) 
plt.close(fig)
print('Made SFH plot')
'''
# and now we want to write out all of these outputs so we can have them for later!
# make a lil class that will just save all of the outputs we give it, so that it's easy to pack all these together later
class Output:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

# gather all the spectra (obs AND best-fit plus uncertainties)
# spec_obs = np.stack((wspec, obs['spectrum'], obs['unc'])) # maggies
# spec_fit = np.stack((wspec, spec16, spec50, spec84)) # maggies
phot_obs = np.stack((wphot, obs['maggies']))
phot_fit = np.stack((wphot, phot16, phot50, phot84))


# make an instance of the output structure for this dict
out = Output(phot_obs=phot_obs, phot_fit=phot_fit, sfrs=sfrPercent, mass=massPercent, 
    objname=str(obs['objid']), massfrac=massFrac, percentiles=percentiles, massTot=totmassPercent)

# and save it
if not os.path.exists('dicts/'):
    os.mkdir('dicts/')
np.savez('dicts/'+str(obs['objid'])+'.npz', res=out)
'''
