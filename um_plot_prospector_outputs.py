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

def trap(x, y):
        return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]))/2. 

# set some params
# tflex=2 
# nflex=5
# nfixed=3
# Edited for new PSB model: youngest bin is 'tquench' long, and it is preceded by 'nflex' young flexible bins, then 'nfixed' older fixed bins

##############################################################################  
'''
res -- results dictionary
obs -- contains filters, photometry, logmass, redshift
mod -- model parameters (free/fixed)
gal -- contains ID, mass, sfr, redshift, sfh
spsdict -- contains truth logzsol, dust2, dust index
'''
res, obs, mod = results_from("{}".format(outroot), dangerous=True) 
print("{}".format(outroot))
sps = get_sps(res)
gal = (np.load('/oak/stanford/orgs/kipac/users/michpark/JWST_Programs/mockgalaxies/obs-z3/umobs_'+str(obs['objid'])+'.npz', allow_pickle=True))['gal']
spsdict = (np.load('/oak/stanford/orgs/kipac/users/michpark/JWST_Programs/mockgalaxies/obs-z3/umobs_'+str(obs['objid'])+'.npz', allow_pickle=True))['params'][()]
#gal = (np.load('/Users/michpark/JWST_Programs/mockgalaxies/obs-z3/umobs_'+str(obs['objid'])+'.npz', allow_pickle=True))['gal']
#spsdict = (np.load('/Users/michpark/JWST_Programs/mockgalaxies/obs-z3/umobs_'+str(obs['objid'])+'.npz', allow_pickle=True))['params'][()]

print('Object ID: ' + str(obs['objid']))

print('Loaded results')

# make sure plot directory exits
if not os.path.exists(plotdir):
    os.mkdir(plotdir)

##############################################################################  
# TRACEPLOT
tracefig = traceplot(res, figsize=(10,5))
# make sure directory exists
'''
if not os.path.exists(plotdir+'trace'):
    os.mkdir(plotdir+'trace')

#check to see if duplicates exist
counter=0
filename = str(int(obs['objid'])) + '_{}.pdf' #defines filename for all objects
while os.path.isfile(plotdir+'trace/'+filename.format(counter)):
    counter += 1
filename = filename.format(counter) #iterate until a unique file is made
'''
# Save the traceplot
'''
plt.savefig(plotdir+'trace/'+filename, bbox_inches='tight')
print('saved tracefig to '+plotdir+'trace/'+filename)
plt.close()
'''
##############################################################################  
# CORNER PLOT
'''
Maximum a posteriori (of the locations visited by the MCMC sampler)
"Truth array" holds input values for e.g. stellar mass, dust, metallicity, etc. 
ORDER of galaxy parameters: 
['zred','logzsol','dust2','logmass','tlast','logsfr_ratio_young',
'logsfr_ratio_old_1','logsfr_ratio_old_2', 'logsfr_ratio_old_3',
'logsfr_ratios_1','logsfr_ratios_2','logsfr_ratios_3','logsfr_ratios_4','dust_index']
'''

# Truth values
truth_array = [gal['z'], spsdict['logzsol'], spsdict['dust2'], obs['logM'], 0, 0, 0, 0, 0, 0, 0, 0, 0, spsdict['dust_index']]
imax = np.argmax(res['lnprobability'])
#imax = res['lnprobability'].argsort()[-2] # finds the ith most likely value
theta_max = res['chain'][imax, :].copy()

print('MAP value: {}'.format(theta_max))
#fig, axes = plt.subplots(len(theta_max), len(theta_max), figsize=(15,15))

# Zoomed in cornerplots to the log SFR ratios
'''
fig, axes = plt.subplots(len(theta_max[3:8]), len(theta_max[3:8]), figsize=(10,10))
axes = um_cornerplot.allcorner(res['chain'].T[3:8], mod.theta_labels()[3:8], axes, show_titles=True, 
    span=[0.9]*len(mod.theta_labels()[3:8]), weights=res.get("weights", None), 
    label_kwargs={"fontsize": 8}, tick_kwargs={"labelsize": 6}, title_kwargs={'fontsize':11}, truths=truth_array[3:8])

# label axes
for diag in [0,1,2,3,4]:
    start, end = axes[diag, diag].get_xlim()
    axes[diag,diag].xaxis.set_ticks(np.arange(start, end, np.abs(start-end)/3), fontsize=6)
    axes[diag,diag].xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
'''
# save cornerplot
'''
if not os.path.exists(plotdir+'corner'):
    os.mkdir(plotdir+'corner')    
fig.savefig(plotdir+'corner/'+filename, bbox_inches='tight')  
print('saved cornerplot to '+plotdir+filename)  
#plt.show()
#plt.close(fig)
'''
    
print('Made cornerplot')
##############################################################################  
# SPECTRA (SED)
# generate model at MAP value
mspec_map, mphot_map, _ = mod.mean_model(theta_max, obs, sps=sps) # restframe, in maggies
# wavelength vectors
#a = 1.0 + mod.params.get('zred', 0.0) # cosmological redshifting # FIXED ZRED
a = 1.0 + obs['zred']

# photometric effective wavelengths
wphot = np.array(obs["wave_effective"]) 
# spectroscopic wavelengths
if obs["wavelength"] is None:
    # *restframe* spectral wavelengths, since obs["wavelength"] is None
    wspec = sps.wavelengths.copy() #rest frame
    wspec *= a #redshift them to observed frame
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

# median spectrum
spec50 = np.array([quantile(allspec[1,i,:], 50, weights = weights[idx]) for i in range(allspec.shape[1])])
    
print('Done calculating spectra')

# Make plot of data and model
c = 2.99792458e18

# NORMAL PLOTTING
#fig, ax = plt.subplots(3,1,figsize=(8,14))

################ PLOT SPECTRA IN FLAM ################
def convertMaggiesToFlam(w, maggies):
    # converts maggies to f_lambda units
    # For OBS DICT photometries - use w as obs['wave_effective'] - observed wavelengths 
    c = 2.99792458e18 #AA/s
    flux_fnu = maggies * 10**-23 * 3631 # maggies to cgs fnu
    flux_flambda = flux_fnu * c/w**2 # v Fnu = lambda Flambda
    return flux_flambda

# MAP spectrum (plotting mspec_map)
# wphot = wave_effective
#ax[0].plot(wspec, convertMaggiesToFlam(wspec, mspec_map), label='MAP Model spectrum',
#       lw=1.5, color='grey', alpha=0.7, zorder=10)    

# Median spectrum (plotting spec50)
norm_wl = ((wspec>6300) & (wspec<6500))
norm = np.nanmax(convertMaggiesToFlam(wphot, obs['maggies'])[obs['phot_mask']])

'''
ax[0].plot(wspec, convertMaggiesToFlam(wspec, spec50), label='Median spectrum',
                   lw=1.5, color='grey', alpha=0.7, zorder=10) 
ax[0].errorbar(wphot[obs['phot_mask']], convertMaggiesToFlam(wphot, phot50)[obs['phot_mask']], label='Model photometry',
         yerr = (convertMaggiesToFlam(wphot, phot84) - convertMaggiesToFlam(wphot,phot16))[obs['phot_mask']],
         marker='s', markersize=10, alpha=0.8, ls='', lw=3, 
         markerfacecolor='none', markeredgecolor='green', 
         markeredgewidth=3)
ax[0].errorbar(wphot[obs['phot_mask']], convertMaggiesToFlam(wphot, obs['maggies'])[obs['phot_mask']], yerr=(convertMaggiesToFlam(wphot, obs['maggies_unc']))[obs['phot_mask']], 
         label='Observed photometry', ecolor='red', 
         marker='o', markersize=10, ls='', lw=3, alpha=0.8, 
         markerfacecolor='none', markeredgecolor='black', 
         markeredgewidth=3)            
ax[0].set_ylim((-0.2*norm, norm*2)) #top=1.5e-19 roughly
ax[0].set_xlim((1e3, 1e5))
ax[0].set_xlabel('Observed Wavelength (' + r'$\AA$' + ')', fontsize=10)
ax[0].set_ylabel(r"F$_\lambda$ in ergs/s/cm$^2$/AA", fontsize=10) # in flam units
ax[0].set_xscale('log')
ax[0].legend(loc='best', fontsize=9)
ax[0].set_title(str(int(obs['objid'])))
ax[0].tick_params(axis='both', which='major', labelsize=10)
print('Made spectrum plot')
'''

######################## SFH for FLEXIBLE continuity model ########################
from um_prospector_param_file import updated_logsfr_ratios_to_masses_psb, updated_psb_logsfr_ratios_to_agebins

# obtain sfh from universemachine
um_sfh = gal['sfh'][:,1]

# actual sfh percentiles
flatchain = res["chain"]
niter = res['chain'].shape[-2]
if 'zred' in mod.theta_index:
    tmax = cosmo.age(np.min(flatchain[:,mod.theta_index['zred']])).value #matches scales
else: 
    tmax = cosmo.age(obs['zred']).value

# interpolate to get everything on same time scale
if tmax > 2:
    lbt_interp = np.concatenate((np.arange(0,2,.001),np.arange(2,tmax,.01),[tmax])) 
    # lookback time 
else:
    lbt_interp = np.arange(0,tmax+.005, .001)    
lbt_interp[0] = 1e-9    
age_interp = tmax - lbt_interp # age of the universe

# nflex, nfixed, and tflex_frac are the same for all draws, so we can grab them here:
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
        
    # See which ratio corresponds with which SFR bin
    #logr = np.array([0,0,0,0])
    #logr_young = np.array([0])
    #logr_old = np.array([1,0,0])
    
    # MAP SFH 
    #logr_young = np.array([theta_max[4]])
    #logr_old = theta_max[5:8]
    #logr = theta_max[8:12]
    
    agebins = updated_psb_logsfr_ratios_to_agebins(logsfr_ratios=logr, agebins=mod.params['agebins'], 
        tlast_fraction=tlast_fraction, tflex_frac=tflex_frac, nflex=nflex, nfixed=nfixed, zred=zred)
    allagebins[iteration, :] = agebins
    dt = 10**agebins[:, 1] - 10**agebins[:, 0]
    
    masses = updated_logsfr_ratios_to_masses_psb(logsfr_ratios=logr, logmass=logmass, agebins=agebins,
        logsfr_ratio_young=logr_young, logsfr_ratio_old=logr_old,
        tlast_fraction=tlast_fraction, tflex_frac=tflex_frac, nflex=nflex, nfixed=nfixed, zred=zred)
    allsfrs[iteration,:] = (masses  / dt)

'''
Calculate interpolated SFR and cumulative mass  
with each likelihood draw you can convert the agebins from units of lookback time to units of age 
using the redshift at that likelihood draw, and put it on your fixed grid of ages  
'''
allagebins_lbt = 10**allagebins/1e9  # LBT
allagebins_age = np.zeros_like(allagebins_lbt) + np.nan # populate with AGE
allsfrs_interp = np.zeros((flatchain.shape[0], len(lbt_interp))) # LBT (x-axis = lbt_interp)
masscum_interp = np.zeros_like(allsfrs_interp)
totmasscum_interp = np.zeros_like(allsfrs_interp)
dt = ((lbt_interp - np.insert(lbt_interp,0,0)[:-1])) * 1e9
for i in range(flatchain.shape[0]):
    # interpolate with LBT
    allsfrs_interp[i,:] = stepInterp(allagebins_lbt[i,:], allsfrs[i,:], lbt_interp)
    allsfrs_interp[i,-1] = 0
    
    # Calculate cumulative mass fraction by summing over SFRs
    masscum_interp[i,:] = 1 - (np.cumsum(allsfrs_interp[i,:] * dt) / np.nansum(allsfrs_interp[i,:] * dt))
    totmasscum_interp[i,:] = np.nansum(allsfrs_interp[i,:] * dt) - (np.cumsum(allsfrs_interp[i,:] * dt))
    
# sfr and cumulative mass percentiles in LBT
sfrPercent = np.array([quantile(allsfrs_interp[:,i], [16,50,84], weights=res.get('weights', None)) 
    for i in range(allsfrs_interp.shape[1])])
sfrPercent = np.concatenate((lbt_interp[:,np.newaxis], sfrPercent), axis=1) # add time  
massPercent = np.array([quantile(masscum_interp[:,i], [16,50,84], weights=res.get('weights', None)) 
    for i in range(masscum_interp.shape[1])])    
massPercent = np.concatenate((lbt_interp[:,np.newaxis], massPercent), axis=1) # add time       
totmassPercent = np.array([quantile(totmasscum_interp[:,i], [16,50,84], weights=res.get('weights', None)) 
    for i in range(totmasscum_interp.shape[1])])    
totmassPercent = np.concatenate((lbt_interp[:,np.newaxis], totmassPercent), axis=1) # add time

# all percentiles...
percentiles = get_percentiles(res, mod) # stores 16 50 84 percentiles for dif parameters
print(percentiles) 

# mass fraction in the last Gyr 
massFrac = 1 - massPercent[lbt_interp==1, 1:].flatten()[::-1]  

# StudentT vs. prior samples HISTOGRAM
'''
plt.interactive(True)
plt.figure()
dist = priors.StudentT(mean=0, scale=0.3, df=1) # same for logr + logr old
samples = np.squeeze(np.array([dist.sample() for i in range(10000)]))
plt.hist(samples, bins=np.arange(-5, 5, 0.2), color='blue', alpha=0.5, label='StudentT Dist.', density=True)
plotindexold = 4+np.argmin(res['chain'][i, mod.theta_index['logsfr_ratio_old']]) + 1
plt.hist(dynesty.utils.resample_equal(res['chain'], weights=res.get("weights", None))[:,plotindexold], bins=np.arange(-5,5,0.2), color='orange', alpha=0.5, label='Prior Samples', density=True)
plt.title("logsfr_ratio_old_" + str(np.argmin(res['chain'][i, mod.theta_index['logsfr_ratio_old']]) + 1))
plt.legend()

plt.figure()
plt.hist(samples, bins=np.arange(-5, 5, 0.2), color='blue', alpha=0.5, label='StudentT Dist.', density=True)
plotindex = 7+np.argmin(res['chain'][i, mod.theta_index['logsfr_ratios']]) + 1
plt.hist(dynesty.utils.resample_equal(res['chain'], weights=res.get("weights", None))[:,plotindex], bins=np.arange(-5,5,0.2), color='orange', alpha=0.5, label='Prior Samples', density=True)
plt.title("logsfr_ratio_" + str(np.argmin(res['chain'][i, mod.theta_index['logsfr_ratios']]) + 1))
plt.legend()
'''

################ SFH PLOTTING in LBT ################
'''
# OUTPUT SFH
ax[1].fill_between(lbt_interp, sfrPercent[:,1], sfrPercent[:,3], color='grey', alpha=.5)
ax[1].plot(lbt_interp, sfrPercent[:,2], color='black', lw=1.5, label='Output SFH (z = {0:.3f})'.format(mod.params['zred'][0])) 

# INPUT SFH
# Convert x-axis from age to LBT (tuniv - age)
ax[1].plot(cosmo.age(obs['zred']).value - cosmo.age(gal['sfh'][:,0]).value, um_sfh, label='Input SFH (z = {0:.3f})'.format(spsdict['zred']), color='blue', lw=1, marker="o") 
ax[1].axvline(tflex_frac*cosmo.age(obs['zred']).value, color='orange', label="tflex fraction")
ax[1].set_xlim(cosmo.age(gal['sfh'][:,0]).value[-1], 0)
ax[1].set_yscale('log')
ax[1].legend(loc='best', fontsize=9)
ax[1].tick_params(axis='both', which='major', labelsize=10)
ax[1].set_ylabel('SFR [' + r'$M_{\odot} /yr$' + ']', fontsize = 10)
ax[1].set_xlabel('Lookback Time [Gyr]', fontsize = 10)
print('Finished SFH')
'''
 
################ CUMULATIVE MASS FRACTION ################
# Square interpolation - SFR(t1) and SFR(t2) are two snapshots, then for t<(t1+t2)/2 you assume SFR=SFR(t1) and t>(t1+t2)/2 you assume SFR=SFR(t2)
input_massFracSFR = np.array([])
trapsfh = um_sfh
traplbt = (cosmo.age(obs['zred']).value - cosmo.age(gal['sfh'][:,0]).value)
# step through SFH, accumulating mass following square interpolation
for n in range(len(trapsfh)-1):
    traplbtprep = np.array([traplbt[n], traplbt[n+1]])
    trapsfhprep = np.array([trapsfh[n], trapsfh[n+1]])
    if(len(input_massFracSFR) == 0): # accumulate mass
        input_massFracSFR = np.append(input_massFracSFR, trap(traplbtprep*10**9,trapsfhprep))
    else:
        input_massFracSFR = np.append(input_massFracSFR, input_massFracSFR[-1] + trap(traplbtprep*10**9,trapsfhprep))
    
# convert to mass percent
inputmassPercent = input_massFracSFR/input_massFracSFR[len(input_massFracSFR)-1]
inputmassLBT = (cosmo.age(obs['zred']).value - cosmo.age(gal['sfh'][:,0]).value)[1:len(cosmo.age(obs['zred']).value - cosmo.age(gal['sfh'][:,0]).value)]

'''
# PlOT + display input/output mass
ax[2].fill_between(lbt_interp, massPercent[:,1], massPercent[:,3], color='grey', alpha=.5)
ax[2].plot(lbt_interp,massPercent[:,2],color='black',lw=1.5,label='Output SFH')
ax[2].plot(inputmassLBT, inputmassPercent, color='blue',lw=1.5,label='Input SFH')
'''

################ t50/t95 calculations ################
# t50, t95 INPUT SFH
x_in_t50 = inputmassLBT[np.argmin(np.abs(0.50 - inputmassPercent))]
x_in_t95 = inputmassLBT[np.argmin(np.abs(0.95 - inputmassPercent))]

# t50, t95 OUTPUT SFH
x_rec_t50 = lbt_interp[np.argmin(np.abs(0.50 - massPercent[:,2]))]
x_rec_t95 = lbt_interp[np.argmin(np.abs(0.50 - massPercent[:,2]))]

'''
# plot t50, t95 on SFH + mass frac plots
ax[2].axvline(x_in_t50, linestyle='dotted', lw=1, color='blue')
ax[2].axvline(x_in_t95, linestyle='dotted', lw=1, color='blue')
ax[2].axvline(x_rec_t50, linestyle='dotted', lw=1, color='black')
ax[2].axvline(x_rec_t95, linestyle='dotted', lw=1, color='black')

ax[1].axvline(x_in_t50, linestyle='dotted', lw=1, color='blue')
ax[1].axvline(x_in_t95, linestyle='dotted', lw=1, color='blue')
ax[1].axvline(x_rec_t50, linestyle='dotted', lw=1, color='black')
ax[1].axvline(x_rec_t95, linestyle='dotted', lw=1, color='black')

ax[2].set_xlim(cosmo.age(gal['sfh'][:,0]).value[-1], 0)
ax[2].set_ylabel('Cumulative Mass Fraction')
ax[2].legend()
ax[2].set_xlabel('Lookback Time [Gyr]')
ax[2].set_ylim(0,1)
'''

'''
plt.show()
# save plot
fig.tight_layout()
if not os.path.exists(plotdir+'sfh'):
    os.mkdir(plotdir+'sfh')    
fig.savefig(plotdir+'sfh/' + filename, bbox_inches='tight')
  
print('saved sfh to '+plotdir+'sfh/'+filename) 
#plt.close(fig)
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
out = Output(phot_obs=phot_obs, phot_fit=phot_fit, output_sfh=sfrPercent, cmf=massPercent, 
    objname=str(obs['objid']), massfrac=massFrac, percentiles=percentiles, massTot=totmassPercent, 
    input_lbt=(cosmo.age(obs['zred']).value - cosmo.age(gal['sfh'][:,0]).value), output_lbt=lbt_interp, input_sfh=um_sfh,
    input_t50=x_in_t50, input_t95=x_in_t95, output_t50=x_rec_t50, output_t95=x_rec_t95, obs=obs, gal=gal, spsdict=spsdict)

# and save it
if not os.path.exists('dicts/z3mb/'):
    os.mkdir('dicts/z3mb/')
np.savez('dicts/z3mb/'+str(obs['objid'])+'.npz', res=out)

