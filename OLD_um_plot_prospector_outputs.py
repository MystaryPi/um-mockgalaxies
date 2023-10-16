import numpy as np
from prospect.models import priors, SedModel
from prospect.models.sedmodel import PolySedModel
from prospect.models.templates import TemplateLibrary
from prospect.sources import CSPSpecBasis
from sedpy.observate import load_filters
import sedpy
from astropy.io import fits
from scipy import signal
import dynesty
import h5py
from matplotlib import pyplot as plt, ticker as ticker
#plt.interactive(True)
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

import um_prospector_param_file #to get the loadobs function to compare spectra
import make_spectrum #to get the metallicity values from the redone metallicity function. there needs to be an easier way to get gal

# set up cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=.3)

# read in a command-line argument for which output to plot where
if len(sys.argv) > 0:
    outroot = sys.argv[1]
    #plotdir = sys.argv[2]
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
        newval[(ts <= ab[i,0]) & (ts > ab[i,1])] = val[i] #previously >= and <
    return newval    
  
##############################################################################  
    
# grab results (dictionary), the obs dictionary, and our corresponding models
res, obs, mod = results_from("{}".format(outroot), dangerous=True)
gal = (np.load('obs-z3/umobs_'+str(obs['objid'])+'.npz', allow_pickle=True))['gal']
spsdict = (np.load('obs-z3/umobs_'+str(obs['objid'])+'.npz', allow_pickle=True))['params'][()]

#changed this for now to get obs b/c i forgot obsid
#npzfile = np.load('obs/umobs_121137241.npz', allow_pickle=True) 
#obs = npzfile['obs'].item()
#obs['objid'] = 121137241
sps = get_sps(res)
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
filename = str(obs['objid']) + '_{}.pdf' #defines filename for all objects
while os.path.isfile(plotdir+'trace/'+filename.format(counter)):
    counter += 1
filename = filename.format(counter) #iterate until a unique file is made

plt.savefig(plotdir+'trace/'+filename, bbox_inches='tight')
print('saved tracefig to '+plotdir+'trace/'+filename)
plt.close()


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
truth_array = [spsdict['zred'], spsdict['logzsol'], spsdict['dust2'], obs['logM'], 0, 0, 0, 0, 0, 0, 0, 0, 0, spsdict['dust_index']]

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
mspec_map, mphot_map, _ = mod.mean_model(theta_max, obs, sps=sps)
# wavelength vectors
a = 1.0 + mod.params.get('zred', 0.0) # cosmological redshifting
# photometric effective wavelengths
wphot = np.array(obs["wave_effective"]) #is this correct?
# spectroscopic wavelengths
if obs["wavelength"] is None:
    # *restframe* spectral wavelengths, since obs["wavelength"] is None
    wspec = sps.wavelengths.copy()
    wspec *= a #redshift them
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
    allspec[0,:,ii] = sps.wavelengths.copy() * (1+res['chain'][i, mod.theta_index['zred']][0])
    allspec[1,:,ii], allphot[:,ii], allmfrac[ii] = mod.mean_model(res['chain'][i,:], obs, sps=sps)
phot16 = np.array([quantile(allphot[i,:], 16, weights = weights[idx]) for i in range(allphot.shape[0])])
phot50 = np.array([quantile(allphot[i,:], 50, weights = weights[idx]) for i in range(allphot.shape[0])])
phot84 = np.array([quantile(allphot[i,:], 84, weights = weights[idx]) for i in range(allphot.shape[0])])
print('Done calculating spectra')

# Make plot of data and model
c = 2.99792458e18
fig, ax = plt.subplots(3,1,figsize=(8,10))
ax[0].plot(wspec, mspec_map*c/wspec**2., label='MAP Model spectrum',
       lw=1.5, color='grey', alpha=0.7, zorder=10)    
ax[0].errorbar(wphot, phot50*c/wphot**2., label='Model photometry',
         yerr = phot84-phot16,
         marker='s', markersize=10, alpha=0.8, ls='', lw=3, 
         markerfacecolor='none', markeredgecolor='green', 
         markeredgewidth=3)
ax[0].errorbar(wphot, obs['maggies'] * c/wphot**2, yerr=obs['maggies_unc']*c/wphot**2, 
         label='Observed photometry', ecolor='red', 
         marker='o', markersize=10, ls='', lw=3, alpha=0.8, 
         markerfacecolor='none', markeredgecolor='black', 
         markeredgewidth=3)            
# originaly maggies * c/wphot**2
norm_wl = ((wspec>6300) & (wspec<6500))
norm = np.nanmax(obs['maggies']*c/wphot**2)
ax[0].set_ylim((-0.2*norm, norm*2))
ax[0].set_xlim((1e3, 1e5))
ax[0].set_xlabel('Wavelength [A]')
ax[0].set_ylabel(r'$F_{\lambda}$')
ax[0].set_xscale('log')
ax[0].legend(loc='best', fontsize=14)
ax[0].set_title(outroot + ', ' + str(obs['objid']))
print('Made spectrum plot')

######################## SFH for FLEXIBLE continuity model ########################
from prospect.models.transforms import logsfr_ratios_to_masses_psb, psb_logsfr_ratios_to_agebins

# from squiggle_flex_continuity import modified_logsfr_ratios_to_masses_flex, modified_logsfr_ratios_to_agebins

# obtain sfh from universemachine
um_sfh = gal['sfh'][:,1]

# actual sfh percentiles
flatchain = res["chain"]
niter = res['chain'].shape[-2]
tmax = cosmo.age(np.min(flatchain[:,mod.theta_index['zred']])).value

# will need to interpolate to get everything on same time scale
# make sure this matches b/t two model types!
'''
if tmax > 2:
    age_interp = np.concatenate((np.arange(0,2,.001),np.arange(2,tmax,.01),[tmax])) #np.append(np.arange(0, tuniv, 0.01), tuniv)
else:
    age_interp = np.arange(0,tmax+.005, .001)  
'''  
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
        logmass = flatchain[iteration, mod.theta_index["logmass"]]      
    agebins = psb_logsfr_ratios_to_agebins(logsfr_ratios=logr, agebins=mod.params['agebins'], tlast=tlast, 
        tflex=mod.params['tflex'], nflex=mod.params['nflex'], nfixed=mod.params['nfixed'])
    allagebins[iteration, :] = agebins
    dt = 10**agebins[:, 1] - 10**agebins[:, 0]
    masses = logsfr_ratios_to_masses_psb(logsfr_ratios=logr, logmass=logmass, agebins=agebins,
        logsfr_ratio_young=logr_young, logsfr_ratio_old=logr_old,
        tlast=tlast, tflex=mod.params['tflex'], nflex=mod.params['nflex'], nfixed=mod.params['nfixed']) #issues
    allsfrs[iteration,:] = (masses  / dt)


# calculate interpolated SFR and cumulative mass  
# with each likelihood draw you can convert the agebins from units of lookback time to units of age 
# using the redshift at that likelihood draw, and put it on your fixed grid of ages

allagebins_ago = 10**allagebins/(1e9) 
allsfrs_interp = np.zeros((flatchain.shape[0], len(age_interp)))
masscum_interp = np.zeros_like(allsfrs_interp)
totmasscum_interp = np.zeros_like(allsfrs_interp)
dt = (age_interp - np.insert(age_interp,0,0)[:-1]) * 1e9
for i in range(flatchain.shape[0]):
    tuniv = cosmo.age(flatchain[i,mod.theta_index['zred']]).value
    allsfrs_interp[i,:] = stepInterp(tuniv - allagebins_ago[i,:], allsfrs[i,:], age_interp) #age_interp in age now, allagebins_ago in age #age = t_univ(z) - lookback time
    allsfrs_interp[i,-1] = 0
    masscum_interp[i,:] = 1 - (np.cumsum(allsfrs_interp[i,:] * dt) / np.sum(allsfrs_interp[i,:] * dt))
    totmasscum_interp[i,:] = np.sum(allsfrs_interp[i,:] * dt) - (np.cumsum(allsfrs_interp[i,:] * dt))

# sfr and cumulative mass percentiles 
sfrPercent = np.array([quantile(allsfrs_interp[:,i], [16,50,84], weights=res.get('weights', None)) 
    for i in range(allsfrs_interp.shape[1])])
sfrPercent = np.concatenate((age_interp[:,np.newaxis], sfrPercent), axis=1) # add time # NEEDS REDSHIFT IN AGE_INTERP
massPercent = np.array([quantile(masscum_interp[:,i], [16,50,84], weights=res.get('weights', None)) 
    for i in range(masscum_interp.shape[1])])    
massPercent = np.concatenate((age_interp[:,np.newaxis], massPercent), axis=1) # add time          
totmassPercent = np.array([quantile(totmasscum_interp[:,i], [16,50,84], weights=res.get('weights', None)) 
    for i in range(totmasscum_interp.shape[1])])    
totmassPercent = np.concatenate((age_interp[:,np.newaxis], totmassPercent), axis=1) # add time

# all percentiles...
percentiles = get_percentiles(res, mod)

# mass fraction in the last Gyr
massFrac = 1 - massPercent[age_interp==1, 1:].flatten()[::-1]  

# plot sfh and percentiles
ax[1].fill_between(age_interp, sfrPercent[:,1], sfrPercent[:,3], color='grey', alpha=.5)
ax[1].plot(age_interp, sfrPercent[:,2], label='Output SFH', color='black', lw=1.5)
ax[1].plot(cosmo.age(gal['sfh'][:,0]).value, um_sfh, label='Input SFH', color='blue', lw=1, marker="o") #convert age since beginning of universe

ax[1].set_xlim(0, tuniv)
ax[1].set_yscale('log')
ax[1].legend(loc='best', fontsize=12)
ax[1].set_ylabel('SFR [Msun/yr]')
#ax[1].set_xlabel('years before observation [Gyr]')
ax[1].set_xlabel('Age [Gyr]')

# add secondary axis for redshift

x_formatter = [1, 2, 3, 5, 7, 10, 15]
x_locator = [5.75164694, 3.22662706, 2.11252719, 1.15475791, 0.75081398, 0.46588724, 0.26562898]
secax = ax[1].twiny()
secax.set_xticks(x_locator)
secax.set_xticklabels(x_formatter)
secax.tick_params(axis='x',which='both')
secax.set_xlabel('Redshift [z]')
fig.tight_layout() 


# cumulative mass fraction plot
#ax[2].fill_between(age_interp, massPercent[:,1], massPercent[:,3], color='grey', alpha=.5)
#ax[2].plot(age_interp, massPercent[:,2], color='black', lw=1.5)
#ax[2].set_xlim((tmax+.1,-.1))
#ax[2].set_ylabel('Cumulative mass fraction')
#ax[2].set_xlabel('years before observation [Gyr]')

# save plot
if not os.path.exists(plotdir+'sfh'):
    os.mkdir(plotdir+'sfh')    
fig.savefig(plotdir+'sfh/' + filename, bbox_inches='tight')
  
print('saved sfh to '+plotdir+'sfh/'+filename) 
plt.close(fig)
print('Made SFH plot')

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

