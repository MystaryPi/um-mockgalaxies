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
"um_plot_prospector_outputs.py" 668L, 29394C

# corner plot
# maximum a posteriori (of the locations visited by the MCMC sampler)
# also put “truth” = input values for e.g. stellar mass, dust, metallicity,

# obtain gal from obs dictionary file
#load_values = np.load(res)
#gal = (np.load(res))['gal']

# Truth values


imax = np.argmax(res['lnprobability'])
theta_max = res['chain'][imax, :].copy()


print('MAP value: {}'.format(theta_max))
fig, axes = plt.subplots(len(theta_max), len(theta_max), figsize=(15,15))
axes = um_cornerplot.allcorner(res['chain'].T, mod.theta_labels(), axes, show_titles=True,
    span=[0.997]*len(mod.theta_labels()), weights=res.get("weights", None),

if not os.path.exists(plotdir+'corner'):
    os.mkdir(plotdir+'corner')
fig.savefig(plotdir+'corner/'+filename, bbox_inches='tight')
print('saved cornerplot to '+plotdir+filename)
plt.close(fig)
print('Made cornerplot')
##############################################################################
# generate model at MAP value
mspec_map, mphot_map, _ = mod.mean_model(theta_max, obs, sps=sps) #usually in restframe
# wavelength vectors
a = 1.0 + mod.params.get('zred', 0.0) # cosmological redshifting
# photometric effective wavelengths
wphot = np.array(obs["wave_effective"]) #is this correct?
# spectroscopic wavelengths
if obs["wavelength"] is None:
    wspec *= a #redshift them to be observed frame
else:
    wspec = obs["wavelength"]

print('Starting to calculate spectra...')
weights = res.get('weights',None)
    # wavelength obs = wavelength rest * (1+z), so this is observed wavelength
    allspec[0,:,ii] = sps.wavelengths.copy() * (1+res['chain'][i, mod.theta_index['zred']])
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
         marker='o', markersize=10, ls='', lw=3, alpha=0.8,
         markerfacecolor='none', markeredgecolor='black',
         markeredgewidth=3)
# MAP Model spectrum#
'''

###### FULL CONVERSION TO MAKE_SPECTRUM.PY
# wphot = wave_effective
c = 2.99792458e8 #m/s
ax[0].errorbar(wphot, (phot50*c/(wphot/(1+obs['zred']))**2.), label='Model photometry',
               yerr = (phot84-phot16), marker='s', markersize=10, alpha=0.8, ls='', lw=3,
         marker='o', markersize=10, ls='', lw=3, alpha=0.8,
         markerfacecolor='none', markeredgecolor='black',
         markeredgewidth=3)
# MAP Model spectrum
ax[0].set_yscale('log')


# originaly maggies * c/wphot**2
#norm_wl = ((wspec>6300) & (wspec<6500))
ax[0].set_ylim((1e-10, 1e-6))
ax[0].set_xlim((1e3, 1e5))
#ax[0].set_yscale('log')
ax[0].legend(loc='best', fontsize=12)
ax[0].set_title(str(int(obs['objid'])))
ax[0].tick_params(axis='both', which='major', labelsize=12)
print('Made spectrum plot')

######################## SFH for FLEXIBLE continuity model ########################
from prospect.models.transforms import logsfr_ratios_to_masses_psb, psb_logsfr_ratios_to_agebins


# obtain sfh from universemachine
um_sfh = gal['sfh'][:,1]

niter = res['chain'].shape[-2]
tmax = cosmo.age(np.min(flatchain[:,mod.theta_index['zred']])).value #matches scales

# will need to interpolate to get everything on same time scale
# make sure this matches b/t two model types!
'''
if tmax > 2:
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
    allagebins[iteration, :] = agebins
    dt = 10**agebins[:, 1] - 10**agebins[:, 0]
    allsfrs[iteration,:] = (masses  / dt)


# calculate interpolated SFR and cumulative mass
# with each likelihood draw you can convert the agebins from units of lookback time to units of age
# using the redshift at that likelihood draw, and put it on your fixed grid of ages

allagebins_ago = 10**allagebins/(1e9)
allsfrs_interp = np.zeros((flatchain.shape[0], len(age_interp)))
masscum_interp = np.zeros_like(allsfrs_interp)
totmasscum_interp = np.zeros_like(allsfrs_interp)

    tuniv = cosmo.age(flatchain[i,mod.theta_index['zred']]).value
    allsfrs_interp[i,-1] = 0
    masscum_interp[i,:] = 1 - (np.cumsum(allsfrs_interp[i,:] * dt) / np.sum(allsfrs_interp[i,:] * dt))
    totmasscum_interp[i,:] = np.sum(allsfrs_interp[i,:] * dt) - (np.cumsum(allsfrs_interp[i,:] * dt))

# sfr and cumulative mass percentiles
sfrPercent = np.array([quantile(allsfrs_interp[:,i], [16,50,84], weights=res.get('weights', None))
    for i in range(allsfrs_interp.shape[1])])
massPercent = np.array([quantile(masscum_interp[:,i], [16,50,84], weights=res.get('weights', None))
    for i in range(masscum_interp.shape[1])])
    for i in range(totmasscum_interp.shape[1])])
totmassPercent = np.concatenate((age_interp[:,np.newaxis], totmassPercent), axis=1) # add time
percentiles = get_percentiles(res, mod)

# mass fraction in the last Gyr
massFrac = 1 - massPercent[age_interp==1, 1:].flatten()[::-1]

############## MAXIMUM LIKELIHOOD ESTIMATE ##################
# Collect values for MLE, straight from UMachine...
t_obs = cosmo.age(gal['sfh'][:,0]).value
t_obs = tuniv - t_obs # convert t_obs (Gyr) to lookback time (log10(yr))

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


soln = minimize(nll, initial, args=(t_obs, sfr_obs, sfr_obs_err))
sfr_ml, agebins_ml = psb(t_obs, soln.x[0:4], soln.x[4], soln.x[5], soln.x[6:9], mtot=soln.x[9])

t_plot = tuniv - t_obs
t_plot = t_plot[::-1]

###### SFH ADJUSTED PLOTS ##############
# One-dimensional linear interpolation.
sfhadjusted_lower = np.interp(cosmo.age(gal['sfh'][:,0]).value, age_interp, sfrPercent[:,1])

######## PLOTTING ##########
#label='Input SFH (z = {0:.3f})'.format(spsdict['zred'])

for index in range(len(sfhadjusted_upper)):
    if np.isnan(sfhadjusted_upper[index]):


ax[1].plot(t_plot, sfr_ml[::-1], 'g--', lw=2, label='Maximum Likelihood SFH') # MLE SFH

#ax[1].fill_between(age_interp, sfrPercent[:,1], sfrPercent[:,3], color='grey', alpha=.5)
#ax[1].plot(age_interp, sfrPercent[:,2], label='Output SFH', color='black', lw=1.5, marker="o")

#ax[1].set_xlim(0, tuniv)
ax[1].set_xlim(0, cosmo.age(gal['sfh'][:,0]).value[-1])
ax[1].set_yscale('log')
#ax[1].legend(loc='best', fontsize=14)
ax[1].tick_params(axis='both', which='major', labelsize=12)
ax[1].set_ylabel('SFR [' + r'$M_{\odot} /yr$' + ']', fontsize = 12)
#ax[1].set_xlabel('years before observation [Gyr]')
ax[1].set_xlabel('Age [Gyr]', fontsize = 12)

# add secondary axis for redshift
#x_formatter = [1, 2, 3, 5, 7, 10, 15]
#x_locator = [5.75164694, 3.22662706, 2.11252719, 1.15475791, 0.75081398, 0.46588724, 0.26562898]
#secax = ax[1].twiny()
#secax.set_xticks(x_locator)
#secax.set_xticklabels(x_formatter)
#secax.tick_params(axis='x',which='both')
#secax.set_xlabel('Redshift [z]')

######## Derivative for SFH ###########
# Find derivatives of input + output SFH, age is adjusted b/c now difference between points
y_d_input = np.diff(um_sfh) / np.diff(cosmo.age(gal['sfh'][:,0]).value)
y_d_output = np.diff(sfhadjusted) / np.diff(cosmo.age(gal['sfh'][:,0]).value)

# Use intersect package to determine where derivatives intersect the quenching threshold
x_i, y_i = intersection_function(x_d, np.full(len(x_d), -100), y_d_input)
x_o, y_o = intersection_function(x_d, np.full(len(x_d), -100), y_d_output)

# Plot derivative for input + output SFH, + quenching threshold from Wren's paper
# Plot vertical lines for the quench time on the SFH plot
if len(x_i) != 0:
    ax[1].axvline(x_i[-1], linestyle='--', lw=1, color='blue')
else:

if len(x_o != 0):
    ax[1].axvline(x_o[-1], linestyle='--', lw=1, color='black')
else:
ax[2].axhline(-100, linestyle='--', color='maroon', label='-100 ' + r'$M_{\odot} yr^{-2}$' + ' quenching threshold') # Quench threshold

ax[2].set_xlim(0, tuniv)
ax[2].set_ylabel("SFH Time Derivative " + r'$[M_{\odot} yr^{-2}]$', fontsize=12)
ax[2].set_xlabel('Age [Gyr]', fontsize=12)
ax[2].legend(loc='best', fontsize=12)
ax[2].tick_params(axis='both', which='major', labelsize=12)

# cumulative mass fraction plot
#ax[2].fill_between(age_interp, massPercent[:,1], massPercent[:,3], color='grey', alpha=.5)
#ax[2].plot(age_interp, massPercent[:,2], color='black', lw=1.5)
#ax[2].set_xlim((tmax+.1,-.1))
#ax[2].set_ylabel('Cumulative mass fraction')
#ax[2].set_xlabel('years before observation [Gyr]')

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
