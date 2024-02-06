'''
Duplicate of plotting file, except that it takes plots the SFH of various tflex values.

Input objid of the galaxy to plot + will automatically retrieve tflex versions:
run tflex-plots.py 559120319
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
from scipy.optimize import minimize
from scipy.stats import chisquare
from scipy.optimize import fsolve
import dynesty
from dynesty import utils
import h5py
from matplotlib import pyplot as plt, ticker as ticker; plt.interactive(True)
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
from um_prospector_param_file import updated_logsfr_ratios_to_masses_psb, updated_psb_logsfr_ratios_to_agebins


# set up cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=.3)

# read in a command-line argument for THE OBJID of the galaxy
if len(sys.argv) > 0:
    objid = str(sys.argv[1]) # example: 559120319

plotdir = '/Users/michpark/JWST_Programs/mockgalaxies/plots-tflex/'

# Retrieve correct mcmc files for mb + nomb
root = '/Users/michpark/JWST_Programs/mockgalaxies/tflex-mcmc/'
# 0.45, 0.55, 0.65, 0.75 - spaced apart by 0.1
tflex_array = [0.45, 0.55, 0.65, 0.75]
tflex_directory = [""]*4
for p in range(len(tflex_array)):
    directory_string = root + 'tflex-' + str(tflex_array[p]) + '/'
    for files in os.walk(directory_string):
            for filename in files[2]:
                if objid in filename:
                    name_path = os.path.join(directory_string,filename)
                    tflex_directory[p] = name_path # populate the directory of paths    

# Display each tflex directory
print('Making plots for...')
for p in range(len(tflex_array)):
    print(str(tflex_array[p]) + ": " + tflex_directory[p])

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

##############################################################################  
# make sure plot directory exits
if not os.path.exists(plotdir):
    os.mkdir(plotdir)

#check to see if duplicates exist
counter=0
filename = objid + '_tflex_{}.pdf' #defines filename for all objects
while os.path.isfile(plotdir+filename.format(counter)):
    counter += 1
filename = filename.format(counter) #iterate until a unique file is made

fig, ax = plt.subplots(figsize=(8,6))

for p in range(len(tflex_array)):
    # outroot_index = 0 is MB, outroot_index = 1 is no MB
    # grab results (dictionary), the obs dictionary, and our corresponding models
    res, obs, mod = results_from("{}".format(tflex_directory[p]), dangerous=True) 
    sps = get_sps(res)

    gal = (np.load('/Users/michpark/JWST_Programs/mockgalaxies/obs-z3/umobs_'+str(obs['objid'])+'.npz', allow_pickle=True))['gal']
    spsdict = (np.load('/Users/michpark/JWST_Programs/mockgalaxies/obs-z3/umobs_'+str(obs['objid'])+'.npz', allow_pickle=True))['params'][()]


    print('Object ID: ' + str(obs['objid']))
    print('Loaded results')
    
    ##############################################################################  
    # corner plot
    truth_array = [gal['z'], spsdict['logzsol'], spsdict['dust2'], obs['logM'], 0, 0, 0, 0, 0, 0, 0, 0, 0, spsdict['dust_index']]
    imax = np.argmax(res['lnprobability'])
    theta_max = res['chain'][imax, :].copy()
    '''
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
    '''

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

    ######################## SFH for FLEXIBLE continuity model ########################
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
    ''' wren update: name of variable now matches units! so "lbt_interp" is the lookback time, and "age_interp" is in age of universe. '''
    if tmax > 2:
        lbt_interp = np.concatenate((np.arange(0,2,.001),np.arange(2,tmax,.01),[tmax])) #np.append(np.arange(0, tuniv, 0.01), tuniv)
    else:
        lbt_interp = np.arange(0,tmax+.005, .001)    
    lbt_interp[0] = 1e-9    
    age_interp = tmax - lbt_interp

    # nflex, nfixed, and tflex_frac are the same for all draws, so we can grab them here:
    nflex = mod.params['nflex']
    nfixed = mod.params['nfixed']
    tflex_frac = mod.params['tflex_frac']
    zred_thisdraw = np.array([])
    
    # actual sfh percentiles
    allsfrs = np.zeros((flatchain.shape[0], len(mod.params['agebins'])))
    allagebins = np.zeros((flatchain.shape[0], len(mod.params['agebins']), 2))
    for iteration in range(flatchain.shape[0]):
        zred = flatchain[iteration, mod.theta_index['zred']]
        zred_thisdraw = np.append(zred_thisdraw, zred) # collect zred values for each draw
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
        masscum_interp[i,:] = 1 - (np.cumsum(allsfrs_interp[i,:] * dt) / np.sum(allsfrs_interp[i,:] * dt))
        totmasscum_interp[i,:] = np.sum(allsfrs_interp[i,:] * dt) - (np.cumsum(allsfrs_interp[i,:] * dt))
    
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
    print(percentiles) # prints percentiles

    # mass fraction in the last Gyr
    massFrac = 1 - massPercent[lbt_interp==1, 1:].flatten()[::-1]  

    ######## SFH PLOTTING in LBT ##########
    # Obtain a range of colors (corresponding with tflex)
    color_table = ['#83deff', '#419cbd', '#107394', '#104a62']

    ax.plot(cosmo.age(obs['zred']).value - cosmo.age(gal['sfh'][:,0]).value, um_sfh, label='True SFH' if p == 0 else "", color='black', lw=1.7, marker="o") # INPUT SFH
    # Each for loop iteration plots output SFH in a different color
    ax.plot(lbt_interp, sfrPercent[:,2], lw=1.7, color=color_table[p], label=str(tflex_array[p])) 
    ax.fill_between(lbt_interp, sfrPercent[:,1], sfrPercent[:,3], color=color_table[p], alpha=.1)

    print('Finished SFH')
 
    ######## Derivative for SFH ###########
    # Eliminates 0 values from the SFHs, which can skew the derivative; limits quenchtime search for output
    # SFH to only be within input SFH's range
    input_mask = [i for i in enumerate(um_sfh) if i == 0]
    output_mask = [i for i, n in enumerate(sfrPercent[:,2]) if n == 0]

    input_sfh = um_sfh[::-1]
    input_lbt = (cosmo.age(obs['zred']).value - cosmo.age(gal['sfh'][:,0]).value)[::-1]
    output_sfh = sfrPercent[:,2]
    output_lbt = lbt_interp

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
        ax.axvline(x_i[0], linestyle='--', lw=1.5, color='black')
    
    if len(x_o != 0):
        ax.axvline(x_o[0], linestyle='--', lw=1.5, color=color_table[p])

ax.set_xlim(cosmo.age(gal['sfh'][:,0]).value[-1], 0)
ax.set_yscale('log')
ax.legend(title="tflex_fraction", loc='best', fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.set_ylabel('SFR [' + r'$M_{\odot} /yr$' + ']', fontsize = 10)
#ax.set_xlabel('years before observation [Gyr]')
ax.set_xlabel('Lookback Time [Gyr]', fontsize = 10)

plt.show()
# save plot
fig.tight_layout() 
fig.savefig(plotdir + filename, bbox_inches='tight')

print('saved tflex comparison plot to '+plotdir+filename) 
print('Made tflex plot')

