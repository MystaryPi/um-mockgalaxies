# direct_fit.py
# Stores functions that will fit Prospector SFH models directly to input SFHs
# Edited for a fixed redshift and adjustable tflex_frac when running Prospector
#
# Original functions by Mia
####################################################

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.optimize import minimize
from scipy.stats import t

# Import prospector packages
from prospect.models import priors, SedModel
from prospect.models.sedmodel import PolySedModel
from prospect.models.templates import TemplateLibrary
from prospect.sources import CSPSpecBasis
from sedpy.observate import load_filters
#from prospect.models.transforms import logsfr_ratios_to_masses
import emcee

# set up cosmology
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=.3)
#zred = 0.01
zred = 0.01
tuniv = cosmo.age(zred).value

from prospect.io.read_results import results_from, get_sps
from prospect.io.read_results import traceplot, subcorner
from um_prospector_param_file import updated_logsfr_ratios_to_masses_psb, updated_psb_logsfr_ratios_to_agebins

def trap(x, y):
        return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]))/2. 

def stepInterp(ab, val, ts):
    '''ab: agebins vector
    val: the original value (sfr, etc) that we want to interpolate
    ts: new values we want to interpolate to '''
    newval = np.zeros_like(ts) + np.nan
    for i in range(0,len(ab)):
        newval[(ts >= ab[i,0]) & (ts < ab[i,1])] = val[i]
    return newval  

# Test with PSB model
####################################################
def fit_flexpsb(galidx, outroot, verbose=False, massfree=True, nsteps=5000, discard=2000):  
    # retrieved t_obs, sfr_obs, sfr_obs_err, mtot_true 
    gal = (np.load('/Users/michpark/JWST_Programs/mockgalaxies/obs-z3/umobs_'+str(galidx)+'.npz', allow_pickle=True))['gal']
    
    sfr_obs = gal['sfh'][:,1]
    sfr_obs_err = np.ones_like(sfr_obs) * 1e-3 * max(sfr_obs) 
    t_obs = cosmo.age(gal['sfh'][:,0]).value #age
    mtot_true = trap(t_obs*1e9, sfr_obs) 

    res, obs, mod = results_from("{}".format(outroot), dangerous=True) 
    print("{}".format(outroot))
    sps = get_sps(res)

    # nflex, nfixed, and tflex_frac are the same for all draws, so we can grab them here:
    nflex = mod.params['nflex']
    nfixed = mod.params['nfixed']
    tflex_frac = mod.params['tflex_frac']
    tflex = tflex_frac[0]*tuniv
    zred = mod.params['zred'] #CHECK THIS
    tlast_fraction = mod.params['tlast_fraction']

    # Define priors
    def priors(tflex=tflex, nflex=nflex, nfixed=nfixed, tquench=0.2):
        '''set smarter priors on the logSFR ratios. given a 
        redshift zred, use the closest-z universe machine SFH
        to set a better guess than a constant SFR. returns
        agebins, logmass, and set of logSFR ratios that are
        self-consistent. '''

        # nflex, nfixed are arrays with single values, retrieve index
        nflex = nflex[0]
        nfixed = nfixed[0]

        # define default agebins
        agelims = np.array([1, tquench*1e9] + \
            np.linspace((tquench+.1)*1e9, (tflex)*1e9, nflex).tolist() \
            + np.linspace(tflex*1e9, tuniv*1e9, nfixed+1)[1:].tolist())
        agebins = np.array([np.log10(agelims[:-1]), np.log10(agelims[1:])]).T 

        # subsample to make sure we'll have umachine bin for everything
        newages = np.arange(0, tuniv, 0.001)
        sfh = np.ones_like(newages) / len(newages)
        mass = sfh*0.001*1e9 # mass formed in each little bin    

        # get default agebins in same units
        abins_age = 10**agebins/1e9 # in Gyr

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
        agebins = np.log10(1e9*abins_age)

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

    # Define model 
    def flexpsb(ages, logsfr_ratios, tquench, logsfr_ratio_young, logsfr_ratio_old, mtot, tflex=tflex, tflex_frac=tflex_frac, nflex=nflex, nfixed=nfixed):
        # define initial agebins
        agelims = np.array([1, tquench*1e9] + \
            np.linspace((tquench+.1)*1e9, (tflex)*1e9, nflex[0]).tolist() \
            + np.linspace(tflex*1e9, tuniv*1e9, nfixed[0]+1)[1:].tolist())
        agebins = np.array([np.log10(agelims[:-1]), np.log10(agelims[1:])]).T 

        # This is the correct format 
        logsfr_ratio_young = np.array([logsfr_ratio_young])
        logmass = np.array([mtot]) # already in logmass     
        
        agebins = updated_psb_logsfr_ratios_to_agebins(logsfr_ratios=logsfr_ratios, agebins=agebins, 
            tlast_fraction=tlast_fraction, tflex_frac=tflex_frac, nflex=nflex, nfixed=nfixed, zred=zred) 

        #agebins = updated_psb_logsfr_ratios_to_agebins(logsfr_ratios=logsfr_ratios, agebins=mod.params['agebins'], 
        #    tlast_fraction=tlast_fraction, tflex_frac=tflex_frac, nflex=nflex, nfixed=nfixed, zred=zred)
    
        dt = 10**agebins[:, 1] - 10**agebins[:, 0]
        masses = updated_logsfr_ratios_to_masses_psb(logsfr_ratios=logsfr_ratios, logmass=logmass, agebins=agebins, 
            logsfr_ratio_young=logsfr_ratio_young, logsfr_ratio_old=logsfr_ratio_old, tlast_fraction=tlast_fraction, tflex_frac=tflex_frac, 
            nflex=nflex, nfixed=nfixed, zred=zred)
        sfrs = (masses  / dt)

        # interpolate SFR to time array
        agebins_ago = 10**agebins/1e9  
        age_interp = ages
        dt = (age_interp - np.insert(age_interp,0,0)[:-1]) * 1e9

        #print('test', age_interp, agebins_ago)

        sfh = stepInterp(agebins_ago, sfrs, age_interp)
        sfh[-1] = 0.
        sfh[np.isnan(sfh)] = 0.
        
        # and normalize it so that the total stellar mass formed is mtot
        # mtot is in logmass, 10**mtot gives Msun units
        sfh = sfh * (10**mtot / trap(ages*1e9, sfh)) # ok, now we're in Msun/yr units

        return sfh, agebins_ago

    # Fit model to observations
    def log_likelihood(theta, x, y, yerr):
        if massfree:
            logr0, logr1, logr2, logr3, tquench, logsfr_ratio_young, logrold0, logrold1, logrold2, mtot = theta
        else:
            logr0, logr1, logr2, logr3, tquench, logsfr_ratio_young, logrold0, logrold1, logrold2 = theta

        # group some of the variables
        logsfr_ratios = np.array([logr0, logr1, logr2, logr3])
        logsfr_ratio_old = np.array([logrold0, logrold1, logrold2])

        # run model
        if massfree:
            model, _ = flexpsb(x, logsfr_ratios, tquench, logsfr_ratio_young, logsfr_ratio_old, mtot, tflex, tflex_frac, nflex, nfixed)
        else:
            model, _ = flexpsb(x, logsfr_ratios, tquench, logsfr_ratio_young, logsfr_ratio_old, 1.0, tflex, tflex_frac, nflex, nfixed)

        sigma2 = yerr**2
        test = -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
        return test

    # Define prior and total probability function
    def log_prior(theta):
        if massfree:
            logr0, logr1, logr2, logr3, tquench, logsfr_ratio_young, logrold0, logrold1, logrold2, mtot = theta
        else:
            logr0, logr1, logr2, logr3, tquench, logsfr_ratio_young, logrold0, logrold1, logrold2 = theta
        logsfr_ratio_young_init, logsfr_ratios_init, logsfr_ratio_old_init = priors(tquench=tquench_init)
        logr_rv = t(df=np.ones(nflex-1), loc=logsfr_ratios_init, scale=np.ones(nflex-1)*0.3)
        logrold_rv = t(df=np.ones(nfixed), loc=logsfr_ratio_old_init, scale=np.ones(nfixed)*0.3)
        logryoung_rv = t(df=1, loc=logsfr_ratio_young_init, scale=0.5)

        if not 0.01 < tquench < 1.5:
            return -np.inf
        if massfree and (not 4 < np.log10(mtot) < 11):
            return -np.inf
        
        test = np.sum(np.log(logr_rv.pdf([logr0,logr1,logr2,logr3]))) + np.sum(np.log(logrold_rv.pdf([logrold0,logrold1,logrold2]))) + np.log(logryoung_rv.pdf(logsfr_ratio_young))
        return test

    def log_probability(theta, x, y, yerr):
        lp = log_prior(theta)
        ll = log_likelihood(theta, x, y, yerr)
        if ~np.isfinite(lp) or ~np.isfinite(ll):
            return -np.inf
        return lp + ll

    nll = lambda *args: -log_likelihood(*args)
    tquench_init = 0.2
    mtot_init = obs['logM']
    logsfr_ratio_young_init, logsfr_ratios_init, logsfr_ratio_old_init = priors(tquench=tquench_init)
    logr0_init, logr1_init, logr2_init, logr3_init = logsfr_ratios_init
    logrold0_init, logrold1_init, logrold2_init = logsfr_ratio_old_init
    if massfree:
        initial = np.array([logr0_init, logr1_init, logr2_init, logr3_init, tquench_init, logsfr_ratio_young_init, logrold0_init, logrold1_init, logrold2_init, mtot_init])
    else:
        initial = np.array([logr0_init, logr1_init, logr2_init, logr3_init, tquench_init, logsfr_ratio_young_init, logrold0_init, logrold1_init, logrold2_init])

    print("initial: " + str(initial))
    soln = minimize(nll, initial, args=(t_obs, sfr_obs, sfr_obs_err)) #RUNS MLE HERE!!!!
    print("soln: " + str(soln))

    pos = initial + 1e-4 * np.random.randn(32, len(initial))
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(t_obs, sfr_obs, sfr_obs_err)
    )

    sampler.run_mcmc(pos, nsteps, progress=True)

    if verbose:
        fig, axes = plt.subplots(ndim, figsize=(ndim, 7), sharex=True)
        samples = sampler.get_chain()
        #labels = ["tau", "tstart"]
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            #ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
        plt.show()

    flat_samples = sampler.get_chain(discard=discard, flat=True)
    results_mc = np.percentile(flat_samples, [16,50,84], axis=0)

    print("results mc: " + str(results_mc))

    print("sampler: " + str(sampler))

    if massfree:
        mtot_mc = results_mc[1,9]
    else:
        mtot_mc = 1.0
    sfr_mc, agebins_mc = flexpsb(t_obs, results_mc[1,0:4], results_mc[1,4], results_mc[1,5], results_mc[1,6:9], mtot=mtot_mc) 
    # MIGHT NEED TO ADD NEW PARAMS HERE
    
    # check mass?
    if massfree:
        print('True mass:', np.log10(mtot_true))
        print('Output mass:', results_mc[1,9]) # this is already in logmass 

    # try converting the input SFH to match the binning of the Prospector SFH
    '''
    sfr_rebin = np.zeros(len(agebins_mc))
    for binnum, bin in enumerate(agebins_mc):
        binidx = np.where((t_obs >= bin[0]) & (t_obs < bin[1]))[0]
        print(len(binidx))
        if len(binidx)==1:
            sfr_rebin[binnum] = sfr_obs[binnum]
        elif len(binidx)==0:
            sfr_rebin[binnum] = np.nan
        else:
            sumsfr = trap(t_obs[binidx],sfr_obs[binidx])
            sfr_rebin[binnum] = sumsfr/(t_obs[binidx][-1]-t_obs[binidx][0])
    
    if np.any(np.isnan(sfr_rebin)):
        badidx = np.where(np.isnan(sfr_rebin))[0]
        sfr_rebin = np.delete(sfr_rebin, badidx)
        agebins_mc[badidx-1,1] = agebins_mc[badidx,1]
        agebins_mc = np.delete(agebins_mc, badidx, axis=0)
    sfr_rebin = stepInterp(agebins_mc, sfr_rebin, t_obs)
    '''
    # plot the final SFHs
    '''
    t_plot = tuniv - t_obs
    plt.fill_between(t_plot, sfr_obs-sfr_obs_err, sfr_obs+sfr_obs_err, color='gray', alpha=0.5)
    plt.plot(t_plot, sfr_obs, 'k-', lw=2, label='Input SFH')
    inds = np.random.randint(len(flat_samples), size=100)
    for ind in inds:
        sample = flat_samples[ind]
        if massfree:
            plt.plot(t_plot, flexpsb(t_obs, sample[0:4], sample[4], sample[5], sample[6:9], mtot=sample[9])[0], 'C1', lw=1, alpha=0.1)
        else:
            plt.plot(t_plot, flexpsb(t_obs, sample[0:4], sample[4], sample[5], sample[6:], mtot=1.0)[0], 'C1', lw=1, alpha=0.1)
    plt.plot(t_plot, sfr_mc, 'r--', lw=2, label='Best-fit SFH')
    plt.plot(t_plot, sfr_rebin, 'b:', lw=2, label='Input SFH, binned')
    
    if massfree:
        plt.title('True mass:{0:.2f}, Output mass:{1:.2f}'.format(np.log10(mtot_true),np.log10(results_mc[1,9])), fontsize=12)

    plt.legend(loc='best')
    plt.xlabel('t (Gyr)')
    plt.ylabel('Normalized SFH')
    #plt.savefig('figs/'+str(galidx)+'_psbfit.png', bbox_inches='tight')
    plt.show()

    if verbose:
        import corner
        fig = corner.corner(flat_samples)
    '''
    print("t_obs: " + str(t_obs))
    print("sfr_mc: " + str(sfr_mc))
    print("agebins_mc:" + str(agebins_mc))
    return t_obs, sfr_mc
