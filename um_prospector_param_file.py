import numpy as np
from prospect.models import priors, SedModel
from prospect.models.sedmodel import PolySedModel
from prospect.models.templates import TemplateLibrary, describe
from prospect.sources import CSPSpecBasis, FastStepBasis
from sedpy.observate import load_filters
import sedpy
from astropy.io import fits
from scipy import signal
import dynesty
import h5py
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import truncnorm
import os
from prospect.likelihood import NoiseModel
from prospect.likelihood.kernels import Uncorrelated
from prospect.utils.obsutils import fix_obs
import glob
from astropy.cosmology import z_at_value
import sedpy

# set up cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=.3)

''' This fit: uses the flexible continuity SFH. Make sure to set last agebin to something relatively short: it and
the first agebin do not get to move, but the rest do. '''

# --------------
# Run Params Dictionary Setup
# --------------

run_params = {'verbose':True, #this controls how much output to screen
              'debug':False, #if true, stops before mcmc burn in so you can look at model and stuff
              'outfile':'squiggle', 
              'output_pickles': False,
              'rescale_spectrum': False, # we have real units on our spectrum
              # --- Optimization parameters ---
              'do_levenburg': False,          # Do Levenburg optimization? (requires do_powell=False)
              'nmin': 10,                     # number of initial locations (drawn from prior) for Levenburg-Marquardt
              'do_powell': False,             # Do powell minimization? (deprecated)
              'ftol':0.5e-5, 'maxfev': 5000,  # Parameters for powell
              # --- emcee parameters ---
              'nwalkers': 128,                # number of emcee walkers
              'nburn': [16, 16, 32, 64],      # number of iterations in each burn-in round
              'niter': 512,                   # number of iterations in production
              'interval': 0.2,                # fraction of production run at which to save to hdf5.
              'initial_disp': 0.1,            # default dispersion in parameters for walker ball
              # Obs data parameters
              'objid':0,
              'mediumBands': True,
              #'catfile': '/Users/michpark/JWST_Programs/UNCOVER_DR1_LW_D032_catalog.fits',
              'phottable':None,                                     
              'logify_spectrum':False,
              'normalize_spectrum':False,
              # Model parameters
              'add_neb': True,
              'add_dust': True,
              # SPS parameters
              'zcontinuous': 1,
              'zspec': None, # this leaves redshift a free parameter
              'tflex_frac': 0.6,
              'tlast_max_frac': 0.3,
              # --- dynesty parameters ---              
              'dynesty':True,
              'nested_bound': 'multi',        # bounding method
              'nested_sample': 'rwalk',       # sampling method
              #'nested_walks': 32,     # original-64 sampling gets very inefficient w/ high S/N spectra
              'nested_nlive_init': 1600, # a finer resolution in likelihood space # can decrease this to ~400 if fixing to z_spec for shorter runtime
              'nested_nlive_batch': 400,
              'nested_maxbatch': None, # was None-- changed re ben's email 5/21/19
              'nested_maxcall': None,
              'nested_maxcall_init':None,
              'nested_bootstrap': 20,
              'nested_dlogz_init': 0.01,
              'nested_weight_kwargs':{'pfrac': 1.0},
              'nested_target_n_effective':20000,
              'nested_first_update':{'min_ncall': 20000, 'min_eff': 7.5},
              #'nested_stop_kwargs': {"post_thresh": 0.1}, # might want to lower this to 0.02ish once I get things working
              # --- nestle parameters ---
              'nestle_npoints': 2000,         # Number of nested sampling live points
              'nestle_method': 'multi',       # Nestle sampling method
              'nestle_maxcall': int(1e7),     # Maximum number of likelihood calls (ends even if not converged)
              }


# --------------
# OBS
# --------------
def load_obs(objid, mediumBands, **kwargs):
    """Load an UniverseMachine spectrum.
             
    Load photometry from an ascii file.  Assumes the following columns:
    `objid`, `filterset`, [`mag0`,....,`magN`] where N >= 11.  The User should
    modify this function (including adding keyword arguments) to read in their
    particular data format and put it in the required dictionary.

    :param objid:
        The object id for the row of the photomotery file to use.  Integer.
        Requires that there be an `objid` column in the ascii file.

    :param phottable:
        Name (and path) of the ascii file containing the photometry.

    :param luminosity_distance: (optional)
        The Johnson 2013 data are given as AB absolute magnitudes.  They can be
        turned into apparent magnitudes by supplying a luminosity distance.

    :returns obs:
        Dictionary of observational data.

    :returns gal:
        Dictionary of original UniverseMachine galaxy characteristics
    """    
                         
    # say what we're doing...
    print('loading ID '+str(objid))      

    # load file obs dictionary 
    with np.load('obs/umobs_'+str(objid)+'.npz', allow_pickle=True) as d:
        obs = d['obs'].item()
        gal = d['gal']
        sps = d['params'].item()  
    if(mediumBands == True):
        obs['phot_mask'] = [True]*len(obs['maggies']) #always true because our fake data is all good
    else:
        # No medium bands - exclude the bands that we don't need, updated for HST+UNCOVER vs. MB
        obs['phot_mask'] = [True, True, True, True, True, True, True, False, False, False, 
                False, False, False, False, False, False, False, False, False, False, False]
    
    obs['objid'] = objid
    obs = fix_obs(obs)
    assert 'phot_mask' in obs.keys()
    
    obs['filters'] = sedpy.observate.load_filters(obs['filternames'])
     
    return obs
    
    
# -----------------
# Helper Functions
# ------------------    
def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
    return dust1_fraction*dust2     
    
def to_tlast(tlast_fraction=None, zred=None, **extras):
    # convert from tlast_fraction to age of universe
    tuniv = cosmo.age(zred[0]).value
    tlast = tlast_fraction * tuniv
    # make sure we don't go below 10 Myr absolute age
    return np.clip(tlast, a_min=0.01, a_max=1)
        
def z_to_tflex(zred=None,tflex_frac=None,**extras):
    return tflex_frac * cosmo.age(zred[0]).value  
    
# -------------
# updated SFH
# -------------
def updated_logsfr_ratios_to_masses_psb(logmass=None, logsfr_ratios=None,
                                 logsfr_ratio_young=None, logsfr_ratio_old=None,
                                 tlast_fraction=None, tflex_frac=None, nflex=None, nfixed=None,
                                 agebins=None, zred=None, **extras):
    """This is a modified version of logsfr_ratios_to_masses_flex above. This now
    assumes that there are nfixed fixed-edge timebins at the beginning of
    the universe, followed by nflex flexible timebins that each form an equal
    stellar mass. The final bin has variable width and variable SFR; the width
    of the bin is set by the parameter tlast.

    The major difference between this and the transform above is that
    logsfr_ratio_old is a vector.
    """

    # clip for numerical stability
    nflex = nflex[0]; nfixed = nfixed[0]
    logsfr_ratio_young = np.clip(logsfr_ratio_young[0], -7, 7)
    logsfr_ratio_old = np.clip(logsfr_ratio_old, -7, 7)
    syoung, sold = 10**logsfr_ratio_young, 10**logsfr_ratio_old
    sratios = 10.**np.clip(logsfr_ratios, -7, 7) # numerical issues...

    # get agebins
    abins = updated_psb_logsfr_ratios_to_agebins(logsfr_ratios=logsfr_ratios,
            agebins=agebins, tlast_fraction=tlast_fraction, tflex_frac=tflex_frac, nflex=nflex, nfixed=nfixed, zred=zred, **extras)

    # get find mass in each bin
    dtyoung, dt1 = (10**abins[:2, 1] - 10**abins[:2, 0])
    dtold = 10**abins[-nfixed-1:, 1] - 10**abins[-nfixed-1:, 0]
    old_factor = np.zeros(nfixed)
    for i in range(nfixed):
        old_factor[i] = (1. / np.prod(sold[:i+1]) * np.prod(dtold[1:i+2]) / np.prod(dtold[:i+1]))
    mbin = 10**logmass / (syoung*dtyoung/dt1 + np.sum(old_factor) + nflex)
    myoung = syoung * mbin * dtyoung / dt1
    mold = mbin * old_factor
    n_masses = np.full(nflex, mbin)

    return np.array(myoung.tolist() + n_masses.tolist() + mold.tolist())


def updated_psb_logsfr_ratios_to_agebins(logsfr_ratios=None, agebins=None,
                                 tlast_fraction=None, tflex_frac=None, nflex=None, nfixed=None, zred=None, **extras):
    """This is a modified version of logsfr_ratios_to_agebins above. This now
    assumes that there are nfixed fixed-edge timebins at the beginning of
    the universe, followed by nflex flexible timebins that each form an equal
    stellar mass. The final bin has variable width and variable SFR; the width
    of the bin is set by the parameter tlast.

    For the flexible bins, we again use the equation:
        delta(t1) = tuniv  / (1 + SUM(n=1 to n=nbins-1) PROD(j=1 to j=n) Sn)
        where Sn = SFR(n) / SFR(n+1) and delta(t1) is width of youngest bin

    """
    
    # get age of universe at this z
    tuniv = cosmo.age(zred[0]).value
    
    # dumb way to de-arrayify values...
    tlast = tlast_fraction[0]*tuniv; tflex = tflex_frac[0]*tuniv
    try: nflex = nflex[0]
    except IndexError: pass
    try: nfixed = nfixed[0]
    except IndexError: pass

    # numerical stability
    logsfr_ratios = np.clip(logsfr_ratios, -7, 7)

    # flexible time is t_flex - youngest bin (= tlast, which we fit for)
    # this is also equal to tuniv - upper_time - lower_time
    tf = (tflex - tlast) * 1e9

    # figure out other bin sizes
    n_ratio = logsfr_ratios.shape[0]
    sfr_ratios = 10**logsfr_ratios
    dt1 = tf / (1 + np.sum([np.prod(sfr_ratios[:(i+1)]) for i in range(n_ratio)]))

    # translate into agelims vector (time bin edges)
    agelims = [1, (tlast*1e9), dt1+(tlast*1e9)]
    for i in range(n_ratio):
        agelims += [dt1*np.prod(sfr_ratios[:(i+1)]) + agelims[-1]]
        
    # here's our update -- previous code just copied over the fixed agebins
    # from the previous draw... but that doesn't work now if our z is changing    
    # agelims += list(10**agebins[-nfixed:,1])
    
    # instead, we need to re-calculate this based on our new zphot
    agelims += np.linspace(tflex*1e9, tuniv*1e9, nfixed+1)[1:].tolist()
    
    abins = np.log10([agelims[:-1], agelims[1:]]).T

    return abins
    
    
# --------------
# SPS Object
# --------------

def load_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    sps = FastStepBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=compute_vega_mags)                       
    return sps

    
# -----------------
# Gaussian Process
# ------------------

def load_gp(**extras):
    return None, None   

def prior_transform(u):        
    return model.prior_transform(u)

# -----------------
# Noise Model
# ------------------    
def build_noise(**extras):
    jitter = Uncorrelated(parnames = ['spec_jitter'])
    spec_noise = NoiseModel(kernels=[jitter],metric_name='unc',weight_by=['unc'])
    return spec_noise, None 

# -----------------
# SED Model
# ------------------    

def load_model(zspec=None, zphot=None, fixed_metallicity=None, add_dust=False,
               add_neb=True, luminosity_distance=None, agelims=None, objname=None,
               catfile=None, binmax=None, tquench=None, 
               tflex=None, nflex=None, nfixed=None, tflex_frac=None, tlast_max_frac=None, **extras):          
               
    """Construct a model.  This method defines a number of parameter
    specification dictionaries and uses them to initialize a
    `models.sedmodel.SedModel` object.

    :param object_redshift:
        If given, given the model redshift to this value.

    :param add_dust: (optional, default: False)
        Switch to add (fixed) parameters relevant for dust emission.

    :param add_neb: (optional, default: False)
        Switch to add (fixed) parameters relevant for nebular emission, and
        turn nebular emission on.

    :param luminosity_distance: (optional)
        If present, add a `"lumdist"` parameter to the model, and set it's
        value (in Mpc) to this.  This allows one to decouple redshift from
        distance, and fit, e.g., absolute magnitudes (by setting
        luminosity_distance to 1e-5 (10pc))
    """         
               
    # make sure we didn't put nonsense values for fractions of SFH in fixed/flex bins...
    assert (tflex_frac + tlast_max_frac) < 1.0, "tflex_frac + tlast_max_frac must be less than 1"        
               
    # --- Use the PSB SFH template. ---
    model_params = TemplateLibrary["continuity_psb_sfh"]
    
    # update SFH transforms to use slightly-modified functions above
    model_params['agebins']['depends_on'] = updated_psb_logsfr_ratios_to_agebins
    model_params['mass']['depends_on'] = updated_logsfr_ratios_to_masses_psb
    
    # set the redshift
    if zspec is not None:
        model_params['zred'] = {'N':1, 'isfree':False, 'init': zred, 'prior':priors.TopHat(mini=zspec-0.1, maxi=zspec+0.1)}
    elif zphot is not None:
        model_params['zred'] = {'N':1, 'isfree':True, 'init': zphot, 'prior':priors.TopHat(mini=0, maxi=6)}
    else: #if zspec is none + zphot is none
        model_params['zred'] = {'N':1, 'isfree':True, 'init': 2, 'prior':priors.TopHat(mini=0, maxi=6)} #set zred to value
    
    # set tflex to a fraction of age of universe at given z
    model_params['tflex_frac'] = {'N':1, 'isfree':False, 'init':tflex_frac}
    model_params['tflex'] = {'N':1, 'isfree':False, 'depends_on':z_to_tflex, 'init':tflex_frac * model_params['zred']['init']}   
    
    # tlast -- we want to set this to vary between ~10 Myr and X% of age of universe
    # easiest to do this with a transform -- sample the max tlast fraction, then transform to real values
    model_params['tlast_fraction'] = {'N':1, 'isfree':True, 'prior':priors.TopHat(mini=0.01, maxi=tlast_max_frac), 'init':0.1}
    model_params['tlast'] = {'N':1, 'isfree':False, 'depends_on':to_tlast, 'init':0.1*cosmo.age(model_params['zred']['init']).value}

    # set IMF to chabrier (default is kroupa)
    model_params['imf_type']['init'] = 1
                                            
    # make sure mass units are right
    model_params['mass_units'] = {'name': 'mass_units', 'N': 1,
                          'isfree': False,
                          'init': 'mformed'}
    
    # # massmet controls total mass and metallicity
    #model_params['massmet'] = {'name':'massmet', 'N':2, 'isfree':True, 'init':[10,0], 
    #                           'prior':MassMet(z_mini=-0.5, z_maxi=1.0, mass_mini=9.5, mass_maxi=12.5)}
    
    # # default includes a free 'logmass' -- update it based on massmet
    #model_params['logmass'] = {'N':1, 'depends_on':massmet_to_logmass, 
    #     'isfree':True, 'init':model_params['logmass']['init']}
    #
    # # metallicity-- depends on massmet prior
    #model_params['logzsol'] = {'N':1, 'depends_on':massmet_to_logzsol, 'isfree':True,
    #    'init':model_params['logzsol']['init']}
    
    # dust: kc03 dust law   
    model_params['add_dust_emission'] = {'N':1, 'isfree':False, 'init':True}
    model_params['dust_type'] = {'N':1, 'isfree':False, 'init':4}
    model_params['dust_index'] = {'N':1, 'isfree':True, 'init':0,
        'prior':priors.TopHat(mini=-1, maxi=0.4)}
    model_params['dust2']['init'] = 0.5 / (2.5*np.log10(np.e))
    model_params['dust2']['prior'] = priors.TopHat(mini=0.0,
          maxi=2.5 / (2.5*np.log10(np.e))) # factor is for AB magnitudes -> optical depth (v small correction...)
          
    # change IR SED params to match joel
    model_params['duste_gamma'] = {'N':1, 'isfree':False, 'init':0.01}
    model_params['duste_umin'] = {'N':1, 'isfree':False, 'init':1.0}
    model_params['duste_qpah'] = {'N':1, 'isfree':False, 'init':2.0}
    
    # similar as vivienne: fix young stars twice as attenuated as old stars (e.g., dust1_fraction = 1)
    model_params['dust1'] = {'N':1, 'isfree':False, 'depends_on':to_dust1, 'init':1.0}
    model_params['dust1_fraction'] = {'N':1, 'isfree':False, 'init':1.0}

    if add_neb:
        # Add nebular emission (with fixed parameters)
        model_params.update(TemplateLibrary["nebular"])

    # Now instantiate the model using this new dictionary of parameter specifications
    model = SedModel(model_params)
    
    # # for now, test a version with polynomial optimization
    # print('using polynomial optimization model')
    # model = PolySedModel(model_params)
    
    print('MODEL-- redshift: '+str(model_params['zred']['init']))
    print('MODEL-- logmass: '+str(model_params['logmass']['init']))
    print('MODEL-- logzsol: '+str(model_params['logzsol']['init']))
    print('MODEL-- dust2: '+str(model_params['dust2']['init']))
    print('MODEL-- dust_index: '+str(model_params['dust_index']['init']))
    
    print(describe(model_params))

    return model    
            
