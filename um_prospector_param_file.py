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
import glob
from astropy.cosmology import z_at_value

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
              #'catfile': '/Users/michpark/JWST_Programs/UNCOVER_DR1_LW_D032_catalog.fits',
              'phottable':None,                                     
              'logify_spectrum':False,
              'normalize_spectrum':False,
              # Model parameters
              'add_neb': True,
              'add_dust': True,
              # SPS parameters
              'zcontinuous': 1,
              'zred': None, #should be a free parameter
              # --- SFH parameters ---
              # 'agelims': [0.0,7.4772,8.0,8.5,9.0,9.5,9.8,10.0],
              # 'tquench': .2,
              # 'tflex': 2,
              # 'nflex': 5,
              # 'nfixed': 3,
              # --- dynesty parameters ---
              'dynesty':True,
              'nested_bound': 'multi',        # bounding method
              'nested_sample': 'rwalk',       # sampling method
              'nested_walks': 70,     # sampling gets very inefficient w/ high S/N spectra
              'nested_nlive_init': 350, # a finer resolution in likelihood space
              'nested_nlive_batch': 300,
              'nested_maxbatch': None, # was None-- changed re ben's email 5/21/19
              'nested_maxcall': 7500000, # was 5e7 -- changed to 5e6 re ben's email on 5/21/19
              'nested_maxcall_init':7500000,
              'nested_bootstrap': 20,
              'nested_dlogz_init': 0.02,
              'nested_weight_kwargs':{'pfrac': 1.0},
              'nested_target_neff':20000,
              'nested_first_update':{'min_ncall': 20000, 'min_eff': 7.5},
              'nested_stop_kwargs': {"post_thresh": 0.1}, # might want to lower this to 0.02ish once I get things working
              # --- nestle parameters ---
              'nestle_npoints': 2000,         # Number of nested sampling live points
              'nestle_method': 'multi',       # Nestle sampling method
              'nestle_maxcall': int(1e7),     # Maximum number of likelihood calls (ends even if not converged)
              }


# --------------
# OBS
# --------------
def load_obs(objid, **kwargs):
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
    
    obs['objid'] = objid
     
    return obs
    
    
# -----------------
# Helper Functions
# ------------------    
def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
    return dust1_fraction*dust2     
    
    
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

def load_model(zred=None, zphot=None, fixed_metallicity=None, add_dust=False,
               add_neb=True, luminosity_distance=None, agelims=None, objname=None,
               catfile=None, binmax=None, tquench=None, 
               tflex=None, nflex=None, nfixed=None, **extras):
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
               
    # --- Use the PSB SFH template. ---
    model_params = TemplateLibrary["continuity_psb_sfh"]

    # set the redshift
    # if we give a spec-z, we should fix it there
    # if we give a photo-z, we should let it vary but initialize it at the photo-z
    if zred is not None:
        model_params['zred'] = {'N':1, 'isfree':False, 'init': zred, 'prior':priors.TopHat(mini=zred-0.1, maxi=zred+0.1)}
    elif zphot is not None:
        model_params['zred'] = {'N':1, 'isfree':True, 'init': zphot, 'prior':priors.TopHat(mini=0, maxi=10)}
    else: #if zred is none + zphot is none
        model_params['zred'] = {'N':1, 'isfree':True, 'init': 2, 'prior':priors.TopHat(mini=0, maxi=10)} #set zred to value 
        #adjusted to 0 to 5 (where most quenched galaxies are)
    
    # set tlast 
    # maximum of tlast should be 0.3 * the age of the universe at the redshift
    # tflex is set to 0.6 * age of the universe at the redshift
    model_params['tlast']['prior'] = priors.TopHat(mini=0.01,maxi=0.3*cosmo.age(model_params['zred']['init']).value)
    model_params['tflex']['init'] = 0.6*cosmo.age(model_params['zred']['init']).value

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
            
