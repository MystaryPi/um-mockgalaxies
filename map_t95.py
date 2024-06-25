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
from matplotlib import cm, colors

# set up cosmology
cosmo = FlatLambdaCDM(H0=70, Om0=.3)
plt.interactive(True)

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

# # set some params
# tflex=2
# nflex=5
# nfixed=3
#Edited for new PSB model: youngest bin is 'tquench' long, and it is preceded by 'nflex' young flexible bins, then 'nfixed' older fixed bins

##############################################################################  
    
# grab results (dictionary), the obs dictionary, and our corresponding models
res, obs, mod = results_from("{}".format(outroot), dangerous=True) # This displays the model parameters too
print("{}".format(outroot))
sps = get_sps(res)

#obs = (np.load('obs-z3/umobs_'+str(obs_mcmc['objid'])+å'.npz', allow_pickle=True))['obs']
gal = (np.load('/Users/michpark/JWST_Programs/mockgalaxies/obs-z1/umobs_'+str(obs['objid'])+'.npz', allow_pickle=True))['gal']
spsdict = (np.load('/Users/michpark/JWST_Programs/mockgalaxies/obs-z1/umobs_'+str(obs['objid'])+'.npz', allow_pickle=True))['params'][()]


print('Object ID: ' + str(obs['objid']))

print('Loaded results')

# make sure plot directory exits
if not os.path.exists(plotdir):
    os.mkdir(plotdir)
    
t95_total = np.zeros(15598) # manually put in the size
##############################################################################  
for j in range(1,len(res['lnprobability'])): #range(1,50)
    print("### index = " + str(j) + " ###")
    truth_array = [gal['z'], spsdict['logzsol'], spsdict['dust2'], obs['logM'], 0, 0, 0, 0, 0, 0, 0, 0, 0, spsdict['dust_index']]
    #imax = np.argmax(res['lnprobability'])
    imax = res['lnprobability'].argsort()[-1*j] # finds the ith most likely value
    theta_max = res['chain'][imax, :].copy()

    # Make plot of data and model
    c = 2.99792458e18

    ###### PLOTS IN FLAM ###### 
    def convertMaggiesToFlam(w, maggies):
        # converts maggies to f_lambda units
        # For OBS DICT photometries - use w as obs['wave_effective'] - observed wavelengths 
        c = 2.99792458e18 #AA/s
        flux_fnu = maggies * 10**-23 * 3631 # maggies to cgs fnu
        flux_flambda = flux_fnu * c/w**2 # v Fnu = lambda Flambda
        return flux_flambda

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
        
        # temporarily set to EQUAL
        #logr = np.array([0,0,0,0])
        #logr_young = np.array([0])
        #logr_old = np.array([1,0,0])
    
        # MAP SFH?
        logr_young = np.array([theta_max[4]])
        logr_old = theta_max[5:8]
        logr = theta_max[8:12]
    
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
    dt = ((lbt_interp - np.insert(lbt_interp,0,0)[:-1])) * 1e9
    t95draw = np.array([])
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
        
        x_rec_t95, y = intersection_function(lbt_interp, np.full(len(masscum_interp[i,:]), 0.95), masscum_interp[i,:])
        t95draw = np.append(t95draw, x_rec_t95[0])
        
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
    massPercent = np.array([quantile(masscum_interp[:,i], [16,50,84], weights=res.get('weights', None)) 
        for i in range(masscum_interp.shape[1])])

    # all percentiles...
    percentiles = get_percentiles(res, mod) # stores 16 50 84 percentiles for dif parameters
    print(percentiles) # prints percentiles

    # mass fraction in the last Gyr
    massFrac = 1 - massPercent[lbt_interp==1, 1:].flatten()[::-1]  

    #### COMPARING posterior WITH STUDENTT DISTRIBUTION (prior)
    input_massFracSFR = np.array([])
    trapsfh = um_sfh
    traplbt = (cosmo.age(obs['zred']).value - cosmo.age(gal['sfh'][:,0]).value)
    for n in range(len(trapsfh)-1):
        traplbtprep = np.array([traplbt[n], traplbt[n+1]])
        trapsfhprep = np.array([trapsfh[n], trapsfh[n+1]])
        if(len(input_massFracSFR) == 0): # accumulate mass
            input_massFracSFR = np.append(input_massFracSFR, trap(traplbtprep*10**9,trapsfhprep))
        else:
            input_massFracSFR = np.append(input_massFracSFR, input_massFracSFR[-1] + trap(traplbtprep*10**9,trapsfhprep))

    input_massFracSFR = -input_massFracSFR
    inputmassPercent = input_massFracSFR/input_massFracSFR[len(input_massFracSFR)-1]
    inputmassLBT = (cosmo.age(obs['zred']).value - cosmo.age(gal['sfh'][:,0]).value)[1:len(cosmo.age(obs['zred']).value - cosmo.age(gal['sfh'][:,0]).value)]

    # t50, t90 - intersection function
    #x_in_t50, y = intersection_function(inputmassLBT, np.full(len(inputmassPercent), 0.5), inputmassPercent)
    x_in_t95, y = intersection_function(inputmassLBT, np.full(len(inputmassPercent), 0.95), inputmassPercent)

    # t50, t90 - intersection function
    
    #x_rec_t95, y = intersection_function(lbt_interp, np.full(len(massPercent[:,2]), 0.95), massPercent[:,2])
    
    t95_total = np.vstack((t95_total, t95draw))
    #print("t95: " + str(x_rec_t95[0]))

#np.savetxt('alldraws.txt', t95_total)
    
#### PLOTTING FROM A TXT FILE WITH THE t95 DRAWS ####
plt.interactive(True)

t95_total = np.loadtxt('thousanddraws.txt')
print(t95_total)

# (if you decide to just run this in ipython)
# grab results (dictionary), the obs dictionary, and our corresponding models
outroot = "/Users/michpark/JWST_Programs/mockgalaxies/final/z1mb/z1mb_mcmc_18_121086658_1706898391_mcmc.h5"
res, obs, mod = results_from("{}".format(outroot), dangerous=True) # This displays the model parameters too
sps = get_sps(res)

gal = (np.load('/Users/michpark/JWST_Programs/mockgalaxies/obs-z1/umobs_'+str(obs['objid'])+'.npz', allow_pickle=True))['gal']
spsdict = (np.load('/Users/michpark/JWST_Programs/mockgalaxies/obs-z1/umobs_'+str(obs['objid'])+'.npz', allow_pickle=True))['params'][()]

print('Object ID: ' + str(obs['objid']))

print('Loaded results')

# plot histograms for evenly spaced numbers of the first n draws 
cmap = plt.get_cmap('viridis')
for draw_no in np.arange(len(t95_total)/5, len(t95_total)+len(t95_total)/5, len(t95_total)/5):
    fig, ax = plt.subplots()  
    draw_no = int(draw_no)
    t_indices = np.arange(0, draw_no, 1)
    
    # colorbar to represent the draw number
    color_gradients = cmap(t_indices)  
    norm = colors.Normalize(t_indices[0], t_indices[-1])

    for index in range(1, draw_no): # skip first row of zeroes
        plt.hist(dynesty.utils.resample_equal(t95_total[index], weights=res.get("weights", None)), bins = np.linspace(0, 1, 30), histtype='step', density=True, color=cmap(norm(t_indices[index]))) # prior samples

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Draw number")

    plt.title("t95 values for first " + str(draw_no) + " draws, " + str(obs['objid']))
    plt.xlabel('t95')
    plt.ylabel('Frequency')
    #plt.axvline(x_in_t95[0], ls='--', label='Input SFH t95') #1.59766483 for the given object
    plt.axvline(1.59766483, ls='--', label='Input SFH t95') # Hardcoded this in for now
    plt.legend()
    plt.show()


'''
plt.figure()
plt.title("t95 values for the 50 most likely draws, " + str(obs['objid']))
plt.hist(t95_total, bins = np.linspace(0, 4, 100))
plt.axvline(x_in_t95[0], ls='--', label='Input SFH t95')
#plt.xlim((0,1))
plt.xlabel('t95')
plt.show()
'''
