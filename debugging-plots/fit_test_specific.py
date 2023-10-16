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
from prospect.models.transforms import logsfr_ratios_to_masses_psb, psb_logsfr_ratios_to_agebins
import fsps
import math

def stepInterp(ab, val, ts):
    '''ab: agebins vector
    val: the original value (sfr, etc) that we want to interpolate
    ts: new values we want to interpolate to '''
    newval = np.zeros_like(ts) + np.nan
    for i in range(0,len(ab)):
        newval[(ts <= ab[i,0]) & (ts > ab[i,1])] = val[i] #previously >= and <
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

# assign directory
directory = '/Users/michpark/JWST_Programs/mockgalaxies/z3_maggies_mcmc/' #FOR THE UPDATED PARAM 7/6 BATCH
#directory = '/oak/stanford/orgs/kipac/users/michpark/JWST_Programs/mockgalaxies/'
#plotdir = '/oak/stanford/orgs/kipac/users/michpark/JWST_Programs/mockgalaxies/scatterplots/'
plotdir = '/Users/michpark/JWST_Programs/mockgalaxies/scatterplots/'
cosmo = FlatLambdaCDM(H0=70, Om0=.3)

#command line argument (whole thing or just one)
#if len(sys.argv) > 0:
#    dictfile = sys.argv[1]  
#else: can test whole thing later ig
    
 
# iterate over files in that directory
# YES - DICT FILES ARE UNIQUE

# violin vs scatterplot
# the sketchiest method i cant
#mcmcCounter = len(glob.glob1(directory,"*.h5"))

# Redshift + Quench time
zred_array = [] #violin, set at 0.9743 something
#logzsol = [[0]*3] #scatter
quenchtime = [] #scatter

#dust2_df = pd.DataFrame() #violin, set at 0
#dust_index_df = pd.DataFrame() #violin, set at 0

fig, ax = plt.subplots(2,1,figsize=(6,8))

for mcmcfile in os.listdir(directory):
    if mcmcfile.startswith('z3_mb_mcmc_'):  #goes through z2p5 first
        mcmcfile = os.path.join(directory, mcmcfile)
        print('Opening '+str(mcmcfile))

        res, obs_mcmc, mod = results_from("{}".format(mcmcfile), dangerous=True) # previously obs too??
        print("{}".format(mcmcfile))

        with np.load('obs-z3/umobs_'+str(obs_mcmc['objid'])+'.npz', allow_pickle=True) as d:
            obs = d['obs'].item()
            gal = d['gal']
            spsdict = d['params'].item()

        sps = get_sps(res)

        print('----- Object ID: ' + str(obs['objid']) + ' -----')
        # obtain sfh from universemachine
        um_sfh = gal['sfh'][:,1]

        # CORNERPLOT TYPE BEAT
        #['zred','logzsol','dust2','logmass','tlast','logsfr_ratio_young','logsfr_ratio_old_1','logsfr_ratio_old_2',
        # 'logsfr_ratio_old_3','logsfr_ratios_1','logsfr_ratios_2','logsfr_ratios_3','logsfr_ratios_4','dust_index']
        
        imax = np.argmax(res['lnprobability'])
        theta_max = res['chain'][imax, :].copy()
        mspec_map, mphot_map, _ = mod.mean_model(theta_max, obs, sps=sps)

        # SFH THINGS
        # actual sfh percentiles
        flatchain = res["chain"]
        niter = res['chain'].shape[-2]
        tmax = cosmo.age(np.min(flatchain[:,mod.theta_index['zred']])).value

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
            zred = flatchain[iteration,mod.theta_index['zred']] # ZRED COLLECTED HERE
        
        zred_array.append(zred[0])
        
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

        # SFR - error bars + last bin 
        sfhadjusted = np.interp(cosmo.age(gal['sfh'][:,0]).value, age_interp, sfrPercent[:,2])
        sfhadjusted_lower = np.interp(cosmo.age(gal['sfh'][:,0]).value, age_interp, sfrPercent[:,1])
        sfhadjusted_upper = np.interp(cosmo.age(gal['sfh'][:,0]).value, age_interp, sfrPercent[:,3])

        # Find derivatives of input + output SFH, age is adjusted b/c now difference between points
        y_d_input = np.diff(um_sfh) / np.diff(cosmo.age(gal['sfh'][:,0]).value)
        y_d_output = np.diff(sfhadjusted) / np.diff(cosmo.age(gal['sfh'][:,0]).value)
        x_d = (np.array(cosmo.age(gal['sfh'][:,0]).value)[:-1] + np.array(cosmo.age(gal['sfh'][:,0]).value)[1:]) / 2

        # Use intersect package to determine where derivatives intersect the quenching threshold
        x_i, y_i = intersection_function(x_d, np.full(len(x_d), -100), y_d_input)
        x_o, y_o = intersection_function(x_d, np.full(len(x_d), -100), y_d_output)

        # GATHER REDSHIFT VALUES
        
        #PLOT QUENCH TIME
        if len(x_i) == 0 or len(x_o) == 0:
            continue
        else:
            ax[1].plot(x_i[-1], x_o[-1], marker='.', markersize=10, ls='', lw=2, 
            markerfacecolor='navy',markeredgecolor='navy',markeredgewidth=3)
            quenchtime.append([x_i[-1], x_o[-1]])
        

# PLOT THE PLOTS
#ZRED - histogram
ax[0].hist(zred_array, bins=20, range=[1.9,3.1], color='lightcoral')
ax[0].set_xlabel("Recovered redshift")
ax[0].axvline(spsdict['zred'], ls='--',color='black', lw=2, label='Input redshift: {0:.3f}'.format(spsdict['zred']))
ax[0].set_xlim(1.9,3.1)
ax[0].set_ylim(0,40)

# QUENCH TIME - Scatterplot
ax[1].axline((0, 0), slope=1., ls='--', color='black', lw=2)
ax[1].set_ylabel(r'Recovered quench time [Gyr]')
ax[1].set_xlabel(r'Input quench time [Gyr]')
ax[1].set_xlim(0,2.7)
ax[1].set_ylim(0,2.7)

plt.tight_layout()
plt.show()

# save plot 
counter=0
filename = 'scatterplot_z3_maggies_{}.pdf' #defines filename for all objects
while os.path.isfile(plotdir+filename.format(counter)):
    counter += 1
filename = filename.format(counter) #iterate until a unique file is made
fig.savefig(plotdir+filename, bbox_inches='tight')
  
print('saved scatterplot to '+plotdir+filename) 
plt.close(fig)

#np.savetxt("z2p5_nomb_redshift_output.txt", zred)
#with open("z2p5_nomb_redshift_output.txt", "w") as txt_file:
#    for row in zred:
#       txt_file.write(''.join(str(row)) + '\n')

#np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
#with open("z2p5_nomb_quenchtime_output.txt", "w") as txt_file:
#    for row in quenchtime:
#        txt_file.write(' '.join(str(a)[1:-1] for a in row) + '\n')


        



        

        



             
