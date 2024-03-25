'''
Given two directories of MB mcmc files and no MB mcmc files, 
will iterate through each galaxy fit and plot different 
galaxy attributes as scatterplots/histograms.

Differs from mb-nomb-fit_test.py, because this will plot dif in zred vs. dif in mass, color coded by dust
All factors that determine SFH 

python dif-scatter.py /path/to/mb/directory/ /path/to/nomb/directory/
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
import dynesty
import h5py
from matplotlib import pyplot as plt, ticker as ticker; plt.interactive(True)
from matplotlib.ticker import FormatStrFormatter
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
import seaborn as sns
import pandas as pd
import glob

from prospect.models.transforms import logsfr_ratios_to_masses_psb, psb_logsfr_ratios_to_agebins

def stepInterp(ab, val, ts):
    '''ab: agebins vector
    val: the original value (sfr, etc) that we want to interpolate
    ts: new values we want to interpolate to '''
    newval = np.zeros_like(ts) + np.nan
    for i in range(0,len(ab)):
        newval[(ts >= ab[i,0]) & (ts < ab[i,1])] = val[i]  
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

# assign directory #DONT DO IN SHERLOCK - NO SEABORN
if len(sys.argv) > 0:
    mb_directory = str(sys.argv[1]) # example: '/Users/michpark/JWST_Programs/mockgalaxies/final/z3mb/'
    nomb_directory = str(sys.argv[2]) # example: '/Users/michpark/JWST_Programs/mockgalaxies/final/z3nomb/'
  
plotdir = '/Users/michpark/JWST_Programs/mockgalaxies/scatterplots-mb-nomb/'
cosmo = FlatLambdaCDM(H0=70, Om0=.3)

from um_prospector_param_file import updated_logsfr_ratios_to_masses_psb, updated_psb_logsfr_ratios_to_agebins

directory_array = [mb_directory, nomb_directory]
dust2_array = np.empty(shape=(2, len(os.listdir(mb_directory)))) #input is always 0.2
zred_array = np.empty(shape=(2, len(os.listdir(mb_directory)))) #input is set
logmass_array = np.empty(shape=(2, len(os.listdir(mb_directory)))) #mb + nomb

for directory_index, directory in enumerate(directory_array):
    # Iterate through mcmc files in the directory
    counter = 0
    for mcmcfile in os.listdir(directory):
            mcmcfile = os.path.join(directory, mcmcfile)

            res, obs, mod = results_from("{}".format(mcmcfile), dangerous=True)
            print('----- Making plots for '+str(obs['objid']) + ' in ' + str(directory) + ' -----')
            
            gal = (np.load('/Users/michpark/JWST_Programs/mockgalaxies/obs-z3/umobs_'+str(obs['objid'])+'.npz', allow_pickle=True))['gal']
            spsdict = (np.load('/Users/michpark/JWST_Programs/mockgalaxies/obs-z3/umobs_'+str(obs['objid'])+'.npz', allow_pickle=True))['params'][()]

            sps = get_sps(res)
        
            # obtain sfh from universemachine
            um_sfh = gal['sfh'][:,1]
        
            # SFH THINGS
            # actual sfh percentiles
            flatchain = res["chain"]
            niter = res['chain'].shape[-2]
            if 'zred' in mod.theta_index:
                tmax = cosmo.age(np.min(flatchain[:,mod.theta_index['zred']])).value #matches scales
            else: 
                tmax = cosmo.age(obs['zred']).value

            if tmax > 2:
                lbt_interp = np.concatenate((np.arange(0,2,.001),np.arange(2,tmax,.01),[tmax])) 
            else:
                lbt_interp = np.arange(0,tmax+.005, .001)    
            lbt_interp[0] = 1e-9    
            age_interp = tmax - lbt_interp
        
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
            massFrac = 1 - massPercent[lbt_interp==1, 1:].flatten()[::-1]  

            #print("obs logM: " + str(obs['logM']))
            #print("um sfh integral mass: " + str(np.log10(trap(cosmo.age(gal['sfh'][:,0]).value*1e9, um_sfh))))
            #print("output original:" + str(logmass_array[1]))
            #print("output integral mass: " + str(np.log10(trap(cosmo.age(gal['sfh'][:,0]).value*1e9, sfhadjusted))))
            #outputIntegralMass = np.log10(trap(cosmo.age(gal['sfh'][:,0]).value*1e9, sfhadjusted))

            ##### SFR over meaningful timescale #####
            '''
            - 100 Myr - timescale that direct Halpha measurements are sensitive to
            - Built function to take in SFH and an averaging timescale (default 100 Myr) 
            - adds up the total mass formed in that timescale / timescale = average SFR

            timescale: most recent timescale (in Gyr)
            lbt_interp: lookback time of FULL range
            sfh: takes in SFH of FULL range
            '''
            '''
            def averageSFR(lbt, sfh, timescale = 0.1):
                # Obtain LBT + area under SFH over chosen range
                timescaleLBT = [lbt[j]*1e9 for j in range(len(lbt)) if lbt[j] <= timescale]
            
                if(len(timescaleLBT) > 1):
                    timescaleSFH = [sfh[j] for j in range(len(sfh)) if lbt[j] <= timescale]
                    timescaleMass = np.abs(trap(np.array(timescaleLBT), np.array(timescaleSFH))) # in solar masses (yr*Msun/yr)
                    return timescaleMass / (timescale*1e9)
                else: # just one value for LBT (should only occur for input SFH, output SFH is fine resolution)
                    k = len(lbt)
                    return np.abs((sfh[k-2] - sfh[k-1])/2)+sfh[k-2] # get the last two values and average over (Msun/yr)

            inputAverageSFR = averageSFR(cosmo.age(obs['zred']).value - cosmo.age(gal['sfh'][:,0]).value, um_sfh, timescale=0.1)
            outputAverageSFR = averageSFR(lbt_interp, sfrPercent[:,2], timescale=0.1)
            
            print("For " + str(obs['objid']) + ", we have input INST: " + str(um_sfh[-1]) + ", input AVE: " + str(inputAverageSFR) + ", outputINST: " + str(sfrPercent[:,2][0]) + ", output AVE: " + str(outputAverageSFR))
            '''
            
            logmass_array[directory_index][counter] = percentiles['logmass'][1]-obs['logM']
                
            '''
            #SFRs - get last value of um_sfh + 0th value of sfrpercent (both most recent values)
            ax[0,1].errorbar(um_sfh[-1], sfrPercent[:,2][0], yerr=np.vstack((sfrPercent[:,2][0] - sfrPercent[:,1][0], sfrPercent[:,3][0]-sfrPercent[:,2][0])),marker='.', markersize=10, ls='', lw=2, 
                markerfacecolor='maroon',markeredgecolor='maroon',ecolor='maroon',elinewidth=1.4, alpha=0.7) 
                
            #SFR over last 100 Myr
            ax[3,1].plot(inputAverageSFR,outputAverageSFR, marker='.', markersize=10, ls='', lw=2, markerfacecolor='maroon', markeredgecolor='maroon', alpha=0.7)
            ''' 
            
            dust2_array[directory_index][counter] = percentiles['dust2'][1]-0.2
            zred_array[directory_index][counter] = percentiles['zred'][1]-spsdict['zred']             
            counter += 1
         
# Below this point is plotting
fig, ax = plt.subplots(1,2,figsize=(9,5))

# MB plot
scatter0 = ax[0].scatter(logmass_array[0],zred_array[0], c=dust2_array[0], ec='k')
ax[0].set_title("Broad+MB")
ax[0].set_ylabel("Difference in redshift")
ax[0].axline((0, 0), slope=0, ls='--', color='black', lw=2)
ax[0].axvline(0, ls='--', color='black', lw=2)
#ax[0].set_xlim(spsdict['zred']-0.35,spsdict['zred']+0.35)
ax[0].set_xlabel(r'Difference in $log M_{stellar}$ (log $M_{sun}$)')

# NO MB plot
scatter1 = ax[1].scatter(logmass_array[1],zred_array[1], c=dust2_array[1], ec='k')
ax[1].set_title("Broad only")
ax[1].set_ylabel("Difference in redshift")
ax[1].axline((0, 0), slope=0, ls='--', color='black', lw=2)
ax[1].axvline(0, ls='--', color='black', lw=2)
#ax[1].set_xlim(spsdict['zred']-0.35,spsdict['zred']+0.35)
ax[1].set_xlabel(r'Difference in $log M_{stellar}$ (log $M_{sun}$)')

# create legend with the colors
plt.colorbar(scatter1, ax=ax[1], label="Difference in dust2", orientation="vertical") 
#legend0 = ax[0].legend(*scatter0.legend_elements(num=5),
#                    loc="best", title="Difference in dust2")
#ax[0].add_artist(legend0)

plt.tight_layout()

'''
# SFR inst - scatter - TBD
ax[0,1].axline((0, 0), slope=1, ls='--', color='black', lw=2)
ax[0,1].set_ylabel(r'Recovered $log SFR_{inst}$ (log $M_{sun}$ / yr)')
ax[0,1].set_xlabel(r'Input $log SFR_{inst}$ (log $M_{sun}$ / yr)')
ax[0,1].set_xscale('log')
ax[0,1].set_yscale('log')

# SFR - over meaningful timescale (100 Myr)
ax[3,1].axline((0, 0), slope=1., ls='--', color='black', lw=2)
ax[3,1].set_ylabel(r'Recovered $log SFR_{ave, 100 Myr}$ (log $M_{sun}$ / yr)')
ax[3,1].set_xlabel(r'Input $log SFR_{ave, 100 Myr}$ (log $M_{sun}$ / yr)')
ax[3,1].set_xscale('log')
ax[3,1].set_yscale('log')
'''


plt.show()

# make sure plot directory exists
if not os.path.exists(plotdir):
    os.mkdir(plotdir)

counter=0
filename = 'dif_{}_z3.pdf' #defines filename for all objects
while os.path.isfile(plotdir+filename.format(counter)):
    counter += 1
filename = filename.format(counter) #iterate until a unique file is made
fig.savefig(plotdir+filename, bbox_inches='tight')
  
print('saved difference scatterplot to '+plotdir+filename) 

#plt.close(fig)


        

        



             
