# Michelle's edited version

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Usage: 
    run make_spectrum.py --mediumBands True --redshift 3
    run make_spectrum.py (sets it to default options)

Options:
--mediumBands         True if medium bands (Mega Science filters) included. (Default true)
--redshift            Set redshift. (Default 3)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt; plt.interactive(True)
import fsps
from glob import glob
from astropy.cosmology import Planck18 as cosmo # matches UMachine cosmology
from scipy import stats
import astropy.units as u
from prospect.utils.obsutils import fix_obs #new?
import sedpy

# constants
lsun = 3.826e33 # erg/s

# Set up system arguments for medium bands with or without, redshift specifications
# Default: mediumBands = true, redshift = 3
mediumBands = True
redshift = 3
if("--mediumBands" in  sys.argv):
    sysArgMB = sys.argv[sys.argv.index("--mediumBands") + 1]
    if(sysArgMB == "True"): mediumBands = True
    else: mediumBands = False       
if("--redshift" in  sys.argv):
    redshift = float(sys.argv[sys.argv.index("--redshift") + 1]) 
    # account for formatting
    if(redshift.is_integer()): redshift = int(redshift)

# this is copied from gallazzi_05_massmet used in prospector
# mass ranges, metallicity values, +/- 1 standard deviation errors
massmet = np.array([[ 8.870e+00, -6.000e-01, -1.110e+00, -0.000e+00],
    [ 9.070e+00, -6.100e-01, -1.070e+00, -0.000e+00],
    [ 9.270e+00, -6.500e-01, -1.100e+00, -5.000e-02],
    [ 9.470e+00, -6.100e-01, -1.030e+00, -1.000e-02],
    [ 9.680e+00, -5.200e-01, -9.700e-01,  5.000e-02],
    [ 9.870e+00, -4.100e-01, -9.000e-01,  9.000e-02],
    [ 1.007e+01, -2.300e-01, -8.000e-01,  1.400e-01],
    [ 1.027e+01, -1.100e-01, -6.500e-01,  1.700e-01],
    [ 1.047e+01, -1.000e-02, -4.100e-01,  2.000e-01],
    [ 1.068e+01,  4.000e-02, -2.400e-01,  2.200e-01],
    [ 1.087e+01,  7.000e-02, -1.400e-01,  2.400e-01],
    [ 1.107e+01,  1.000e-01, -9.000e-02,  2.500e-01],
    [ 1.127e+01,  1.200e-01, -6.000e-02,  2.600e-01],
    [ 1.147e+01,  1.300e-01, -4.000e-02,  2.800e-01],
    [ 1.168e+01,  1.400e-01, -3.000e-02,  2.900e-01],
    [ 1.187e+01,  1.500e-01, -3.000e-02,  3.000e-01]])

def um_array(parameters, n_gal, n_sfh): 
    # make a dtype for the results           
    cols = [(p, np.float64) for p in parameters]  
    cols += [('z', np.float64), ("sfh", np.float64, (n_sfh, 2))]
    dt = np.dtype(cols)
    
    # initialize and return array of zeros
    arr = np.zeros(n_gal, dtype=dt)  
    return arr
    
    
def trap(x, y):
    # basic trapezoidal integration
    return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]))/2.     

def loadUMachine(data_dir='/Users/michpark/JWST_Programs/um-mockgalaxies/SFH_2048/', 
                 col_names=['gal_id', 'upID', 'VMpeak', 'Vmax', 'Mpeak', 'Mvir', 'Mstar', 'sfr'], 
                 remove_satellites=True):
    #data_dir='/oak/stanford/orgs/kipac/users/michpark/JWST_Programs/mockgalaxies/'
    
    ''' 
    Reads in a UMachine data file and returns the SFH for each galaxy.
    Right now, this column labeling etc is a little janky-- to fix this,
    will need to get Richie to give us more nicely-labeled output files
    from Universe Machine.
    Note: the 82-156 indexing on the SFH is *specific* to the file that
    Richie made for us of quiescent galaxies at z=1-- this is NOT the default
    UMachine output, but has been sub-sampled already.
    
    Parameters
    __________
    
    data_dir : path to where the SFH and scale files are stored
    
    col_names : names / meaning of columns in the data file. This is a super
        janky way to do this, but I don't (yet) have any control over how the
        UMachine input file is created--- eventually, want to get this as an
        array that actually has column names we can read....
    
    remove_satellites : if True, remove satellites and return only central galaxies
    
    
    Returns
    __________
    
    um_data : structured array with all of the umachine info for each galaxy
    
    '''
    # first, check to make sure the paths exist
    
    # Given redshift parameter, determines which UM SFH file to choose
    sfhFileName = 'SFH_sel_z' + str(redshift).replace('.', 'p') + '.txt'
    um_path = glob(os.path.join(data_dir, sfhFileName)) 

    '''
    if len(um_path) == 0:
        print('No SFH files found in directory '+data_dir)
        return
    # for now, default to using the first SFH file found    
    elif len(um_path) > 1:
        print('Multiple SFH files found, using '+um_path[0])
    '''
    um_path = um_path[0]
    try:
        scale_path = glob(os.path.join(data_dir, 'scales.txt'))[0]
    except IndexError:
        print('No scale file found in directory '+data_dir)
        
    # now, load in the scale file and get the redshift of each snapshot in the simulation
    # when loading in different redshift SFH_sel_z* files, different column numbers
    # stored in dictionary below

    scales_dict = {
        1: 74,
        2: 58,
        2.5: 52,
        3: 47,
        3.5: 43,
        4: 38,
        4.5: 35,
        5: 32
    }

    scales = np.loadtxt(scale_path)[:scales_dict[redshift]][:,1]

    z = 1/scales - 1 # Redshift of each snapshot
    
    # load UMachine results and initialize an array
    um_results = np.loadtxt(um_path)
    um_data = um_array(parameters=col_names, n_gal=len(um_results), n_sfh=len(z))
    
    # fill in all the columns
    for i, name in enumerate(col_names):
        um_data[name] = um_results[:,i]
    um_data['Mstar'] = np.log10(um_data['Mstar'])    

    # fill in SFH
    um_data['sfh'][:,:,0] = z 

    # sfr = SFH[:, 8+id_snap:8+2*id_snap]
    um_data['sfh'][:,:,1] = um_results[:,8+scales_dict[redshift]:8+2*scales_dict[redshift]] 
    um_data['z'] = um_data['sfh'][:,-1,0]
    
    # remove satellites if requested
    if remove_satellites:
        um_data = um_data[um_data['upID']==-1]
        
    return(um_data)    
      
    
def loadSPS(sp_params = None):
    ''' 
    Makes an SPS object to use for later
    
    Parameters
    __________
    
    sp_params : dictionary of desired initial values for the SPS
        (these values will be applied to *all* of the mock spectra)
    
    
    Returns
    __________
    
    sps : SPS object
    
    '''
    
    # initialize
    sps = fsps.StellarPopulation(zcontinuous=1)
    
    # set up tabular SFH
    sps.params["sfh"] = 3  
    
    # any other basic params
    if sp_params is not None:
        for k, v in sp_params.items():
            sps.params[k] = v   

    print(sp_params)
    return sps    
    

def getMetallicity(logmass, mini=-0.5, maxi=1.0, nsample=1):
    '''
    Use Gallazzi+15 MZR+scatter to pick a metallicity for
    a galaxy at a given mass and redshift following a truncated
    normal distribution. Adapted from prospector.

    Generate a random metallicity for each input mass value. The random metallicity should follow a gaussian 
    distribution with the mean set to the Gallazzi+05 relation, and width of the gaussian set to the 16-84th percentile 
    value on the relation. Once you re-generate the plot above you should see that most of the points fall within the error bars.
    
    Parameters
    __________
    logmass : log(stellar mass)
    mini / maxi : mininum and maximum logZ
    
    Returns
    __________
    logZ : log(metallicity) ready to input to SPS model
    
    '''
       
    # get mean and std for this logmass
    # change - ends as distances to the 16th and 84th percentile, not the values themselves
    loc = np.interp(logmass, massmet[:,0], massmet[:,1]) #mean
    min_scale = np.interp(logmass, massmet[:,0], massmet[:,1]) - np.interp(logmass, massmet[:,0], massmet[:,2]) 
    max_scale = np.interp(logmass, massmet[:,0], massmet[:,3]) - np.interp(logmass, massmet[:,0], massmet[:,1]) 
    scale = (min_scale + max_scale)/2
    
    # transform bounds into form that truncnorm wants (eg, for standard normal distribution) #standard deviations
    a = (mini - loc)/min_scale
    b = (maxi - loc)/max_scale
    
    # sample from truncated normal and return    
    return stats.truncnorm.rvs(a,b,loc,scale,size=nsample)
    
def getDust(logmass, logsfr, z):
    
    # obviously this should be more complicated... but for now:
    # talk to richie about dust in models
    dust2 = 0.2
    dust_frac = 1.0
    dust_index = 0.0
    
    return(dust2, dust_frac*dust2, dust_index)   

def convertMaggiesToFlam(maggies):
    # converts maggies to f_lambda units, for OBS DICT photometries
    c = 2.99792458e18 #AA/s
    flux_fnu = maggies * 10**-23 * 3631 # maggies to cgs fnu
    flux_flambda = flux_fnu * c/obs['wave_effective'][obs['phot_mask']]**2 # v Fnu = lambda Flambda
    return flux_flambda

        
def getMags(sps, filternames=['jwst_f115w','jwst_f150w','jwst_f200w','jwst_f277w','jwst_f356w','jwst_f410m','jwst_f444w','jwst_f070w','jwst_f090w','jwst_f140m','jwst_f162m','jwst_f182m','jwst_f210m','jwst_f250m','jwst_f300m',
    'jwst_f335m','jwst_f360m','jwst_f410m','jwst_f430m','jwst_f460m','jwst_f480m']):
    '''get AB magnitudes in a set of filters given some SFH/t to set
    tabular SFH and a redshift.
    '''
            
    # get spectrum in cgs units
    pc = 3.085677581467192e18  # in cm
    z = sps.params['zred']

    tuniv = cosmo.age(z).value # in Gyr 
    w, spec = sps.get_spectrum(tage=tuniv, peraa=True) # Lsun/AA 
    spec = spec * lsun / (4 * np.pi * (cosmo.luminosity_distance(z=z).value*1e6*pc)**2 * (1+z)) # erg/s/cm^2/AA (f_lamda))

    # get flux in filters
    filters = sedpy.observate.load_filters(filternames)
    magObs = np.zeros(len(filters))-99
    wObs = w * (1+sps.params['zred'])

    for i, fil in enumerate(filters):
        magObs[i] = fil.ab_mag(wObs, spec) #magObs in AB magnitude #input: AA, cgs Flambda units
  
    '''
    spec_new = spec * 3.34*10**4 * w**2 / 3631 
    plt.plot(wObs, spec, label='Galaxy spectrum',lw=1.5, color='grey', alpha=0.7, zorder=10)    
    plt.xscale('log')
    plt.xlim(1e3, 1e5)
    plt.ylim(1e-10, 1e-6)
    plt.yscale('log')
    plt.xlabel("Observed wavelength (AA)")
    plt.ylabel(r"F$_\nu$ in maggies")
    plt.show()
    '''

    return(filters, magObs)
            

def build_obs(filters, mags, gal, depths): #should be build_obs technically
    # set limiting magnitudes (value detectable at 5 sigma, faintest depth reliably detect) -- these are default for JADES-medium
    # and GOODS-N from Skelton+14 Table 6
    if depths == None:
        # Depths from UNCOVER vs UNCOVER + mega science JWST
        depths = {'jwst_f115w':30.05,
                    'jwst_f150w':30.18,
                    'jwst_f200w':30.12,
                    'jwst_f277w':29.75,
                    'jwst_f356w':29.79,
                    'jwst_f410m':29.03,
                    'jwst_f444w':29.25,
                    'jwst_f070w': 28.9, 
                    'jwst_f090w':29.6, 
                    'jwst_f140m': 28.9, 
                    'jwst_f162m': 29, 
                    'jwst_f182m':29.2, 
                    'jwst_f210m':29, 
                    'jwst_f250m':28.3, 
                    'jwst_f300m':28.7, 
                    'jwst_f335m':28.8, 
                    'jwst_f360m':28.8, 
                    'jwst_f410m':28.8, 
                    'jwst_f430m':28.1, 
                    'jwst_f460m':27.8, 
                    'jwst_f480m':27.8}
            
    # divided by 5 because these are 5 sigma errors      
    depths_maggies = {key:10**(-0.4*value)/5 for key, value in depths.items()}   
    unc = [depths_maggies[f.name] for f in filters] # enforce sorted order by filters
                
    # get magnitudes also in maggies 
    # mags are in AB magnitude        
    maggies = 10**(-0.4*mags) 
    
    # perturb the maggies w/i error bars
    maggiesUnc = np.random.normal(loc=maggies, scale=unc) 
    
    # build a nice obs dict for prospector
    obs = {}
    obs['filters'] = filters
    obs['wave_effective'] = np.array([filt.wave_effective for filt in obs['filters']]) 
    if(mediumBands == True):
        obs['phot_mask'] = [True]*len(maggies) #always true b/c our fake data is all good
    else:
        # No medium bands - exclude the bands that we don't need
        obs['phot_mask'] = [True, True, True, True, True, True, True, False, False, False, 
                False, False, False, False, False, False, False, False, False, False, False]
    
    # make a mask -- photometry exists & errors are positive
    #obs['phot_mask'] = np.logical_and(np.isfinite(obs['maggies']),np.array(obs['maggies_unc']))
    obs['maggies'] = np.array(maggiesUnc)
    obs['maggies_unc'] =  np.array(unc)
    obs['maggies_orig'] = np.array(maggies)
    obs['wavelength'] = None
    obs['spectrum'] = None
    obs['logify_spectrum'] = False
    obs['zred'] = gal['z']
    obs['objid'] = gal['gal_id']

    # enforce an error floor (5% assumed)
    too_small = (obs['maggies_unc'] / obs['maggies']) < 0.05
    obs['maggies_unc'][too_small] = obs['maggies'][too_small] * 0.05

    return obs


if __name__ == "__main__":
    # load in UniverseMachine SFHs
    um = loadUMachine()
    
    # set up FSPS
    sps = loadSPS({'dust_type':4}) #,'zred':um['z'][0],'redshift_colors':1

    # for each object, get a spectrum and fill in photometry
    for gal in um[0:1]: 
        print("Object ID: " + str(gal['gal_id']))
        # set SFH and get total stellar mass
        # (different from UMachine value b/c latter incluldes mass loss)

        sps.set_tabular_sfh(cosmo.age(gal['sfh'][:,0]).value, gal['sfh'][:,1]) 

        # check up on this-- should we be doing a different integration instead?
        logM = np.log10(trap(cosmo.age(gal['sfh'][:,0]).value*1e9, gal['sfh'][:,1]))
        print("logmass: " + str(logM)) # slightly dif from gal['logM'] - mass loss?

        # set redshift, metallicity, dust params
        sps.params['logzsol'] = getMetallicity(logM, mini=-0.5, maxi=1.0, nsample=1)
        sps.params['dust2'], sps.params['dust1'], sps.params['dust_index'] = getDust(logM, um['sfr'], um['z'])
        sps.params['zred'] = gal['z'] 

        #print("redshift: " + str(sps.params['zred']))
        
        # convolve with filters to get photometry
        filters, magObs = getMags(sps) #observed (1+z factor)

        obs = build_obs(filters, magObs, gal, depths=None) 
        obs['logM'] = logM
        obs = fix_obs(obs)

        # save
        #np.savez('obs/umobs_'+str(int(gal['gal_id']))+'.npz', gal=gal, obs=obs, params=sps.params) #before=**spsdict
        
        # gal type is np void object

        #### PLOTTING ####
        pc = 3.085677581467192e18  # in cm
        lsun = 3.846e+33
        
        z = sps.params['zred']
        w, spec = sps.get_spectrum(tage=cosmo.age(z).value, peraa = True) # Lsun/AA
        spec = spec * lsun / (4 * np.pi * (cosmo.luminosity_distance(z=z).value*1e6*pc)**2 * (1+z)) # erg/s/cm^2/AA (f_lamda))

        sfig, saxes = plt.subplots(2,1, figsize=(8, 6))
        saxes[0].plot(cosmo.age(gal['sfh'][:,0]).value, gal['sfh'][:,1], lw=1.2, alpha=.6)
        
        ####### Plots in Flambda Units ########
        # Created new convertMaggiesToFlam function that will do a conversion to flambda given maggies (ALR inputs wave_eff)
        saxes[1].plot(w*(1+obs['zred']), spec, label='Galaxy spectrum',lw=1.5, color='grey', alpha=0.7, zorder=10)    
        saxes[1].errorbar(obs['wave_effective'][obs['phot_mask']], convertMaggiesToFlam(obs['maggies_orig'][obs['phot_mask']]), label='Intrinsic photometry', marker='s', markersize=10, 
           alpha=0.8, ls='', lw=3, markerfacecolor='none', markeredgecolor='green', markeredgewidth=3)
        saxes[1].errorbar(obs['wave_effective'][obs['phot_mask']], convertMaggiesToFlam(obs['maggies'][obs['phot_mask']]), yerr=convertMaggiesToFlam(obs['maggies_unc'][obs['phot_mask']]), label='Observed photometry',
            ecolor='red', marker='o', markersize=10, ls='', lw=3, alpha=0.8, markerfacecolor='none', markeredgecolor='black', markeredgewidth=3)

        saxes[1].set_xscale('log')
        saxes[1].set_xlim(1e3, 1e5)
        saxes[1].legend(loc='best', fontsize=10)
        #saxes[1].set_ylim(1e-10, 1e-6)
        #saxes[1].set_yscale('log')
        saxes[1].set_xlabel("Observed wavelength (AA)")
        #text = saxes[1].text(1300, 10**-17.9, "Mass: " + str("%.3f" % logM[0]) + "\nMetallicity: " + str("%.3f" % logZ[0])) 
        # z = 0.974 
        saxes[1].set_ylabel(r"F$_\lambda$ in ergs/s/cm$^2$/AA")
        #saxes[1].set_ylabel(r"F$_\nu$ in maggies")
            

        saxes[0].set_yscale('log')
        saxes[0].set_xlim(0,cosmo.age(z).value)
        saxes[0].set_xlabel('Age (Gyr)')
        saxes[0].set_ylabel('Star Formation Rate (Msun/yr)')
        sfig.tight_layout()
        
        plt.show()
        
        '''
        
        ##### TEMPORARY - plot filters
        # establish bounds
        fig = plt.figure(figsize=(8,4))
        counter = 0
        for f in obs['filters']:
            w, t = f.wavelength.copy(), f.transmission.copy()
            if counter < 7:
                plt.plot(w, t*3, lw=2, color="royalblue")
                print(counter)
            else:
                plt.plot(w, t, lw=2, color="green")
            counter+=1

        #Legend labels
        plt.plot(w[0], t[0], lw=2, color="royalblue",label="UNCOVER bands")
        plt.plot(w[-1], t[-1], lw=2, color="green",label="Mega Science bands")

        # prettify
        plt.xlabel('Wavelength [' + r'$\AA$' + ']')
        plt.legend(fontsize=14)
        plt.ylim(8e-2,7)
        plt.xlim(1000, 100000)
        plt.xscale("log")
        plt.yscale("log")
        plt.yticks([])
        plt.tight_layout()
        plt.savefig("filters.png", dpi=400)
        
        '''
    
    """
    # MASS VS METALLICITY (compare with gallazzi)
    #fig1 = plt.figure(1)
    plt.scatter(logM, logZ, color='black')
    #errors
    plt.scatter(mass_massmet, metallicity_massmet, color='red')
    plt.scatter(mass_massmet, lowererror_massmet, color='red', marker='^')
    plt.scatter(mass_massmet, uppererror_massmet, color='red', marker='v')
    plt.xlabel("Total mass (solar masses)", x=1, ha='right', fontsize = 11)
    plt.ylabel("Metallicity", y=1, ha='right', fontsize=11)

    # Extract values from massmets
    plt.scatter(mass_massmet, metallicity_massmet, color='black')
    plt.xlabel("Mass bins (solar masses)", x=1, ha='right', fontsize = 11)
    plt.ylabel("Expected metallicity from Gallazzi 2005", y=1, ha='right', fontsize=11)

    # REDSHIFT AND AGE vs SFR
    fig1 = plt.figure(1)
    plt.scatter(redshift, sfr, color='black')
    plt.plot(redshift, sfr)
    plt.xlabel("Redshift", x=1, ha='right', fontsize = 11)
    plt.ylabel("Star formation rate (solar masses/year)", y=1, ha='right', fontsize=11)

    fig2 = plt.figure(2)
    plt.scatter(time_beginning, sfr, color='black')
    plt.plot(time_beginning,sfr)
    plt.xlabel("Time since beginning of universe (Gyr)", x=1, ha='right', fontsize = 11)
    plt.ylabel("Star formation rate (solar masses/year)", y=1, ha='right', fontsize=11)
 
    # Normal dist histograms for metallicities
    print(np.matrix(test_gallazi_metallicities))
    plt.hist(test_gallazi_metallicities, bins=15)

    """
    
    '''
    sfrtest = plt.figure()
    for sfh in sfr:
        plt.plot(time_beginning, sfr, lw=1.2, alpha=.6)
    plt.yscale('log')
    plt.xlim(0,cosmo.age(2.5).value)
    plt.xlabel('Time since beginning of the universe (Gyr)')
    plt.ylabel('SFR')
    plt.tight_layout()
    plt.show()
    '''
    
    
    #sps.set_tabular_sfh(time_beginning, sfr) #set metallicity? Z=logZ[0].item()
    #w, spec = sps.get_spectrum(tage=cosmo.age(obs['zred']).value) #spec in Lsun/Hz #originall tage=-99
    #iwnorm = np.argmin(np.abs(w - 4050))
    
    # Plot double plot of SFR vs time AND galaxy spectra in physical units
    '''
    sfig, saxes = plt.subplots(2,1, figsize=(8, 6))
    saxes[0].plot(time_beginning, sfr, lw=1.2, alpha=.6)

    #Also plot PHOTOMETRY!!
    #solar luminosities/hz (spec) * erg/s / cm^2 (specscaled) * 10**23 * 3.631 in jankskys / 3631 to maggies
    saxes[1].plot(w,spec * lsun / (4*np.pi*cosmo.luminosity_distance(z=sps.params['zred']).to(u.cm).value**2) * 10**20, label='Galaxy spectrum',lw=1.5, color='grey', alpha=0.7, zorder=10)    
    
    #Also plot the intrinsic (maggies orig) + obs (maggies + maggies_unc) photometry
    saxes[1].errorbar(obs['wave_effective']/(1+obs['zred']), obs['maggies_orig'], label='Intrinsic photometry', marker='s', markersize=10, 
        alpha=0.8, ls='', lw=3, markerfacecolor='none', markeredgecolor='green', markeredgewidth=3)
    saxes[1].errorbar(obs['wave_effective']/(1+obs['zred']), obs['maggies'], yerr=obs['maggies_unc'], label='Observed photometry',
        ecolor='red', marker='o', markersize=10, ls='', lw=3, alpha=0.8, markerfacecolor='none', markeredgecolor='black', markeredgewidth=3)

    #saxes[1].set_xlim((1.2e3, 6.8e3))
    saxes[1].set_xscale('log')
    #saxes[1].set_xlim(1e3, 1e5)
    saxes[1].legend(loc='best', fontsize=10)
    #saxes[1].set_ylim(1e-12, 1e-7)
    saxes[1].set_yscale('log')
    saxes[1].set_xlabel("Rest-frame wavelength (AA)")
    #text = saxes[1].text(1300, 10**-17.9, "Mass: " + str("%.3f" % logM[0]) + "\nMetallicity: " + str("%.3f" % logZ[0])) 
    # z = 0.974 
    # can remove easily with text.remove()
    #saxes[1].set_ylabel(r"F$_\lambda$ in ergs/s/cm$^2$/AA")
    saxes[1].set_ylabel(r"F$_\nu$ in maggies")
        

    saxes[0].set_yscale('log')
    saxes[0].set_xlim(0,cosmo.age(2.5).value)
    saxes[0].set_xlabel('Time since beginning of the universe (Gyr)')
    saxes[0].set_ylabel('SFR')
    sfig.tight_layout()
    
    #plt.show()
    '''
    
    #fig.savefig("figures/quenching_spectra.png", dpi=400)
