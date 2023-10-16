#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import matplotlib.pyplot as plt; plt.interactive(True)
import fsps
from glob import glob
from astropy.cosmology import Planck18 as cosmo # matches UMachine cosmology
from scipy import stats
import astropy.units as u
import sedpy

# constants
lsun = 3.826e33 # erg/s

def um_array(parameters, n_gal, n_sfh): 
    # make a dtype for the results
    # TODO add phot?            
    cols = [(p, np.float64) for p in parameters]  
    cols += [('z', np.float64), ("sfh", np.float64, (n_sfh, 2))]
    dt = np.dtype(cols)
    
    # initialize and return array of zeros
    arr = np.zeros(n_gal, dtype=dt)  
    return arr
    
    
def trap(x, y):
    # basic trapezoidal integration
    return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]))/2.     

def loadUMachine(data_dir='data/', col_names=['gal_id', 'upID', 'VMpeak', 'Vmax', 'Mpeak', 'Mvir', 'Mstar', 'sfr'], remove_satellites=True):
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
    um_path = glob(os.path.join(data_dir, 'SFH*'))
    if len(um_path) == 0:
        print('No SFH files found in directory '+data_dir)
        return
    # for now, default to using the first SFH file found    
    elif len(um_path) > 1:
        print('Multiple SFH files found, using '+um_path[0])
    um_path = um_path[0]
    try:
        scale_path = glob(os.path.join(data_dir, 'scales.txt'))[0]
    except IndexError:
        print('No scale file found in directory '+data_dir)
        return
        
    # now, load in the scale file and get the redshift of each snapshot in the simulation
    # TODO: this indexing is hard-coded based on what Richie gave us-- make this cleaner...
    scales = np.loadtxt(scale_path)[:74][:,1]
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
    um_data['sfh'][:,:,1] = um_results[:, 82:156]  
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
    
    return sps    
    
    
def getMetallicity(logmass, mini=-0.5, maxi=1.0, nsample=1):
    '''
    Use Gallazzi+15 MZR+scatter to pick a metallicity for
    a galaxy at a given mass and redshift following a truncated
    normal distribution. Adapted from prospector.
    
    Parameters
    __________
    
    logmass : log(stellar mass)
    
    mini / maxi : mininum and maximum logZ
        
    
    Returns
    __________
    
    logZ : log(metallicity) ready to input to SPS model
    
    '''
    
    # this is copied from gallazzi_05_massmet used in prospector
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
       
       
    # get mean and std for this logmass
    loc = np.interp(logmass, massmet[:,0], massmet[:,1])
    scale = np.interp(logmass, massmet[:,0], massmet[:,3]) - np.interp(logmass, massmet[:,0], massmet[:,2])
    
    # transform bounds into form that truncnorm wants (eg, for standard normal distribution)
    a = (mini - loc) / scale
    b = (maxi - loc) / scale
    
    # sample from truncated normal and return
    return stats.truncnorm.rvs(a, b, loc=loc, scale=scale, size=nsample)    
    
    
def getDust(logmass, logsfr, z):
    
    # obviously this should be more complicated... but for now:
    dust2 = 0.0
    dust_frac = 1.0
    dust_index = 0.0
    
    return(dust2, dust_frac*dust2, dust_index)   

        
def getMags(sps, filternames=['jwst_f090w', 'jwst_f115w', 'jwst_f150w', 'jwst_f200w',
             'jwst_f277w', 'jwst_f335m', 'jwst_f356w', 
             'jwst_f410m', 'jwst_f444w',
             'mayall_mosaic_U_k1001', 'acs_wfc_f435w', 'subaru_suprimecam_B', 'keck_lris_g', 'acs_wfc_f606w', 'subaru_suprimecam_rp', 'keck_lris_Rs',
             'subaru_suprimecam_ip', 'acs_wfc_f775w', 'subaru_suprimecam_zp', 'acs_wfc_f850lp', 'wfc3_ir_f125w', 'subaru_moircs_J', 'wfc3_ir_f140w',
             'wfc3_ir_f160w', 'subaru_moircs_H', 'subaru_moircs_Ks',  'spitzer_irac_ch1', 'spitzer_irac_ch2', 'spitzer_irac_ch3', 'spitzer_irac_ch4' ]):
    '''get AB magnitudes in a set of filters given some SFH/t to set
    tabular SFH and a redshift.
    '''
             
    # get spectrum in cgs units
    w, spec = sps.get_spectrum(tage=-99, peraa=True) # Lsun/A
    specScaled = spec * lsun / \
        (4*np.pi*cosmo.luminosity_distance(z=sps.params['zred']).to(u.cm).value**2) # erg/s/cm^2/A
        
    # get flux in filters
    filters = sedpy.observate.load_filters(filternames)   
    magObs = np.zeros(len(filters))-99
    wObs = w * (1+sps.params['zred'])  
    for i, fil in enumerate(filters):   
        magObs[i] = fil.ab_mag(wObs, specScaled)  
        
    return(filters, magObs)    
             
    
def addNoise(filters, mags, gal, depths=None):
    
    # set limiting magnitudes -- these are default for JADES-medium
    # and GOODS-N from Skelton+14 Table 6
    if depths == None:
        depths = {'jwst_f070w': 28.8,
                'jwst_f090w': 29.4,
                'jwst_f115w':29.6,
                'jwst_f150w':29.7,
                'jwst_f200w':29.8,
                'jwst_f277w':29.4,
                'jwst_f335m':28.8,
                'jwst_f356w':29.4,
                'jwst_f410m':28.9,
                'jwst_f444w':29.1,
                'mayall_mosaic_U_k1001':26.4, 
                'acs_wfc_f435w':27.1, 
                'subaru_suprimecam_B':26.7, 
                'keck_lris_g':26.3, 
                'acs_wfc_f606w':27.4, 
                'subaru_suprimecam_rp':26.2, 
                'keck_lris_Rs':25.6,
                'subaru_suprimecam_ip':25.8, 
                'acs_wfc_f775w':26.9, 
                'subaru_suprimecam_zp':25.5, 
                'acs_wfc_f850lp':26.7, 
                'wfc3_ir_f125w':26.7, 
                'subaru_moircs_J':25.0, 
                'wfc3_ir_f140w':25.9,
                'wfc3_ir_f160w':26.1, 
                'subaru_moircs_H':24.3, 
                'subaru_moircs_Ks':24.7,  
                'spitzer_irac_ch1':24.5, 
                'spitzer_irac_ch2':24.6, 
                'spitzer_irac_ch3':22.8, 
                'spitzer_irac_ch4':22.7}   
    depths_maggies = {key:10**(-0.4*value)/5 for key, value in depths.items()}   
    unc = [depths_maggies[f.name] for f in filters]         
                
    # get magnitudes also in maggies                         
    maggies = 10**(-0.4*mags)   
    
    # perturb the maggies w/i error bars
    maggiesUnc = np.random.normal(loc=maggies, scale=unc) 
    
    # build a nice obs dict for prospector
    obs = {}
    obs['filters'] = filters
    obs['wave_effective'] = np.array([filt.wave_effective for filt in obs['filters']])
    obs['phot_mask'] = [True]*len(maggies)
    obs['maggies'] = maggiesUnc
    obs['maggies_unc'] =  unc
    obs['maggies_orig'] = maggies
    obs['wavelength'] = None
    obs['spectrum'] = None
    obs['logify_spectrum'] = False
    obs['zred'] = gal['z']
    obs['gal_id'] = gal['gal_id']
    
    return obs
    
            


if __name__ == "__main__":

    
    # load in UniverseMachine SFHs
    um = loadUMachine()
    
    # set up FSPS
    sps = loadSPS({'dust_type':4})
    
    # for each object, get a spectrum and fill in photometry
    for gal in um:
        
        print(int(gal['gal_id']))
        
        # set SFH and get total stellar mass
        # (different from UMachine value b/c latter incluldes mass loss)
        sps.set_tabular_sfh(cosmo.age(gal['sfh'][:,0]).value, gal['sfh'][:,1])
        # check up on this-- should we be doing a different integration instead?
        logM = np.log10(trap(cosmo.age(gal['sfh'][:,0]).value*1e9, gal['sfh'][:,1]))
                
        # set redshift, metallicity, dust params
        sps.params['logzsol'] = getMetallicity(logM)
        sps.params['dust2'], sps.params['dust1'], sps.params['dust_index'] = getDust(logM, um['sfr'], um['z'])
        sps.params['zred'] = gal['z']        
        
        # convolve with filters to get photometry
        filters, magObs = getMags(sps)
        
        # noise up the photometry
        obs = addNoise(filters, magObs, gal)
        
        # save
        np.savez('obs/umobs_'+str(int(gal['gal_id']))+'.npz', gal=gal, obs=obs)
    

    # specs = []
    # for sfh in sfr[:10]:
    #     sps.set_tabular_sfh(age, sfh)
    #     w, spec = sps.get_spectrum(tage=-99)
    #     specs.append(spec * 3e18/w**2)
    #
    # iwnorm = np.argmin(np.abs(w - 4050))
    #
    # plt.ion()
    #
    # sfig, saxes = pl.subplots(2,1, figsize=(8, 6))
    # for i, spec in enumerate(specs):
    #     saxes[0].plot(age, sfr[i], lw=1.2, alpha=.6)
    #     # saxes[1].plot(w, spec)
    #     renorm = np.median(specs[i][iwnorm-20:iwnorm+20])
    #     saxes[1].plot(w, specs[i] / renorm, lw=1, alpha=.6)
    # saxes[1].set_xlim((1.2e3, 6.8e3))
    # saxes[1].set_ylim((0,2))
    # saxes[1].set_xlabel("Wavelength (AA)")
    # saxes[1].set_ylabel(r"F$_\nu$ (renormalized)")
    # saxes[0].set_yscale('log')
    # saxes[0].set_xlabel('age (Gyr)')
    # saxes[0].set_ylabel('SFR')
    #
    # sfig.savefig("figures/quenching_spectra.png", dpi=400)