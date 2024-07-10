'''
Plots a medium band galaxy fit with its non-medium band counterpart for 3 examples. 
(FOR free redshift, flexible adjustments)

Plots true (input) SFH in black, Broad+MB in maroon, Broad only in navy.

Input objid of the galaxy to plot + will automatically retrieve MB + no MB versions:
run three-spectra.py 
Uses default examples below

run three-spectra.py [objid] [objid] [objid] 
Uses objids provided
'''
import numpy as np
from matplotlib import pyplot as plt, ticker as ticker; plt.interactive(True)
from matplotlib.ticker import FormatStrFormatter
import sys
import os
import pandas as pd
import glob
from prospect.io.read_results import results_from, get_sps
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=.3)

# read in a command-line argument for THE OBJID of the galaxy
if len(sys.argv) > 2:
    objid1 = str(sys.argv[1]) # amazing example: 559488394
    objid2 = str(sys.argv[2]) # sfh when quench example: 559156555
    objid3 = str(sys.argv[3]) # poorly fit example: 559296300
else: 
    objid1 = str(559488394)
    objid2 = str(559156555)
    objid3 = str(559296300) # for now
    
plotdir = '/Users/michpark/JWST_Programs/mockgalaxies/final-plots/mb-nomb/'

class Output:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
        
objid_directory = [objid1, objid2, objid3] 
fig, ax = plt.subplots(3,2,figsize=(10,10)) 
for objid_index, objid in enumerate(objid_directory):   
    # Retrieve correct dict files for mb + nomb
    root = '/Users/michpark/JWST_Programs/mockgalaxies/final-dicts/'
    for files in os.walk(root + 'z3mb/'):
            for filename in files[2]:
                if objid in filename:
                    name_path = os.path.join(root + 'z3mb/',filename)
                    outroot_mb = name_path
    for files in os.walk(root + 'z3nomb/'):
            for filename in files[2]:
                if objid in filename:
                    name_path = os.path.join(root + 'z3nomb/',filename)
                    outroot_nomb = name_path        

    print('Making plots for...')
    print('MB: '+outroot_mb)
    print('No MB: '+outroot_nomb)

    outroot_array = [outroot_mb, outroot_nomb]

    # make sure plot directory exits
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)
    if not os.path.exists(plotdir+'sfh'):
        os.mkdir(plotdir+'sfh')
    
    #check to see if duplicates exist
    counter=0
    filename = '3spectra_{}.pdf' #defines filename for all objects
    while os.path.isfile(plotdir+'sfh/'+filename.format(counter)):
        counter += 1
    filename = filename.format(counter) #iterate until a unique file is made


    #create the inset axes for the zred 
    #inset_ax[objid_index] = fig.add_axes([0.16, 0.75, 0.2, 0.12]) # without tightlayout
    inset_ax_height = [0.8415, 0.5135, 0.184]
    inset_ax_leftflank = [0.09, 0.325, 0.09]
    inset_ax = {}
    inset_ax[objid_index] = fig.add_axes([inset_ax_leftflank[objid_index], inset_ax_height[objid_index], 0.13, 0.12]) # left of legend
    
    for outroot_index, outroot in enumerate(outroot_array):
        res = np.load(outroot, allow_pickle=True)['res'][()]

        print('----- Object ID: ' + str(res.objname) + ' -----')
    
        wspec = res.spec_fit[0]
            
        ###### PLOTS IN FLAM ###### 
        def convertMaggiesToFlam(w, maggies):
            # converts maggies to f_lambda units
            # For OBS DICT photometries - use w as obs['wave_effective'] - observed wavelengths 
            c = 2.99792458e18 #AA/s
            flux_fnu = maggies * 10**-23 * 3631 # maggies to cgs fnu
            flux_flambda = flux_fnu * c/w**2 # v Fnu = lambda Flambda
            return flux_flambda

        if(outroot_index == 0): # medium bands
            #MEDIAN SPECTRA
            mb_model = ax[objid_index, 0].plot(res.spec_fit[0], convertMaggiesToFlam(res.spec_fit[0], res.spec_fit[2]),
                   lw=1.5, color='maroon', alpha=0.6, zorder=0)  
        
            # MEDIAN PHOTOMETRY
            ax[objid_index, 0].plot(res.phot_fit[0][res.obs['phot_mask']], convertMaggiesToFlam(res.phot_fit[0], res.phot_fit[2])[res.obs['phot_mask']], 
                     marker='s', markersize=10, alpha=0.6, ls='', lw=2, 
                     markerfacecolor='none', markeredgecolor='maroon', 
                     markeredgewidth=2, zorder=5, label='UNCOVER+MB model')
    
            ax[objid_index, 0].plot(res.phot_fit[0][res.obs['phot_mask']], convertMaggiesToFlam(res.phot_fit[0], res.obs['maggies'])[res.obs['phot_mask']],  
                     marker='o', markersize=7, ls='', lw=1.5, alpha=1, 
                     markerfacecolor='black', markeredgecolor='maroon', 
                     markeredgewidth=3, zorder = 5)   
                     
            norm_wl = ((wspec>6300) & (wspec<6500))
            norm = np.nanmax(convertMaggiesToFlam(res.phot_fit[0], res.obs['maggies'])[res.obs['phot_mask']])
            
            
        if(outroot_index == 1): # no medium bands  
            nomb_model = ax[objid_index, 0].plot(res.spec_fit[0], convertMaggiesToFlam(res.spec_fit[0], res.spec_fit[2]),
                       lw=1.5, color='navy', alpha=0.6, zorder=0)
                   
            ax[objid_index, 0].plot(res.phot_fit[0][res.obs['phot_mask']], convertMaggiesToFlam(res.phot_fit[0], res.obs['maggies'])[res.obs['phot_mask']], 
                         marker='o', markersize=10, ls='', lw=1.5, alpha=0.6, 
                         markerfacecolor='none', markeredgecolor='navy', 
                         markeredgewidth=2, zorder = 10)
            ax[objid_index, 0].plot(res.phot_fit[0][res.obs['phot_mask']], convertMaggiesToFlam(res.phot_fit[0], res.phot_fit[2])[res.obs['phot_mask']], 
                          marker='s', markersize=7, alpha=1, ls='', lw=2, 
                          markerfacecolor='black', markeredgecolor='navy', 
                          markeredgewidth=3, zorder=5, label='UNCOVER only model')
            ax[objid_index, 0].scatter([], [], color='black', marker='o', s=10, label=r'Observed photometry') # adds a black dot onto the legend, representing observed

        # reincorporate scaling
        ax[objid_index, 0].set_ylim((-0.1*norm, norm*3)) #top=1.5e-19 roughly
        ax[objid_index, 0].set_xlim((2e3, 1e5))
        ax[objid_index, 0].set_xlabel('Observed Wavelength [' + r'$\AA$' + ']', fontsize=11)
        ax[objid_index, 0].set_ylabel(r"F$_\lambda$ [ergs/s/cm$^2$/$\AA$]", fontsize=11) # in flam units
        ax[objid_index, 0].set_xscale('log')
    
        ax[objid_index, 0].tick_params(axis='both', which='major', labelsize=10)
    
        print('Made spectrum plot')

        ######################## SFH for FLEXIBLE continuity model ########################
        ####### OBTAIN ZRED INSET PLOT #######
        # dynesty resample_equal function
        # input: weighted zred samples
        # output: new set of samples that are all equally-weighted
        import seaborn as sns
        if(outroot_index == 0): #medium bands
            inset_ax[objid_index].axvline(x = res.obs['zred'], color='black', linestyle='--', label="$z_{spec}$")
            sns.kdeplot(res.zred_weighted, color='maroon', label="$z_{phot}$", ax = inset_ax[objid_index])
        if(outroot_index == 1): #medium bands
            sns.kdeplot(res.zred_weighted, color='navy', label="$z_{phot}$", ax = inset_ax[objid_index])
    
        inset_ax[objid_index].set_xlabel('Redshift', fontsize=10)
        inset_ax[objid_index].set_ylabel('')
        inset_ax[objid_index].set_yticks([])
        inset_ax[objid_index].set_xlim((res.obs['zred'] - 2, res.obs['zred']+ 1.2))
        inset_ax[objid_index].tick_params(labelsize=10)
    
        ######## SFH PLOTTING in LBT ##########
    
        # Convert x-axis from age to LBT
        if(outroot_index == 0): # medium band
            ax[objid_index, 1].plot(res.output_lbt, res.output_sfh[:,2], color='maroon', lw=1.5, label='UNCOVER+MB') 
            ax[objid_index, 1].fill_between(res.output_lbt, res.output_sfh[:,1], res.output_sfh[:,3], color='maroon', alpha=.3)
        if(outroot_index == 1): # no medium band
            ax[objid_index, 1].plot(res.output_lbt, res.output_sfh[:,2], color='navy', lw=1.5, label='UNCOVER only') 
            ax[objid_index, 1].fill_between(res.output_lbt, res.output_sfh[:,1], res.output_sfh[:,3], color='navy', alpha=.3)

        ax[objid_index, 1].plot(res.input_lbt, res.input_sfh, label='True SFH' if outroot_index == 0 else "", color='black', lw=1.7, marker="o") # INPUT SFH
        print('Finished SFH')

        ax[objid_index, 1].set_xlim(cosmo.age(res.gal['sfh'][:,0]).value[-1], 0)
        ax[objid_index, 1].set_yscale('log')
        ax[objid_index, 1].tick_params(axis='both', which='major', labelsize=11)
        ax[objid_index, 1].set_ylabel('Star Formation Rate [' + r'$M_{\odot} /yr$' + ']', fontsize = 11)
        ax[objid_index, 1].set_xlabel('Lookback Time [Gyr]', fontsize = 11)

        
        # Legend settings for inset plot
        # add a color coded legend once all inset plotted
        leg = inset_ax[objid_index].legend(handlelength=0, frameon=False, fontsize=11)
        for line,text in zip(leg.get_lines(),leg.get_texts()):
            text.set_color(line.get_color())

ax[0, 0].legend(loc='upper right', fontsize=11)
ax[0, 1].legend(loc='best', fontsize=11)
# save plot
fig.tight_layout() 
plt.show()
fig.savefig(plotdir+'sfh/' + filename, bbox_inches='tight')
print('saved sfh to '+plotdir+'sfh/'+filename) 

