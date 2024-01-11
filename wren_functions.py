### Functions from Wren's code

import numpy as np

def modified_logsfr_ratios_to_agebins(logsfr_ratios=None, agebins=None, 
                               tquench=None, tflex=None, nflex=None, nfixed=None, **extras):
    """This transforms from SFR ratios to agebins by assuming a constant amount
    of mass forms in each bin agebins = np.array([NBINS,2])

    use equation:
        delta(t1) = tuniv  / (1 + SUM(n=1 to n=nbins-1) PROD(j=1 to j=n) Sn)
        where Sn = SFR(n) / SFR(n+1) and delta(t1) is width of youngest bin
    
    Edited for new PSB model: youngest bin is 'tquench' long, and it is 
    preceded by 'nflex' young flexible bins, then 'nfixed' older fixed bins
    
    """
                           
    # numerical stability
    logsfr_ratios = np.clip(logsfr_ratios, -7, 7)
    
    # flexible time is t_flex - youngest bin (= tquench, which we fit for)
    # this is also equal to tuniv - upper_time - lower_time
    tf = (tflex - tquench) * 1e9
    
    # figure out other bin sizes
    n_ratio = logsfr_ratios.shape[0]
    sfr_ratios = 10**logsfr_ratios
    dt1 = tf / (1 + np.sum([np.prod(sfr_ratios[:(i+1)]) for i in range(n_ratio)]))
    
    # hopefully this is fixed by the new likelihood function?
    # # if dt1 is very small, we'll get an error that 'agebins must be increasing'
    # # to avoid this, anticipate it and return an 'unlikely' solution-- put all the mass
    # # in the first bin. this is complicated by the fact that we CAN'T return two
    # # values from this function or FSPS crashes. instead, return a very weirdly
    # # specific and exact value that we'll never get to otherwise...
    # if dt1 < 1e-4:
    #     agelims = [0, 8, 8.1]
    #     for i in range(n_ratio):
    #         agelims += [agelims[-1]+.1]
    #     agelims += list(agebins[-nfixed:,1])
    #     abins = np.array([agelims[:-1], agelims[1:]]).T
    #     return abins

    # translate into agelims vector (time bin edges)
    agelims = [1, (tquench*1e9), dt1+(tquench*1e9)]
    for i in range(n_ratio):
        agelims += [dt1*np.prod(sfr_ratios[:(i+1)]) + agelims[-1]]
    agelims += list(10**agebins[-nfixed:,1]) 
    abins = np.log10([agelims[:-1], agelims[1:]]).T

    return abins

def modified_logsfr_ratios_to_masses_flex(logmass=None, logsfr_ratios=None,
                                 logsfr_ratio_young=None, logsfr_ratio_old=None,
                                 tquench=None, tflex=None, nflex=None, nfixed=None, 
                                 agebins=None, **extras):
    
    # clip for numerical stability
    logsfr_ratio_young = np.clip(logsfr_ratio_young, -7, 7)
    logsfr_ratio_old = np.clip(logsfr_ratio_old, -7, 7)
    syoung, sold = 10**logsfr_ratio_young, 10**logsfr_ratio_old
    sratios = 10.**np.clip(logsfr_ratios, -7, 7) # numerical issues...

    # get agebins
    abins = modified_logsfr_ratios_to_agebins(logsfr_ratios=logsfr_ratios,
            agebins=agebins, tquench=tquench, tflex=tflex, nflex=nflex, nfixed=nfixed, **extras)
            
    # if qflag=0, we bonked-- put all the mass in the oldest bin as a 'unfavorable' solution
    if np.array_equal(abins[:-nfixed, 1], np.arange(8, 8+.1*(len(abins)-nfixed), .1)):
        fakemasses = np.zeros(len(agebins))
        fakemasses[-1] = 10**logmass
        return fakemasses       
    
    # get find mass in each bin
    dtyoung, dt1 = (10**abins[:2, 1] - 10**abins[:2, 0])
    dtold = 10**abins[-nfixed-1:, 1] - 10**abins[-nfixed-1:, 0]
                    #(10**abins[-2:, 1] - 10**abins[-2:, 0])
    old_factor = np.zeros(nfixed)
    for i in range(nfixed): 
        old_factor[i] = (1. / np.prod(sold[:i+1]) * np.prod(dtold[1:i+2]) / np.prod(dtold[:i+1]))                    
    # sold_factor = 1. / np.array([np.prod(sold[:i+1]) for i in range(nfixed)])
    # mbin = (10**logmass) / (syoung*dtyoung/dt1 + np.sum(sold_factor*dtold[1:]/dtold[:-1]) + nflex)
    mbin = 10**logmass / (syoung*dtyoung/dt1 + np.sum(old_factor) + nflex)
    myoung = syoung * mbin * dtyoung / dt1
    mold = mbin * old_factor #sold_factor * mbin * dtold[1:] / dtold[:-1]
    n_masses = np.full(nflex, mbin)

    return np.array([myoung] + n_masses.tolist() + mold.tolist())
