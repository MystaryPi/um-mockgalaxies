# um-mockgalaxies
Prospector SED fitting of mock quenched, massive, distant galaxies from UniverseMachine. 

## Usage
The example command below runs Prospector on a galaxy, can change command line options to run for different object IDs (will retrieve obs dictionary matching the ID), flexible bin fractions, with/without medium bands, etc (see run parameters). Debug mode will stop before fitting begins and display the run parameters. 

```
python um_prospector_dynesty.py --param_file=um_prospector_param_file.py --objid=559156555 --debug=True
```

To plot the spectra/photometry, star formation history, and time derivative (for quenching time):
```
python um_plot_prospector_outputs.py /path/to/mcmcfile/
```

_fit_test.py_ will produce scatterplots/histograms to compare input to recovered parameters. 

_mb-nomb-plots.py_ will compare the spectra, photometry, SFH, and quenching time recovery of a Broad+MB galaxy fit vs. its Broad-only (no MB) counterpart (for free redshift):
```
python mb-nomb-plots.py 559120319
```
Input objid of the galaxy to plot + will automatically retrieve MB + no MB versions. Plots true (input) SFH in black, Broad+MB in maroon, Broad only in navy.

Similarly, _tflex-plots.py_ will produce overlaid SFH for each ```tflex_frac``` value, given mcmc files corresponding to various ```tflex_frac``` values (0.45, 0.55, 0.65, 0.75):
```
python tflex-plots.py 559120319
```
