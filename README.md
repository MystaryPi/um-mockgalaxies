# um-mockgalaxies
Prospector SED fitting of mock quenched, massive, distant galaxies from UniverseMachine. 

## Usage
The example command below runs Prospector on a galaxy, can change command line options to run for different object IDs (will retrieve obs dictionary matching the ID), flexible bin fractions, with/without medium bands, etc (see run parameters). Debug mode will stop before fitting begins and display the run parameters. 

```
run um_prospector_dynesty.py --param_file=um_prospector_param_file.py --objid=559156555 --debug=True
```

To plot the spectra/photometry, star formation history, and time derivative (for quenching time):
```
run um_plot_prospector_outputs.py /path/to/mcmcfile/
```

_fit_test.py_ will produce scatterplots/histograms to compare input to recovered parameters. 
