from prospect.models.templates import TemplateLibrary, describe
# Show basic description of all pre-defined parameter sets
TemplateLibrary.show_contents()
model_params = TemplateLibrary["parametric_sfh"]
#print(model_params)
#summary of free and fixed parameters
print(describe(model_params))

import prospect.io.read_results as reader
#res, obs, model = reader.results_from('squiggle_1675643832_mcmc.h5')
#res, obs, model = reader.results_from('squiggle_1675185491_mcmc.h5')
#res.keys()
