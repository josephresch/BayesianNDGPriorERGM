# Helper functions shared by 01_find_map.R and 02_find_mass_matrix.R.
#
# The model is hardcoded as edges + kstar(2) because this study fits only that
# specification. The Kapferer studies pass the model in as `model_rhs` instead.

# Monte Carlo estimate of E_theta[s(Y)]
get_expected_stats = function(theta, network)
{
  graphs = simulate(network ~ edges + kstar(2), coef=theta, nsim=20, statsonly=TRUE)
  return(apply(graphs,2,mean))
}

# function to get the gradient of a gaussian prior
get_priorgrad = function(theta, prior_mean, prior_sd)
{
	prior_var = prior_sd*prior_sd
	return(-(theta-prior_mean)/(prior_var))
}

# function to get the hessian of a gaussian prior
get_priorgrad2 = function(theta, prior_mean, prior_sd)
{
	prior_var = prior_sd*prior_sd
	return(-1/(prior_var))
}

# function to get the gradient of the logposterior
get_grad = function(obs_stats, network, theta, prior_mean, prior_sd)
{
  return(obs_stats - get_expected_stats(theta, network) + get_priorgrad(theta, prior_mean, prior_sd))
}
