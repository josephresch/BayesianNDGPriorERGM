# Helper functions shared by 01_find_map.R and 02_find_mass_matrix.R.
#
# get_expected_stats — Monte Carlo estimate of E_theta[s(Y)] for the ERGM
#     y ~ <model_rhs>   (e.g. "edges + kstar(2) + gwesp(0.25, fixed=TRUE)")
# `model_rhs` must match the right-hand side used by the caller's summary()
# call, so that obs_stats and the simulated stats share a column ordering.
get_expected_stats = function(theta, network, model_rhs, nsim = 20)
{
  form   = stats::as.formula(paste("network ~", model_rhs))
  graphs = ergm::simulate_formula(form, coef = theta, nsim = nsim, output = "stats")
  return(apply(graphs, 2, mean))
}

# gradient of log Normal(mean, diag(sd^2)) prior (vectorised over dims)
get_priorgrad = function(theta, prior_mean, prior_sd)
{
  prior_var = prior_sd * prior_sd
  return(-(theta - prior_mean) / prior_var)
}

# Hessian of the log Normal prior: diagonal, -1/var per dimension.
get_priorgrad2 = function(theta, prior_mean, prior_sd)
{
  prior_var = prior_sd * prior_sd
  return(-1 / prior_var)
}

# gradient of the log posterior (score): obs_stats - E[stats] + grad log prior
get_grad = function(obs_stats, network, theta, prior_mean, prior_sd, model_rhs, nsim = 20)
{
  return(obs_stats - get_expected_stats(theta, network, model_rhs, nsim = nsim)
         + get_priorgrad(theta, prior_mean, prior_sd))
}

# Helper: load a "graphformat 1" adjacency file (as used by the C codebase)
# into a symmetric integer matrix.
load_graphformat1 = function(path)
{
  lines = readLines(path)
  stopifnot(grepl("^graphformat 1$", lines[1]))
  stopifnot(grepl("^nnodes ", lines[2]))
  nnodes = as.integer(sub("^nnodes ", "", lines[2]))
  adj_lines = lines[3:(2 + nnodes)]
  adj = do.call(rbind, lapply(adj_lines, function(s) {
    as.integer(strsplit(trimws(s), "\\s+")[[1]])
  }))
  stopifnot(nrow(adj) == nnodes, ncol(adj) == nnodes)
  stopifnot(isSymmetric(adj), all(diag(adj) == 0))
  return(adj)
}
