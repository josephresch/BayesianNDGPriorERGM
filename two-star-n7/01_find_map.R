# 01_find_map.R
# Locate the mode of the ERGM posterior for the g7 network (edges + kstar(2)).
# Run from this study folder; every path below is relative to it.
#
# The resulting MAP is copied by hand into 02_find_mass_matrix.R.

# Load libraries
require(ergm)
source("ergm_helpers.R")

# Load g7 network: reconstruct the observed graph from g7 data
load("data/g7.RData")
load("data/g7kstars.RData")
obs = c(7, 14)  # 7 edges, 14 two-stars

set.seed(10)
select_g7 = sample(which(g7kstars[,2]/2==obs[1] & g7kstars[,3]==obs[2]), 1)
select_id1 = floor(as.numeric(names(g7[select_g7,]))/10)
select_id2 = as.numeric(names(g7[select_g7,]))%%10
adj = diag(0, 7)
for(i in 1:ncol(g7)) {
  adj[select_id1[i], select_id2[i]] = g7[select_g7,][i]
  adj[select_id2[i], select_id1[i]] = g7[select_g7,][i]
}
g7_net = network(adj, directed=FALSE)
obs_stats = as.numeric(summary(g7_net ~ edges + kstar(2)))
cat("Observed stats:", obs_stats, "\n")

# set the prior (zero mean, diagonal covariance of 100)
prior_mean = c(0, 0)
prior_sd = c(10, 10)

# Run robbins munro algorithm
n = 100000  # can be lower depends how close to MAP is required
set.seed(1203) # can be changed
theta = matrix(0, nrow=(n+1), ncol=2) # stores the iterations
p = obs_stats[1]/choose(7,2) # edge density (7 nodes -> choose(7,2) = 21 dyads)
theta[1,] = c(-log((1-p)/p), 0.0) # start at random graph with this edge density
h = 5e-4 # stepsize adjustment
g = 0.50001 # adaptation speed
for(i in 1:n)
{
  alpha = h*(c(1/i, 1/i)^g)
  grad = get_grad(obs_stats, g7_net, theta[i,], prior_mean, prior_sd)
  theta[i+1,] = theta[i,] + alpha*grad

  if((i %% 10000) == 0)
  {
    cat("Iteration", i, "- current MAP estimate: ", format(theta[i+1,1],nsmall=10), format(theta[i+1,2],nsmall=10), "\n")
  }
}

mapcoef = theta[n+1,]
cat("\nFinal MAP estimate:", sprintf("%.16f", mapcoef), "\n")
# converges to approximately: c(-2.4783349784476716, 0.4694443750091412)

# this mapcoef goes to file 02_find_mass_matrix.R
