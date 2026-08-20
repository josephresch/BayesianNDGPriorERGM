# 02_find_mass_matrix.R
# This code calculates the mass matrix at the MAP (Maximum a Posteriori estimate)
# Note: Some random seeds may give empty/full (i.e. degenerate) graphs as an artifact
# of the ERGM TieNoTie sampler. This code does not employ degeneracy checks.

# Load libraries
  require(ergm)
  # Run from this folder (the paths below are relative to it).
  source("ergm_helpers.R")

# Load g7 network
  load("data/g7.RData")
  load("data/g7kstars.RData")
  obs = c(7, 14)

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

# Store MAP estimate from 01_find_map.R
  mapcoef = c(-2.4783349784476716, 0.4694443750091412)
  priormean = c(0, 0)
  priorsd = c(10, 10)

# Simulate 500 networks at this MAP
  set.seed(20398)
  networks = simulate(g7_net ~ edges + kstar(2), coef=mapcoef, nsim=500, statsonly=TRUE)

# Calculate mass matrix from these networks and prior (gaussian)
  mass = cov(networks) - diag(get_priorgrad2(mapcoef, priormean, priorsd))
  cat("Mass matrix:\n")
  print(mass)
  cat("\nMass matrix for C (flat array, row-major):\n")
  cat(sprintf("%.15f, %.15f, %.15f, %.15f\n", mass[1,1], mass[1,2], mass[2,1], mass[2,2]))
  cat("\nInverse mass matrix:\n")
  print(solve(mass))

# Remove items
  rm(g7, g7kstars, obs, select_g7, select_id1, select_id2, adj, g7_net, obs_stats)
