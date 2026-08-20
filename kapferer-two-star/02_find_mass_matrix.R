# 02_find_mass_matrix.R
# Compute the HMC mass matrix at the MAP for the Kapferer/esg model.
#
# Mass is the observed Fisher information of the log posterior at the MAP:
#   mass = cov_theta[s(Y)] - grad^2 log p(theta)
# where the ERGM covariance is estimated by Monte Carlo simulation at the MAP
# and the diagonal prior Hessian comes from ergm_helpers::get_priorgrad2. With
# prior_sd = 10 the prior term is O(1e-2) against ERGM stat variances of
# O(1e3), so it barely moves the result.
#
# The mass matrix is printed as a row-major 9-value block so it can be pasted
# into hmc.c as the `M[MAXSTATS*MAXSTATS]` literal.

suppressPackageStartupMessages({
  require(ergm)
  require(network)
})

if (!file.exists("ergm_helpers.R")) {
  stop("Run from this study folder (ergm_helpers.R not found).")
}
source("ergm_helpers.R")

# ----- data -----
adj          = load_graphformat1("data/kapferer.txt")
kapferer_net = network(adj, directed = FALSE)
model_rhs    = "edges + kstar(2) + gwesp(0.25, fixed = TRUE)"

# ----- MAP from 01_find_map.R -----
if (!file.exists("tuning/map_kapferer.csv")) {
  stop("tuning/map_kapferer.csv not found — run 01_find_map.R first.")
}
mapcoef = read.csv("tuning/map_kapferer.csv")$value
stopifnot(length(mapcoef) == 3)
cat(sprintf("Using MAP: (%.16f, %.16f, %.16f)\n",
            mapcoef[1], mapcoef[2], mapcoef[3]))

prior_mean = c(0, 0, 0)
prior_sd   = c(10, 10, 10)

# ----- simulate at the MAP -----
nsim = 500
seed = 20398
set.seed(seed)
sim_form = stats::as.formula(paste("kapferer_net ~", model_rhs))
networks = ergm::simulate_formula(sim_form, coef = mapcoef,
                                  nsim = nsim, output = "stats")
cat(sprintf("Simulated %d networks at MAP (seed=%d)\n", nsim, seed))

# ----- mass matrix -----
mass = cov(networks) - diag(get_priorgrad2(mapcoef, prior_mean, prior_sd))

cat("\nMass matrix (3x3):\n")
print(mass)

cat("\nMass matrix for C (row-major, paste into hmc.c M[MAXSTATS*MAXSTATS]):\n")
cat(sprintf("  %.15f, %.15f, %.15f,\n", mass[1, 1], mass[1, 2], mass[1, 3]))
cat(sprintf("  %.15f, %.15f, %.15f,\n", mass[2, 1], mass[2, 2], mass[2, 3]))
cat(sprintf("  %.15f, %.15f, %.15f\n",  mass[3, 1], mass[3, 2], mass[3, 3]))

invmass = solve(mass)
cat("\nInverse mass matrix (for diagnostics):\n")
print(invmass)

# ----- persist for downstream consumption -----
saveRDS(mass,    file = "tuning/mass_kapferer.rds")
write.csv(mass,    file = "tuning/mass_kapferer.csv",    row.names = FALSE)
cat("\nWrote tuning/mass_kapferer.{rds,csv}\n")
