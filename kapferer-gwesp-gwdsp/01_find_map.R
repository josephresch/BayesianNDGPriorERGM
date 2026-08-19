# 01_find_map.R
# Locate the posterior mode of the ERGM
#    Kapferer tailor-shop ~ edges + gwesp(0.25, fixed=TRUE) + gwdsp(0.25, fixed=TRUE)
# under the Caimo-Friel prior N(0, diag(100)) on theta = (edges, gwesp, gwdsp).
#
# The MAP is written to tuning/map_kapferer.csv so that 02_find_mass_matrix.R (and any
# downstream runner) can pick it up without manual copy-paste.

suppressPackageStartupMessages({
  require(ergm)
  require(network)
})

if (!file.exists("ergm_helpers.R")) {
  stop("Run from the study folder (ergm_helpers.R not found).")
}
source("ergm_helpers.R")

# ----- data -----
adj          = load_graphformat1("data/kapferer.txt")
kapferer_net = network(adj, directed = FALSE)
nnodes       = nrow(adj)
ndyads       = nnodes * (nnodes - 1) / 2

model_rhs = "edges + gwesp(0.25, fixed = TRUE) + gwdsp(0.25, fixed = TRUE)"
obs_form  = stats::as.formula(paste("kapferer_net ~", model_rhs))
obs_stats = as.numeric(summary(obs_form))
cat(sprintf("Observed stats: edges=%.0f gwesp=%.10f gwdsp=%.10f\n",
            obs_stats[1], obs_stats[2], obs_stats[3]))

# ----- prior -----
prior_mean = c(0, 0, 0)
prior_sd   = c(10, 10, 10)   # N(0, diag(100))

# ----- Robbins-Monro settings -----
# Increase n / nsim for a tighter MAP estimate; these defaults finish in a
# few minutes and are accurate enough for seeding the HMC mass matrix.
n     = 5000    # iterations
nsim  = 20      # graphs per gradient estimate
h     = 5e-4    # step-size scale
g     = 0.50001 # decay exponent
seed  = 1203

set.seed(seed)
theta = matrix(0, nrow = n + 1, ncol = 3)
# start: edge density → logit(p) for edges, 0 for gwesp/gwdsp
p = obs_stats[1] / ndyads
theta[1, ] = c(-log((1 - p) / p), 0.0, 0.0)

cat(sprintf("Start theta: (%.6f, %.6f, %.6f)   seed=%d  n=%d  nsim=%d\n",
            theta[1, 1], theta[1, 2], theta[1, 3], seed, n, nsim))

report_every = max(1, floor(n / 20))
for (i in seq_len(n)) {
  alpha_vec = h * (c(1/i, 1/i, 1/i) ^ g)
  grad      = get_grad(obs_stats, kapferer_net, theta[i, ],
                       prior_mean, prior_sd, model_rhs, nsim = nsim)
  theta[i + 1, ] = theta[i, ] + alpha_vec * grad

  if ((i %% report_every) == 0) {
    cat(sprintf("  iter %6d   theta = (%+.10f, %+.10f, %+.10f)\n",
                i, theta[i + 1, 1], theta[i + 1, 2], theta[i + 1, 3]))
  }
}

mapcoef = theta[n + 1, ]
cat(sprintf("\nFinal MAP estimate: (%.16f, %.16f, %.16f)\n",
            mapcoef[1], mapcoef[2], mapcoef[3]))

# Save both the final MAP and the full trajectory (the latter is handy for
# diagnostics; 02_find_mass_matrix.R only needs mapcoef).
write.csv(data.frame(param = c("edges", "gwesp", "gwdsp"), value = mapcoef),
          file = "tuning/map_kapferer.csv", row.names = FALSE)
saveRDS(theta, file = "tuning/map_kapferer_trajectory.rds")
cat("Wrote tuning/map_kapferer.csv and tuning/map_kapferer_trajectory.rds\n")
