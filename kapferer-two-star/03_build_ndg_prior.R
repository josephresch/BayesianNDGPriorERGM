# 03_build_ndg_prior.R
#
# Generate the NDG (Non-Degeneracy) prior warm-start file for the C code on
# the Kapferer tailor-shop network under the esg model
#   y ~ edges + kstar(2) + gwesp(0.25, fixed=TRUE).
#
# The adaptive NDG prior (see src/prior.c) maintains its own convex hull and
# its own IS pool internally; this script only needs to hand it a batch of
# plausible sufficient statistics to seed that 3D hull. We simulate networks
# from the ERGM at the MPLE (Maximum Pseudo-Likelihood Estimate) and write
# their (edges, kstar2, gwesp) triples to disk. No convex hull is computed in
# R and no reference theta is written — the file is deliberately free of any
# ground-truth information.
#
# Why MPLE rather than MAP: the MPLE is a quick, closed-form-ish fit that
# does not depend on any prior choice (or on the HMC pipeline we are trying
# to warm-start). Using it keeps the NDG warm-start independent of the
# posterior workflow whose prior it feeds.
#
# Output format (read by prior_ndg_warm_start() in src/prior.c):
#   Line 1:       number of rows N
#   Lines 2..N+1: "<s1> <s2> <s3>"  (three whitespace-separated real numbers)
#
# The NDG prior hull/pool is 3D (edges, kstar2, gwesp(0.25, fixed)).
# hull3d.c reads these three columns directly as doubles.

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
model_rhs    = "edges + kstar(2) + gwesp(0.25, fixed = TRUE)"

cat("Observed statistics:\n")
form = stats::as.formula(paste("kapferer_net ~", model_rhs))
print(summary(form))

# ----- reference theta for simulation: MPLE -----
# MPLE fits each dyad's conditional logit given the rest of the network; it
# ignores between-dyad dependence (so it's wrong in general for ERGMs with
# triangle-like structure) but is fast, deterministic, and prior-free. That
# makes it a good "cheap but sensible" centre for simulating the NDG pool.
cat("Fitting MPLE to obtain simulation reference theta...\n")
mple_fit = ergm::ergm(form, estimate = "MPLE",
                      control = ergm::control.ergm(MPLE.type = "logitreg"))
init = stats::coef(mple_fit)
stopifnot(length(init) == 3)
cat("MPLE coefficients:\n")
print(init)

# ----- simulate auxiliary networks -----
set.seed(10)
N_init = 100000
cat(sprintf("Simulating %d networks at reference theta...\n", N_init))
init_aux_stats = ergm::simulate_formula(form, coef = init,
                                        nsim = N_init, output = "stats")

# ----- write warm-start file -----
outfile = "data/ndg_prior.txt"
cat(sprintf("Writing NDG prior warm-start file: %s (%d rows, 3 columns)\n",
            outfile, N_init))
con = file(outfile, "w")
writeLines(formatC(N_init, format = "d"), con)
write.table(init_aux_stats, con,
            row.names = FALSE, col.names = FALSE, sep = " ")
close(con)

cat("Done.\n")
