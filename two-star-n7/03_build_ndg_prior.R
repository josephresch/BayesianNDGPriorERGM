# 03_build_ndg_prior.R
#
# Generate the NDG (Non-Degeneracy) prior warm-start file for the C code.
#
# The adaptive NDG prior (see src/prior.c) maintains its own convex hull
# and its own IS pool internally; this script only needs to hand it a
# batch of plausible sufficient statistics to seed that hull. We simulate
# networks from the ERGM at the MPLE and write their (edges, kstar2)
# pairs to disk. No convex hull is computed in R, and no reference theta
# is written -- the file is deliberately free of any ground-truth
# information.
#
# Output format (read by prior_ndg_warm_start() in src/prior.c):
#   Line 1:     number of rows N
#   Lines 2..N+1: "<s1> <s2>"   (two whitespace-separated integers)

require(ergm)
require(network)

# Construct g7 network directly from adjacency matrix
adj = matrix(c(
  0, 1, 1, 0, 0, 1, 0,
  1, 0, 0, 0, 0, 0, 0,
  1, 0, 0, 1, 1, 1, 1,
  0, 0, 1, 0, 0, 0, 0,
  0, 0, 1, 0, 0, 0, 0,
  1, 0, 1, 0, 0, 0, 0,
  0, 0, 1, 0, 0, 0, 0
), nrow = 7, byrow = TRUE)
g = network(adj, directed = FALSE)

cat("Observed statistics:\n")
print(summary(g ~ edges + kstar(2)))

# MPLE estimate: reference parameter used only for the simulation step.
# It is NOT written to disk -- the C code tracks its own reference theta
# dynamically at each leapfrog step.
init = ergm(g ~ edges + kstar(2), estimate = "MPLE")$coefficients
cat("MPLE coefficients (simulation reference):\n")
print(init)

# Simulate auxiliary networks at MPLE
set.seed(10)
N_init = 100000
cat(sprintf("Simulating %d networks at MPLE...\n", N_init))
init_aux_stats = simulate(g ~ edges + kstar(2),
                          coef = init,
                          nsim = N_init,
                          output = "stats")

# Write the warm-start file: count on line 1, then "s1 s2" per row.
outfile = "data/ndg_prior.txt"
cat(sprintf("Writing NDG prior warm-start file: %s (%d rows)\n",
            outfile, N_init))
con = file(outfile, "w")
writeLines(formatC(N_init, format = "d"), con)
write.table(init_aux_stats, con,
            row.names = FALSE, col.names = FALSE, sep = " ")
close(con)

cat("Done.\n")
