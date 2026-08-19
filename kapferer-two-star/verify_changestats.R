# verify_changestats.R
#
# Cross-check: read the Kapferer adjacency matrix used by the C codebase
# (kapferer.txt, our "graphformat 1" file), build a `network` object, and
# compute ergm::summary(net ~ edges + kstar(2) + gwesp(0.25, fixed=TRUE)).
#
# The values printed here should exactly match (to numerical precision) the
# full-graph stats reported by src/test_changestats on the same file:
#   edges = 158
#   kstar2 = 1566
#   gwesp(0.25) = 185.7891865804

suppressPackageStartupMessages({
  require(network)
  require(ergm)
})

graph_path <- "data/kapferer.txt"
if (!file.exists(graph_path)) {
  stop("Run this from the study folder (data/kapferer.txt not found).")
}

lines <- readLines(graph_path)
stopifnot(grepl("^graphformat 1$", lines[1]))
stopifnot(grepl("^nnodes ", lines[2]))
nnodes <- as.integer(sub("^nnodes ", "", lines[2]))
adj_lines <- lines[3:(2 + nnodes)]
adj <- do.call(rbind, lapply(adj_lines, function(s) as.integer(strsplit(trimws(s), "\\s+")[[1]])))
stopifnot(nrow(adj) == nnodes, ncol(adj) == nnodes)
stopifnot(isSymmetric(adj), all(diag(adj) == 0))

net <- network(adj, directed = FALSE)
alpha <- 0.25

stats <- summary(net ~ edges + kstar(2) + gwesp(alpha, fixed = TRUE))

cat(sprintf("Kapferer: nnodes=%d nedges=%d\n", nnodes, sum(adj) / 2))
cat(sprintf("  edges        = %d\n",               as.integer(stats["edges"])))
cat(sprintf("  kstar2       = %.10f\n",             unname(stats[2])))
cat(sprintf("  gwesp(%.3f) = %.10f\n", alpha, unname(stats[3])))

c_edges  <- 158
c_kstar2 <- 1566
c_gwesp  <- 185.7891865804

tol <- 1e-6
ok_edges  <- stats["edges"] == c_edges
ok_kstar2 <- abs(unname(stats[2]) - c_kstar2) < tol
ok_gwesp  <- abs(unname(stats[3]) - c_gwesp)  < tol

cat("\nC-side values:\n")
cat(sprintf("  edges        = %d\n", c_edges))
cat(sprintf("  kstar2       = %.10f\n", as.double(c_kstar2)))
cat(sprintf("  gwesp(%.3f) = %.10f\n", alpha, c_gwesp))

cat("\nDifferences (R - C):\n")
cat(sprintf("  edges       = %+d\n",   as.integer(stats["edges"]) - c_edges))
cat(sprintf("  kstar2      = %+.3e\n", unname(stats[2]) - c_kstar2))
cat(sprintf("  gwesp       = %+.3e\n", unname(stats[3]) - c_gwesp))

if (ok_edges && ok_kstar2 && ok_gwesp) {
  cat("\nPASS: C totals match ergm::summary within", tol, "\n")
  quit(status = 0)
} else {
  cat("\nFAIL: mismatch between C totals and ergm::summary\n")
  quit(status = 1)
}
