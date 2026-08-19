# 04_run_hmc_single.R
# Run the noisy HMC binary on Kapferer (edges + gwesp(0.25) + gwdsp(0.25)),
# produce a 3-parameter posterior summary, posterior predictive GOF checks
# (modeled + non-modeled statistics), and prior sensitivity comparisons.

library(ergm)
library(ggplot2)

dir.create("results", showWarnings = FALSE, recursive = TRUE)


###############################################################################
#                                                                             #
#   Section 1 — Run HMC                                                      #
#                                                                             #
###############################################################################

if (!file.exists("src/hmc")) {
  cat("\n\nAttention: Run \"make all\" in src directory first!\n\n\n")
  # cd <this folder>/src && make all
} else {

  setwd("./src")

  data         = "../data/kapferer.txt"
  results_file = "../chains/hmc_results.txt"
  log_file     = "../chains/hmc_log.txt"
  time_file    = "../chains/hmc_time.txt"
  seed         = 3000

  main_iters    = 5000          # total HMC iterations
  target_accept = 0.651          # dual-averaging target acceptance rate

  # Initial theta: MPLE (fast, deterministic, prior-free dyad-conditional logit
  # fit). MPLE ignores between-dyad dependence so it's wrong in general for
  # ERGMs with triangle-like structure, but is a sensible cheap warm-start.
  cat("Fitting MPLE to obtain initial theta for HMC...\n")
  set.seed(seed)
  kap_mat = as.matrix(read.table("../data/kapferer.txt", skip = 2))
  kap_net = network::as.network(kap_mat, directed = FALSE)
  mple_form = stats::as.formula(
    "kap_net ~ edges + gwesp(0.25, fixed = TRUE) + gwdsp(0.25, fixed = TRUE)")
  mple_fit  = ergm(mple_form, estimate = "MPLE",
                   control = control.ergm(MPLE.type = "logitreg"))
  mplecoef  = unname(stats::coef(mple_fit))
  stopifnot(length(mplecoef) == 3)
  cat(sprintf("MPLE init: edges=%+.4f  gwesp=%+.4f  gwdsp=%+.4f\n",
              mplecoef[1], mplecoef[2], mplecoef[3]))

  # Prior: "normal", "uniform", or "ndg"
  prior_type = "normal"
  ndg_file   = "../data/ndg_prior.txt"   # 3-column warm-start file (edges, gwesp, gwdsp);
                                  # only used when prior_type == "ndg"
  ndg_p      = 2                  # NDG strength exponent
  grad_gap   = 100                # MC gradient thinning gap

  # hmc CLI has 19 args after the program name (20 total).
  args = c(data, "egd",
           mplecoef[1], mplecoef[2], mplecoef[3],
           "tnt", 0.5,
           grad_gap, "bri",
           prior_type, ndg_file, ndg_p,
           seed,
           results_file, log_file, time_file,
           main_iters, target_accept, "yes")
  system2("./hmc", args = args)

  hmc = read.table(results_file, header = TRUE)

  setwd("../")

  # Save results with a prior-specific name so the sensitivity section can
  # load multiple runs side-by-side.
  filename = if (prior_type == "ndg") {
    paste0("posterior-1chain-ndg-p", sprintf("%g", ndg_p))
  } else {
    paste0("posterior-1chain-", prior_type)
  }
  saveRDS(hmc, file = paste0("results/", filename, ".RDS"))
}


###############################################################################
#                                                                             #
#   Section 2 — Posterior summary                                             #
#                                                                             #
###############################################################################

cat("\n--- HMC posterior summary (theta = (edges, gwesp, gwdsp)) ---\n")
cat(sprintf("  prior: %s\n", prior_type))
cat(sprintf("  n_draws: %d\n", nrow(hmc)))

summarize_column = function(name, x) {
  qs = quantile(x, probs = c(0.025, 0.5, 0.975), names = FALSE)
  cat(sprintf("  %-12s mean=%+.4f  sd=%.4f  2.5%%=%+.4f  median=%+.4f  97.5%%=%+.4f\n",
              name, mean(x), sd(x), qs[1], qs[2], qs[3]))
}
summarize_column("theta_edges", hmc[[1]])
summarize_column("theta_gwesp", hmc[[2]])
summarize_column("theta_gwdsp", hmc[[3]])

# ---------- trace plot ----------
trace_path = paste0("results/trace-1chain-", prior_type, ".png")
png(trace_path, width = 9, height = 6, units = "in", res = 200)
op = par(mfrow = c(3, 1), mar = c(3, 4, 2, 1))
plot(hmc[[1]], type = "l", ylab = "theta_edges", xlab = "", main = "Kapferer HMC trace")
plot(hmc[[2]], type = "l", ylab = "theta_gwesp", xlab = "")
plot(hmc[[3]], type = "l", ylab = "theta_gwdsp", xlab = "iteration")
par(op)
dev.off()
cat(sprintf("Wrote trace plot: %s\n", trace_path))


###############################################################################
#                                                                             #
#   Section 3 — Posterior predictive GOF                                      #
#                                                                             #
#   1. Sample theta from the posterior (thin the HMC trace)                   #
#   2. Simulate one network per theta from the ERGM                          #
#   3. Compare modeled statistics (edges, gwesp, gwdsp) to observed          #
#   4. Compare non-modeled statistics:                                       #
#        - degree distribution                                               #
#        - edgewise shared partner (ESP) distribution                        #
#        - geodesic distance distribution                                    #
#                                                                             #
###############################################################################

# Load the observed Kapferer network
data("kapferer", package = "ergm")
ergm_form = kapferer ~ edges + gwesp(0.25, fixed = TRUE) + gwdsp(0.25, fixed = TRUE)

# Observed modeled statistics
obs_modeled = summary(ergm_form)
names(obs_modeled) = c("edges", "gwesp", "gwdsp")

# Observed degree distribution (max possible degree = 38 for 39-node network)
max_deg = network.size(kapferer) - 1
obs_degree = summary(kapferer ~ degree(0:max_deg))

# Observed edgewise shared partner distribution
max_esp = max_deg - 1
obs_esp = summary(kapferer ~ esp(0:max_esp))

# Observed geodesic distance distribution (BFS on adjacency matrix)
compute_geodist = function(net) {
  g = as.matrix(net)
  n = nrow(g)
  geodist_mat = matrix(Inf, n, n)
  diag(geodist_mat) = 0
  for (i in 1:n) {
    visited = rep(FALSE, n)
    visited[i] = TRUE
    queue = i
    while (length(queue) > 0) {
      v = queue[1]; queue = queue[-1]
      nbrs = which(g[v, ] == 1 & !visited)
      for (nb in nbrs) {
        visited[nb] = TRUE
        geodist_mat[i, nb] = geodist_mat[i, v] + 1
        queue = c(queue, nb)
      }
    }
  }
  geo = geodist_mat[upper.tri(geodist_mat)]
  geo[is.infinite(geo)] = NA
  return(geo)
}

obs_geo = compute_geodist(kapferer)
# Tabulate: distance 1, 2, ..., max_obs, Inf(=disconnected)
obs_geo_max = max(obs_geo, na.rm = TRUE)

# Thin posterior: take every thin_by-th draw, giving ~n_gof samples
n_gof     = 200
thin_by   = max(1, floor(nrow(hmc) / n_gof))
gof_idx   = seq(1, nrow(hmc), by = thin_by)
n_gof     = length(gof_idx)
cat(sprintf("\n--- Posterior predictive GOF: %d networks to simulate ---\n", n_gof))

# Storage
sim_modeled = matrix(NA, n_gof, 3, dimnames = list(NULL, c("edges", "gwesp", "gwdsp")))
sim_degree  = matrix(0, n_gof, max_deg + 1)
sim_esp     = matrix(0, n_gof, max_esp + 1)
# geodesic: store counts at each distance 1..max_possible
max_geo_store = 10  # distances beyond this are rare; lump into ">max"
sim_geo     = matrix(0, n_gof, max_geo_store + 1)  # cols 1..max_geo_store + 1 col for Inf/NA

set.seed(seed + 1)

for (ii in 1:n_gof) {
  theta_i = as.numeric(hmc[gof_idx[ii], ])
  sim_net = simulate(ergm_form, coef = theta_i, nsim = 1,
                     control = control.simulate.formula(
                       MCMC.burnin = 20000, MCMC.interval = 10000))

  # Modeled stats
  sim_modeled[ii, ] = summary(sim_net ~ edges +
                                gwesp(0.25, fixed = TRUE) +
                                gwdsp(0.25, fixed = TRUE))

  # Degree distribution
  sim_degree[ii, ] = summary(sim_net ~ degree(0:max_deg))

  # ESP distribution
  sim_esp[ii, ] = summary(sim_net ~ esp(0:max_esp))

  # Geodesic distance distribution
  geo_i = compute_geodist(sim_net)
  for (d in 1:max_geo_store) {
    sim_geo[ii, d] = sum(geo_i == d, na.rm = TRUE)
  }
  sim_geo[ii, max_geo_store + 1] = sum(geo_i > max_geo_store, na.rm = TRUE) +
                                    sum(is.na(geo_i))

  if (ii %% 50 == 0) cat(sprintf("  simulated %d / %d\n", ii, n_gof))
}
cat("  done.\n")

# --- 3a. Modeled statistics comparison ---
# For each stat: boxplot of simulated values with observed as horizontal line.

gof_modeled_path = paste0("results/gof-1chain-", prior_type, "-modeled.png")
png(gof_modeled_path, width = 9, height = 4, units = "in", res = 200)
op = par(mfrow = c(1, 3), mar = c(4, 4, 2, 1))
stat_names = c("edges", "gwesp(0.25)", "gwdsp(0.25)")
for (s in 1:3) {
  boxplot(sim_modeled[, s], main = stat_names[s], ylab = "value",
          col = "grey85", border = "grey40", outline = FALSE)
  abline(h = obs_modeled[s], col = "red", lwd = 2, lty = 2)
  legend("topright", legend = "observed", col = "red", lty = 2, lwd = 2,
         bty = "n", cex = 0.8)
}
par(op)
dev.off()
cat(sprintf("Wrote modeled-stats GOF: %s\n", gof_modeled_path))

# --- 3b. Degree distribution GOF ---
# For each degree k: boxplot of simulated counts, observed value overlaid.
# Only show degrees 0 through max observed + a few.

obs_deg_vec = as.numeric(obs_degree)
deg_show = 0:min(max_deg, max(which(obs_deg_vec > 0)) + 2)

gof_degree_path = paste0("results/gof-1chain-", prior_type, "-degree.png")
png(gof_degree_path, width = 10, height = 5, units = "in", res = 200)
boxplot(sim_degree[, deg_show + 1], names = deg_show,
        xlab = "degree", ylab = "number of nodes",
        main = "Posterior predictive: degree distribution",
        col = "grey85", border = "grey40", outline = FALSE)
points(seq_along(deg_show), obs_deg_vec[deg_show + 1],
       col = "red", pch = 16, cex = 1.2)
legend("topright", legend = "observed", col = "red", pch = 16, bty = "n")
dev.off()
cat(sprintf("Wrote degree GOF: %s\n", gof_degree_path))

# --- 3c. ESP distribution GOF ---
# Only show ESP values 0 through max observed + a few.

obs_esp_vec = as.numeric(obs_esp)
esp_show = 0:min(max_esp, max(which(obs_esp_vec > 0)) + 2)

gof_esp_path = paste0("results/gof-1chain-", prior_type, "-esp.png")
png(gof_esp_path, width = 10, height = 5, units = "in", res = 200)
boxplot(sim_esp[, esp_show + 1], names = esp_show,
        xlab = "edgewise shared partners", ylab = "number of edges",
        main = "Posterior predictive: ESP distribution",
        col = "grey85", border = "grey40", outline = FALSE)
points(seq_along(esp_show), obs_esp_vec[esp_show + 1],
       col = "red", pch = 16, cex = 1.2)
legend("topright", legend = "observed", col = "red", pch = 16, bty = "n")
dev.off()
cat(sprintf("Wrote ESP GOF: %s\n", gof_esp_path))

# --- 3d. Geodesic distance distribution GOF ---

obs_geo_tab = rep(0, max_geo_store + 1)
for (d in 1:max_geo_store) obs_geo_tab[d] = sum(obs_geo == d, na.rm = TRUE)
obs_geo_tab[max_geo_store + 1] = sum(obs_geo > max_geo_store, na.rm = TRUE) +
                                  sum(is.na(obs_geo))

geo_labels = c(as.character(1:max_geo_store), paste0(">", max_geo_store, "/Inf"))

gof_geo_path = paste0("results/gof-1chain-", prior_type, "-geodesic.png")
png(gof_geo_path, width = 9, height = 5, units = "in", res = 200)
boxplot(sim_geo, names = geo_labels,
        xlab = "geodesic distance", ylab = "number of dyads",
        main = "Posterior predictive: geodesic distance distribution",
        col = "grey85", border = "grey40", outline = FALSE)
points(1:(max_geo_store + 1), obs_geo_tab,
       col = "red", pch = 16, cex = 1.2)
legend("topright", legend = "observed", col = "red", pch = 16, bty = "n")
dev.off()
cat(sprintf("Wrote geodesic GOF: %s\n", gof_geo_path))


###############################################################################
#                                                                             #
#   Section 4 — Prior sensitivity: marginal posterior comparisons             #
#                                                                             #
#   Load posterior traces from all available prior runs and overlay           #
#   kernel density estimates for each parameter.                             #
#                                                                             #
###############################################################################

cat("\n--- Prior sensitivity ---\n")

# Discover and load all available prior runs.
# NDG naming: posterior-1chain-ndg-p<p>.RDS, e.g. p = 2 -> posterior-1chain-ndg-p2.RDS
prior_colors = c(normal = "steelblue", uniform = "darkorange")
prior_data   = list()

for (pl in c("normal", "uniform")) {
  fn = paste0("results/posterior-1chain-", pl, ".RDS")
  if (file.exists(fn)) {
    prior_data[[pl]] = readRDS(fn)
    cat(sprintf("  loaded %s (%d draws)\n", fn, nrow(prior_data[[pl]])))
  }
}
# Discover NDG files (posterior-1chain-ndg-p2.RDS, etc.)
ndg_files = sort(list.files("results", pattern = "^posterior-1chain-ndg-p.*\\.RDS$", full.names = TRUE))
ndg_greens = colorRampPalette(c("#2d8c2d", "#0a3d0a"))(max(length(ndg_files), 1))
for (k in seq_along(ndg_files)) {
  fn = ndg_files[k]
  label = sub("^posterior-1chain-", "", sub("\\.RDS$", "", basename(fn)))
  prior_data[[label]] = readRDS(fn)
  prior_colors[label] = ndg_greens[k]
  cat(sprintf("  loaded %s (%d draws)\n", fn, nrow(prior_data[[label]])))
}

if (length(prior_data) >= 2) {
  param_names = c("theta_edges", "theta_gwesp", "theta_gwdsp")

  sens_path = "results/prior-sensitivity-1chain.png"
  png(sens_path, width = 10, height = 8, units = "in", res = 200)
  op = par(mfrow = c(3, 1), mar = c(4, 4, 2, 1))

  for (p in 1:3) {
    dens_list = list()
    xlims = c(Inf, -Inf)
    ylims = c(0, 0)
    for (pl in names(prior_data)) {
      d = density(prior_data[[pl]][[p]])
      dens_list[[pl]] = d
      xlims = c(min(xlims[1], min(d$x)), max(xlims[2], max(d$x)))
      ylims[2] = max(ylims[2], max(d$y))
    }

    first = TRUE
    for (pl in names(prior_data)) {
      if (first) {
        plot(dens_list[[pl]], xlim = xlims, ylim = ylims,
             main = param_names[p], xlab = param_names[p], ylab = "density",
             col = prior_colors[pl], lwd = 2)
        first = FALSE
      } else {
        lines(dens_list[[pl]], col = prior_colors[pl], lwd = 2)
      }
    }
    legend("topright", legend = names(prior_data),
           col = prior_colors[names(prior_data)],
           lwd = 2, bty = "n", cex = 0.9)
  }

  par(op)
  dev.off()
  cat(sprintf("Wrote prior sensitivity plot: %s\n", sens_path))
} else {
  cat("  fewer than 2 prior runs found; skipping sensitivity plot.\n")
  cat("  re-run this script with different prior_type values to enable comparison.\n")
}
