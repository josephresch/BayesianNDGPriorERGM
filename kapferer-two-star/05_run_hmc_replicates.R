# 05_run_hmc_replicates.R
# -----------------------------------------------------------------------
# Replicate the noisy-HMC algorithm 20 times following the protocol in
# Stoehr, Benson & Friel (2019):
#
#   "The different algorithms were replicated 20 times using a different
#    starting point and random seed for each experiment."
#
# Per replicate:
#   - starting point drawn from N(MPLE, M^{-1})
#   - distinct C-side RNG seed
#   - same prior / main_iters / tuning as 04_run_hmc_single.R
#
# Post-run: per-replicate metrics (accept rate, eps, L, runtime,
# posterior mean/median in theta-space), Table-3-style summary,
# pooled posterior density plot, and prior sensitivity comparison.
# -----------------------------------------------------------------------

library(ergm)
library(ggplot2)

if (!file.exists("src/hmc")) {
  stop("Run \"make all\" in src/ first.")
}

options(scipen = 999)

###############################################################################
#   CONFIG
###############################################################################

n_replicates  = 20
chain_length  = 5000           # main_iters per replicate
warmup        = 1000           # drop before computing metrics
master_seed   = 3000

prior_type    = "ndg"      # "normal", "uniform", or "ndg"
ndg_file      = "../data/ndg_prior.txt"
ndg_p         = 10

target_accept = 0.651
tnt_p         = 0.5
grad_gap      = 100
corr_type     = "bri"

# MPLE coefficients for Kapferer ~ edges + kstar(2) + gwesp(0.25)
cat("Fitting MPLE...\n")
set.seed(master_seed)
kap_mat = as.matrix(read.table("data/kapferer.txt", skip = 2))
kap_net = network::as.network(kap_mat, directed = FALSE)
mple_form = stats::as.formula(
  "kap_net ~ edges + kstar(2) + gwesp(0.25, fixed = TRUE)")
mple_fit  = ergm(mple_form, estimate = "MPLE",
                 control = control.ergm(MPLE.type = "logitreg"))
mplecoef  = unname(stats::coef(mple_fit))
stopifnot(length(mplecoef) == 3)
cat(sprintf("MPLE: edges=%+.4f  kstar2=%+.4f  gwesp=%+.4f\n",
            mplecoef[1], mplecoef[2], mplecoef[3]))

# Mass matrix from 02_find_mass_matrix.R
M    = as.matrix(read.csv("tuning/mass_kapferer.csv"))
Minv = solve(M)

###############################################################################
#   PATHS
###############################################################################

rep_subdir   = "../chains"      # where the C binary writes, relative to src/
full_rep_dir = "chains"         # same directory, relative to this folder
dir.create(full_rep_dir, showWarnings = FALSE, recursive = TRUE)
dir.create("results", showWarnings = FALSE)

###############################################################################
#   DRAW starting points and C-seeds
###############################################################################

set.seed(master_seed)
starts  = MASS::mvrnorm(n_replicates, mu = mplecoef, Sigma = Minv / 4)
c_seeds = sample.int(.Machine$integer.max - 1L, n_replicates)

cat(sprintf("=== %d-replicate run | prior=%s | main_iters=%d | master_seed=%d ===\n",
            n_replicates, prior_type, chain_length, master_seed))
cat("Starting points (rows) drawn from N(MPLE, M^{-1}/4):\n")
print(round(starts, 4))

###############################################################################
#   Log / stdout parsers
###############################################################################

parse_log = function(log_path) {
  lines     = readLines(log_path)
  acc_line  = grep("^accepted:",          lines, value = TRUE)
  time_line = grep("^time_\\(seconds\\):", lines, value = TRUE)
  acc_m  = regmatches(acc_line,
                      regexec("\\(([0-9.eE+-]+) %\\)", acc_line))[[1]]
  time_m = regmatches(time_line,
                      regexec("([0-9.eE+-]+)\\s*$", time_line))[[1]]
  list(accept_rate = as.numeric(acc_m[2]) / 100,
       runtime_sec = as.numeric(time_m[2]))
}

parse_stdout = function(stdout_path) {
  lines = readLines(stdout_path)
  final_line = grep("^1000 eps:", lines, value = TRUE)
  if (length(final_line) >= 1L) {
    m = regmatches(final_line[1],
                   regexec(
                     "^1000 eps: ([0-9.eE+-]+) epsbar: [0-9.eE+-]+ nfrogs: ([0-9]+)",
                     final_line[1]))[[1]]
    return(list(eps = as.numeric(m[2]), L = as.integer(m[3])))
  }
  # Fallback: warm-up didn't complete
  eps_lines = grep("^[0-9]+ eps:", lines, value = TRUE)
  if (length(eps_lines) == 0L) {
    return(list(eps = NA_real_, L = NA_integer_))
  }
  last = tail(eps_lines, 1)
  m = regmatches(last,
                 regexec("epsbar: ([0-9.eE+-]+) nfrogs: [0-9]+ \\(([0-9]+)\\)",
                         last))[[1]]
  list(eps = as.numeric(m[2]), L = as.integer(m[3]))
}

###############################################################################
#   MAIN LOOP
###############################################################################

setwd("./src")
replicates = vector("list", n_replicates)
overall_t0 = proc.time()

for (ii in seq_len(n_replicates)) {

  tag      = sprintf("rep%02d_%s", ii, prior_type)
  res_file = file.path(rep_subdir, sprintf("%s_results.txt", tag))
  log_file = file.path(rep_subdir, sprintf("%s_log.txt",     tag))
  tim_file = file.path(rep_subdir, sprintf("%s_time.txt",    tag))
  std_file = file.path(rep_subdir, sprintf("%s_stdout.txt",  tag))

  theta0 = starts[ii, ]
  seedC  = c_seeds[ii]

  args_vec = c(
    "../data/kapferer.txt", "esg",
    theta0[1], theta0[2], theta0[3],
    "tnt", tnt_p,
    grad_gap, corr_type,
    prior_type, ndg_file, ndg_p,
    formatC(seedC,        format = "d"),
    res_file, log_file, tim_file,
    formatC(chain_length, format = "d"),
    target_accept, "yes"
  )

  cat(sprintf("\n[%s] start=(%+.4f, %+.4f, %+.4f) seed=%d\n",
              tag, theta0[1], theta0[2], theta0[3], seedC))

  t0 = proc.time()
  rc = system2("./hmc", args = args_vec,
               stdout = std_file, stderr = std_file)
  wall_elapsed = (proc.time() - t0)[["elapsed"]]

  if (rc != 0L) {
    warning(sprintf("hmc exited with code %d for %s; see %s",
                    rc, tag, std_file))
    next
  }

  log_info  = parse_log(log_file)
  tune_info = parse_stdout(std_file)

  # Load chain, drop warmup
  hmc_raw = read.table(res_file, header = TRUE)
  samples = as.matrix(hmc_raw)[(warmup + 1L):nrow(hmc_raw), , drop = FALSE]
  colnames(samples) = c("theta_edges", "theta_kstar2", "theta_gwesp")

  replicates[[ii]] = list(
    rep              = ii,
    start_theta      = theta0,
    c_seed           = seedC,
    accept_rate      = log_info$accept_rate,
    runtime_sec      = log_info$runtime_sec,
    eps              = tune_info$eps,
    L                = tune_info$L,
    wall_elapsed_sec = wall_elapsed,
    samples          = samples
  )

  cat(sprintf(
    "  accept=%.2f%%  eps=%.4f  L=%d  runtime=%.1fs\n",
    100 * log_info$accept_rate, tune_info$eps, tune_info$L,
    log_info$runtime_sec))
}

setwd("../")
overall_elapsed = (proc.time() - overall_t0)[["elapsed"]]

# Save
save_filename = if (prior_type == "ndg") {
  paste0("results/posterior-20rep-ndg-p", sprintf("%g", ndg_p), ".RDS")
} else {
  paste0("results/posterior-20rep-", prior_type, ".RDS")
}
saveRDS(replicates, file = save_filename)
cat(sprintf("Saved replicates: %s\n", save_filename))


###############################################################################
#                                                                             #
#   METRICS                                                                   #
#                                                                             #
###############################################################################

# Everything below runs off a STORED replicate set, so the metrics and figures
# can be regenerated without re-running the sampler. These two lines
# deliberately override the CONFIG block at the top of the script; set them to
# whichever run you want to report on.
prior_type = "ndg"
ndg_p = 10
replicates = readRDS(ifelse(prior_type == "ndg",
                            paste0("results/posterior-20rep-ndg-p", sprintf("%g", ndg_p), ".RDS"),
                            paste0("results/posterior-20rep-", prior_type, ".RDS")))

completed = !sapply(replicates, is.null)
cat(sprintf("\n=== Computing metrics for %d completed replicates ===\n",
            sum(completed)))

metrics_rows = lapply(replicates[completed], function(r) {
  pm  = colMeans(r$samples)
  pmed = apply(r$samples, 2, median)
  psd  = apply(r$samples, 2, sd)

  data.frame(
    rep              = r$rep,
    start_theta1     = r$start_theta[1],
    start_theta2     = r$start_theta[2],
    start_theta3     = r$start_theta[3],
    c_seed           = r$c_seed,
    accept_rate      = r$accept_rate,
    eps              = r$eps,
    L                = r$L,
    runtime_sec      = r$runtime_sec,
    wall_elapsed_sec = r$wall_elapsed_sec,
    pm_edges         = pm[1],
    pm_kstar2        = pm[2],
    pm_gwesp         = pm[3],
    pmed_edges       = pmed[1],
    pmed_kstar2      = pmed[2],
    pmed_gwesp       = pmed[3],
    psd_edges        = psd[1],
    psd_kstar2       = psd[2],
    psd_gwesp        = psd[3],
    stringsAsFactors = FALSE
  )
})
metrics = do.call(rbind, metrics_rows)


###############################################################################
#   TABLE-STYLE SUMMARY
###############################################################################

fmt = function(v) {
  v = v[is.finite(v)]
  if (length(v) == 0) return("NA")
  sprintf("%.4f (%.4f)", mean(v), sd(v))
}

cat(sprintf(
  "\n=== Replicate summary | prior=%s | %d chains of %d iters (drop first %d) ===\n",
  prior_type, nrow(metrics), chain_length, warmup))

rows = list(
  c("accept_rate",        fmt(metrics$accept_rate)),
  c("eps (final)",        fmt(metrics$eps)),
  c("L   (final)",        fmt(metrics$L)),
  c("runtime_sec",        fmt(metrics$runtime_sec)),
  c("pm_edges",           fmt(metrics$pm_edges)),
  c("pm_kstar2",          fmt(metrics$pm_kstar2)),
  c("pm_gwesp",           fmt(metrics$pm_gwesp)),
  c("pmed_edges",         fmt(metrics$pmed_edges)),
  c("pmed_kstar2",        fmt(metrics$pmed_kstar2)),
  c("pmed_gwesp",         fmt(metrics$pmed_gwesp)),
  c("psd_edges",          fmt(metrics$psd_edges)),
  c("psd_kstar2",         fmt(metrics$psd_kstar2)),
  c("psd_gwesp",          fmt(metrics$psd_gwesp))
)
for (r in rows) {
  cat(sprintf("  %-22s %s\n", r[1], r[2]))
}



###############################################################################
#                                                                             #
#   POOLED POSTERIOR PLOT (theta-space)                                       #
#                                                                             #
###############################################################################

# Pool post-warmup theta samples from all successful replicates
theta_chains = lapply(replicates,
                      function(r) if (is.null(r)) NULL else r$samples)
pooled = do.call(rbind, theta_chains)

if (is.null(pooled) || nrow(pooled) == 0L) {
  warning("No completed chains to pool; skipping plots.")
} else {
  colnames(pooled) = c("theta_edges", "theta_kstar2", "theta_gwesp")

  cat(sprintf("\nPooling %d chains -> %d total samples\n",
              sum(!sapply(theta_chains, is.null)), nrow(pooled)))

  pooled_median = apply(pooled, 2, median)
  pooled_mean   = colMeans(pooled)

  plot_filename = if (prior_type == "ndg") {
    paste0("ndg-p", sprintf("%g", ndg_p))
  } else {
    prior_type
  }

  # Marginal density plots (3 panels)
  pooled_path = paste0("results/trace-20rep-", plot_filename, "-pooled.png")
  png(pooled_path, width = 10, height = 8, units = "in", res = 200)
  op = par(mfrow = c(3, 1), mar = c(4, 4, 2, 1))
  param_names = c("theta_edges", "theta_kstar2", "theta_gwesp")
  for (p in 1:3) {
    d = density(pooled[, p])
    plot(d, main = param_names[p], xlab = param_names[p], ylab = "density",
         col = "black", lwd = 2)
    abline(v = pooled_median[p], col = "red", lty = 2, lwd = 1.5)
    abline(v = pooled_mean[p], col = "blue", lty = 3, lwd = 1.5)
    legend("topright", legend = c("median", "mean"),
           col = c("red", "blue"), lty = c(2, 3), lwd = 1.5,
           bty = "n", cex = 0.8)
  }
  par(op)
  dev.off()
  cat(sprintf("Wrote pooled posterior: %s\n", pooled_path))

  # Pairwise 2D hex plots (3 panels)
  pairs_path = paste0("results/trace-20rep-", plot_filename, "-pairs.png")
  pairs_list = list(c(1, 2), c(1, 3), c(2, 3))
  png(pairs_path, width = 12, height = 4, units = "in", res = 200)
  print(
    gridExtra::grid.arrange(
      grobs = lapply(pairs_list, function(ij) {
        i = ij[1]; j = ij[2]
        df = data.frame(x = pooled[, i], y = pooled[, j])
        ggplot(df, aes(x = x, y = y)) +
          geom_hex(bins = 80, aes(fill = after_stat(count / sum(count)))) +
          scale_fill_distiller(palette = "Blues", direction = 1, name = "density") +
          labs(x = param_names[i], y = param_names[j]) +
          theme_bw(base_size = 10) +
          theme(panel.grid.minor = element_blank(),
                legend.position = "none")
      }),
      ncol = 3
    )
  )
  dev.off()
  cat(sprintf("Wrote pairwise hex: %s\n", pairs_path))

  # --------------------------------------------------------------------- #
  #   POSTERIOR PREDICTIVE GOF (on pooled 20-chain draws)                  #
  #                                                                       #
  #   1. Thin the pooled posterior to ~n_gof draws                        #
  #   2. Simulate one network per draw from the ERGM                     #
  #   3. Compare modeled statistics (edges, kstar2, gwesp) to observed   #
  #   4. Compare non-modeled statistics:                                 #
  #        - degree distribution                                         #
  #        - edgewise shared partner (ESP) distribution                  #
  #        - geodesic distance distribution                              #
  # --------------------------------------------------------------------- #

  data("kapferer", package = "ergm")
  ergm_form = kapferer ~ edges + kstar(2) + gwesp(0.25, fixed = TRUE)

  # Observed modeled statistics
  obs_modeled = summary(ergm_form)
  names(obs_modeled) = c("edges", "kstar2", "gwesp")

  # Observed degree distribution
  max_deg = network.size(kapferer) - 1
  obs_degree = summary(kapferer ~ degree(0:max_deg))

  # Observed ESP distribution
  max_esp = max_deg - 1
  obs_esp = summary(kapferer ~ esp(0:max_esp))

  # Geodesic distance via BFS
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

  # Thin pooled draws
  n_gof   = 200
  thin_by = max(1, floor(nrow(pooled) / n_gof))
  gof_idx = seq(1, nrow(pooled), by = thin_by)
  n_gof   = length(gof_idx)
  cat(sprintf("\n--- Posterior predictive GOF: %d networks to simulate ---\n", n_gof))

  # Storage
  sim_modeled = matrix(NA, n_gof, 3,
                       dimnames = list(NULL, c("edges", "kstar2", "gwesp")))
  sim_degree  = matrix(0, n_gof, max_deg + 1)
  sim_esp     = matrix(0, n_gof, max_esp + 1)
  max_geo_store = 10
  sim_geo     = matrix(0, n_gof, max_geo_store + 1)

  set.seed(master_seed + 1)

  for (ii in seq_len(n_gof)) {
    theta_i = as.numeric(pooled[gof_idx[ii], ])
    sim_net = simulate(ergm_form, coef = theta_i, nsim = 1,
                       control = control.simulate.formula(
                         MCMC.burnin = 20000, MCMC.interval = 10000))

    sim_modeled[ii, ] = summary(sim_net ~ edges +
                                  kstar(2) +
                                  gwesp(0.25, fixed = TRUE))
    sim_degree[ii, ] = summary(sim_net ~ degree(0:max_deg))
    sim_esp[ii, ]    = summary(sim_net ~ esp(0:max_esp))

    geo_i = compute_geodist(sim_net)
    for (d in 1:max_geo_store) {
      sim_geo[ii, d] = sum(geo_i == d, na.rm = TRUE)
    }
    sim_geo[ii, max_geo_store + 1] = sum(geo_i > max_geo_store, na.rm = TRUE) +
                                      sum(is.na(geo_i))

    if (ii %% 50 == 0) cat(sprintf("  simulated %d / %d\n", ii, n_gof))
  }
  cat("  done.\n")

  # ---------------------------------------------------------------
  # Pooled posterior predictive table (for LaTeX table in paper)
  # Uses the same sim_modeled matrix as the GOF plots above.
  # ---------------------------------------------------------------
  pooled_pred_med = apply(sim_modeled, 2, median)
  pooled_pred_q25 = apply(sim_modeled, 2, quantile, 0.25)
  pooled_pred_q75 = apply(sim_modeled, 2, quantile, 0.75)

  cat("\n--- Pooled posterior predictive table ---\n")
  cat(sprintf("  Observed:  edges = %.0f,  kstar2 = %.0f,  gwesp = %.1f\n",
              obs_modeled[1], obs_modeled[2], obs_modeled[3]))
  stat_names_tbl = c("edges", "kstar2", "gwesp")
  for (s in 1:3) {
    cat(sprintf("  %-8s  median = %7.1f   [Q25 = %7.1f,  Q75 = %7.1f]\n",
                stat_names_tbl[s],
                pooled_pred_med[s], pooled_pred_q25[s], pooled_pred_q75[s]))
  }

  gof_prefix = paste0("results/gof-20rep-", plot_filename)

  # Precompute display ranges
  obs_deg_vec = as.numeric(obs_degree)
  deg_show = 0:min(max_deg, max(which(obs_deg_vec > 0)) + 2)

  obs_esp_vec = as.numeric(obs_esp)
  esp_show = 0:min(max_esp, max(which(obs_esp_vec > 0)) + 2)

  obs_geo_tab = rep(0, max_geo_store + 1)
  for (d in 1:max_geo_store) obs_geo_tab[d] = sum(obs_geo == d, na.rm = TRUE)
  obs_geo_tab[max_geo_store + 1] = sum(obs_geo > max_geo_store, na.rm = TRUE) +
                                    sum(is.na(obs_geo))
  geo_labels = c(as.character(1:max_geo_store), "NR")

  # Normalize to proportions
  n_nodes = network.size(kapferer)
  n_dyads = n_nodes * (n_nodes - 1) / 2

  obs_deg_prop = obs_deg_vec / n_nodes
  sim_deg_prop = sim_degree / n_nodes

  obs_esp_prop = obs_esp_vec / sum(obs_esp_vec)
  sim_esp_prop = sim_esp / rowSums(sim_esp)

  obs_geo_prop = obs_geo_tab / n_dyads
  sim_geo_prop = sim_geo / n_dyads

  # 2x2 GOF panel
  # Layout matrix (6 panels):
  #   top row: degree (1, spans 3 cols) | ESP (2, spans 3 cols)
  #   bot row: geodesic (3, spans 3 cols) | edges (4) | kstar2 (5) | gwesp (6)
  gof_path = paste0(gof_prefix, ".png")
  png(gof_path, width = 10, height = 9, units = "in", res = 200)
  lmat = matrix(c(1, 1, 1, 2, 2, 2,
                   3, 3, 3, 4, 5, 6), nrow = 2, byrow = TRUE)
  layout(lmat)
  par(oma = c(0, 0, 0.5, 0))

  # Panel 1 — degree
  par(mar = c(4.5, 4.5, 1.5, 1))
  boxplot(sim_deg_prop[, deg_show + 1], names = deg_show,
          xlab = "degree", ylab = "proportion of nodes",
          col = "grey85", border = "grey40", outline = FALSE,
          cex.axis = 0.8, cex.lab = 1.2, las = 1)
  lines(seq_along(deg_show), obs_deg_prop[deg_show + 1],
        col = "black", lwd = 2)
  points(seq_along(deg_show), obs_deg_prop[deg_show + 1],
         col = "red", pch = 16, cex = 1.0)

  # Panel 2 — ESP
  par(mar = c(4.5, 4.5, 1.5, 1))
  boxplot(sim_esp_prop[, esp_show + 1], names = esp_show,
          xlab = "edge-wise shared partners", ylab = "proportion of edges",
          col = "grey85", border = "grey40", outline = FALSE,
          cex.axis = 0.8, cex.lab = 1.2, las = 1)
  lines(seq_along(esp_show), obs_esp_prop[esp_show + 1],
        col = "black", lwd = 2)
  points(seq_along(esp_show), obs_esp_prop[esp_show + 1],
         col = "red", pch = 16, cex = 1.0)

  # Panel 3 — minimum geodesic distance
  par(mar = c(4.5, 4.5, 1.5, 1))
  boxplot(sim_geo_prop, names = geo_labels,
          xlab = "minimum geodesic distance", ylab = "proportion of dyads",
          col = "grey85", border = "grey40", outline = FALSE,
          cex.axis = 0.8, cex.lab = 1.2, las = 1)
  lines(1:(max_geo_store + 1), obs_geo_prop,
        col = "black", lwd = 2)
  points(1:(max_geo_store + 1), obs_geo_prop,
         col = "red", pch = 16, cex = 1.0)

  # Panels 4-6 — model statistics (mean value parameters)
  stat_labels = c("edges", "kstar2", "gwesp")
  stat_ylims  = list(c(0, 750), c(0, 28000), c(0, 1000))
  for (s in 1:3) {
    par(mar = c(4.5, ifelse(s == 1, 4.5, 3), 1.5, ifelse(s == 3, 1, 0.5)))
    boxplot(sim_modeled[, s],
            col = "grey85", border = "grey40", outline = FALSE,
            ylab = "", las = 1, cex.axis = 0.8, cex.lab = 1.2,
            ylim = stat_ylims[[s]])
    title(xlab = stat_labels[s], cex.lab = 1.2)
    abline(h = obs_modeled[s], col = "red", lwd = 2, lty = 2)
  }
  dev.off()
  cat(sprintf("Wrote GOF panel: %s\n", gof_path))
}


###############################################################################
#                                                                             #
#   PRIOR SENSITIVITY (on pooled 20-chain draws)                             #
#                                                                             #
#   Discover all available 20-replicate RDS files and overlay kernel         #
#   density estimates for each parameter.                                    #
#                                                                             #
###############################################################################

cat("\n--- Prior sensitivity (pooled replicates) ---\n")

prior_colors = c(normal = "steelblue", uniform = "darkorange")
prior_data   = list()

# Load normal / uniform replicate files
for (pl in c("normal", "uniform")) {
  fn = paste0("results/posterior-20rep-", pl, ".RDS")
  if (file.exists(fn)) {
    reps = readRDS(fn)
    ok   = !sapply(reps, is.null)
    chains = lapply(reps[ok], function(r) r$samples)
    prior_data[[pl]] = as.data.frame(do.call(rbind, chains))
    colnames(prior_data[[pl]]) = c("theta_edges", "theta_kstar2", "theta_gwesp")
    cat(sprintf("  loaded %s (%d chains, %d total draws)\n",
                fn, sum(ok), nrow(prior_data[[pl]])))
  }
}

# Discover NDG replicate files (posterior-20rep-ndg-p2.RDS, etc.)
ndg_files = sort(list.files("results",
                            pattern = "^posterior-20rep-ndg-p.*\\.RDS$",
                            full.names = TRUE))
ndg_greens = colorRampPalette(c("#2d8c2d", "#0a3d0a"))(max(length(ndg_files), 1))
for (k in seq_along(ndg_files)) {
  fn    = ndg_files[k]
  label = sub("^posterior-20rep-", "", sub("\\.RDS$", "", basename(fn)))
  reps  = readRDS(fn)
  ok    = !sapply(reps, is.null)
  chains = lapply(reps[ok], function(r) r$samples)
  prior_data[[label]] = as.data.frame(do.call(rbind, chains))
  colnames(prior_data[[label]]) = c("theta_edges", "theta_kstar2", "theta_gwesp")
  prior_colors[label] = ndg_greens[k]
  cat(sprintf("  loaded %s (%d chains, %d total draws)\n",
              fn, sum(ok), nrow(prior_data[[label]])))
}

if (length(prior_data) >= 2) {
  param_names = c("theta_edges", "theta_kstar2", "theta_gwesp")

  sens_path = "results/prior-sensitivity-20rep.png"
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

