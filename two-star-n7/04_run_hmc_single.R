# 04_run_hmc_single.R

	if(!file.exists("src/hmc")) {
		cat("\n\nAttention: Run \"make all\" in src directory first!\n\n\n")
	  # cd <this folder>/src && make all
	} else {

# Move into the src directory
	setwd("./src")

# Variables ready to be passed to C code
	data 					= "../data/g7.txt"
	results_file 	= "../chains/hmc_results.txt" 	# holds the MCMC chain
	log_file	 		= "../chains/hmc_log.txt" 			# holds the MCMC log file (acceptance rate etc.)
	time_file 		= "../chains/hmc_time.txt" 			# holds the timings
	seed=3000	# seed

# MPLE coefficients for g7 ~ edges + kstar(2), computed via
#   ergm(g ~ edges + kstar(2), estimate = "MPLE")$coefficients
# (same call as 03_build_ndg_prior.R; deterministic given the graph).
mplecoef = c(-3.0385268905675438, 0.6771034983042240)

# Prior: "normal", "uniform", or "ndg"
	prior_type = "ndg"
	ndg_file   = "../data/ndg_prior.txt"    # warm-start samples (only used if prior_type == "ndg");
	                                # only the (s1, s2) columns are read: hull indicator
	                                # and reference-theta columns are discarded
	ndg_p      = 3                # strength exponent on log Phat; ignored unless ndg;
	                                # larger p penalizes degenerate regions more strongly
	                                # (q is fixed at 0 in this build)

# Target acceptance rate for Nesterov dual-averaging step-size tuning
# (delta in hmc_autotune2; applied during the first 1000 warm-up iterations).
# Common values:
#   0.651 -- HMC asymptotic optimum (Beskos, Pillai, Roberts, Sanz-Serna,
#            Stuart 2013, Bernoulli 19(5A):1501-1534) and the paper default
	target_accept = 0.651

# Total number of HMC iterations (including the first 1000 warm-up steps
# used for dual-averaging eps tuning). Passing 0 triggers an interactive
# prompt in the C driver.
	main_iters = 100000

# Number of warm-up rows to drop before posterior summaries / plot.
# Matches the dual-averaging window in hmc.c (i < 1000), so the kept
# samples are draws under the FINAL tuned eps. 05_run_hmc_replicates.R
# uses the same value -- keep these in sync if one is changed.
	warmup     = 1000

# Run HMC (mass matrix is in src/hmc.c file)
# NOTE: main_iters is wrapped in formatC(..., format = "d") because R's default
# numeric->string conversion uses scientific notation for >= 1e+05
# (as.character(1000000) returns "1e+06"), and the C-side sscanf("%zu") would
# stop at the 'e' and read main_iters as 1, making the run finish instantly.
	args = c(data,"es",mplecoef[1],mplecoef[2],"tnt",0.5,100,"bri",
	         prior_type,ndg_file,ndg_p,seed,results_file,log_file,time_file,
	         target_accept,formatC(main_iters, format = "d"),"yes")
	system2("./hmc", args=args)

# After algorithm get HMC posterior mean
	hmc = read.table(results_file,header=TRUE)
	print(apply(hmc,2,mean))

# Move back to base directory
	setwd("../")

	}
	
##############################
#                            #
#   Load gy and freq         #
#                            #
##############################

# Name of the run configured at the top of this script.
filename = ifelse(prior_type == "ndg",
                  paste0("posterior-1chain-ndg-p", sprintf("%g", ndg_p)),
                  paste0("posterior-1chain-", prior_type))

# Uncomment to overwrite the stored chain with the one just simulated.
# saveRDS(hmc, file = paste0("./results/", filename, ".RDS"))

# The reporting below runs off a STORED chain, not the one simulated above, so
# that the figures can be regenerated without re-running the sampler. Set this
# to any file in results/ (posterior-1chain-normal, -uniform, -ndg-p1, ...).
filename = "posterior-1chain-normal"
hmc = readRDS(paste0("./results/", filename, ".RDS"))
	
# These are needed for EtaToMu and PEta
{
  bergm_dir = "data/"
  load(paste0(bergm_dir, "g7.RData"))
  edges <- rowSums(g7)
  load(paste0(bergm_dir, "g7kstars.RData"))
  twostars = g7kstars[,3]
  et <- edges + 22 * twostars
  freq <- table(et)
  etn <- as.numeric(names(freq))
  etk = trunc(etn / 22)
  ete <- etn - 22 * etk
  gy <- cbind(ete, etk)
  rm(twostars, edges, et, etk, ete, etn)
}

# Observed statistics (target in mu-space)
target = c(7, 14)

# Find which row of gy corresponds to the observed stats
select = which(gy[,1] == target[1] & gy[,2] == target[2])
cat("Observed stats (target):", target, "-> gy row:", select, "\n")



##############################
#                            #
#   Eta -> Mu transform      #
#                            #
##############################

# Drop the first `warmup` rows (dual-averaging adaptation phase, where eps
# is changing each iteration -- not draws from the stationary distribution).
# Same slice 05_run_hmc_replicates.R applies, so PMSE / posterior summaries
# computed here match the per-replicate values produced by 10 exactly.
samples_eta = as.matrix(hmc)[(warmup + 1L):nrow(hmc), , drop = FALSE]
samples_mu = samples_eta
for (i in 1:nrow(samples_eta)){
  samples_mu[i,] = RcppERGM::EtaToMu(samples_eta[i,], gy, freq)$`mean-value`
}
colnames(samples_eta) = c("eta1", "eta2")
colnames(samples_mu) = c("mu1", "mu2")


##############################
#                            #
#   Summary statistics       #
#                            #
##############################

# Mean and Median
posterior_mean   = colMeans(samples_mu)
posterior_median = apply(samples_mu, 2, median)

cat("Target:          ", target, "\n")
cat("Posterior mean:  ", round(posterior_mean,3), "\n")
cat("Posterior median:", round(posterior_median,3), "\n")

# PMSE (Posterior Mean Squared Error) = Variance + Bias^2, per dimension
true_mu1 = target[1]
true_mu2 = target[2]
pm1 = posterior_mean[1]
pm2 = posterior_mean[2]
pmse_mu1 = var(samples_mu[,1]) + (pm1 - true_mu1)^2
pmse_mu2 = var(samples_mu[,2]) + (pm2 - true_mu2)^2
cat("PMSE:            ", round(c(pmse_mu1, pmse_mu2), 3), "\n")

# Bin posterior samples onto the same grid as `grd` / `eta_mv` below, so
# the cells carry probability MASS (not density). Two advantages over the
# previous MASS::kde2d approach:
#   1. Cell indices match grd row indices exactly -- no off-by-one /
#      smoothing-bandwidth concerns when computing E[PND] under the posterior.
#   2. `posterior` sums to 1, so the post-predictive probability is just
#      sum(pnd * posterior) -- no Riemann factor (* dx) needed.
byf      = 0.01
x_breaks = seq(0,  21, by = byf)
y_breaks = seq(0, 105, by = byf * 105/21)
x_idx = round(samples_mu[, 1] / byf)             + 1
y_idx = round(samples_mu[, 2] / (byf * 105/21))  + 1
x_idx = pmax(1, pmin(x_idx, length(x_breaks)))
y_idx = pmax(1, pmin(y_idx, length(y_breaks)))
bin_counts = table(paste(x_idx, y_idx, sep = "_"))
grd_keys = paste(rep(seq_along(x_breaks), times = length(y_breaks)),
                 rep(seq_along(y_breaks), each  = length(x_breaks)), sep = "_")
posterior = as.numeric(bin_counts[match(grd_keys, names(bin_counts))])
posterior[is.na(posterior)] = 0
posterior = posterior / nrow(samples_mu)

# Load grid data for PND and KL (precomputed for g7)
onhull = readRDS(file = paste0(bergm_dir, "onhull.RDS"))
inhull = readRDS(file = paste0(bergm_dir, "inhull.RDS"))
grd = as.matrix(expand.grid(ete = seq(0, 21, by = byf),
                              etk = seq(0, 105, by = byf * 105/21)))
eta_mv = cbind(grd[, c(1,2)], NA, NA, 0)
eta_mv[onhull, 3:4] = readRDS(file = paste0(bergm_dir, "onhull_eta.RDS"))[, 1:2]

# Posterior predictive probability: integral of PND * posterior over mu-space
cat("Computing posterior predictive probability...\n")
pnd = rep(0, nrow(eta_mv))
for(i in (1:nrow(eta_mv))[onhull]) {
  pnd[i] = RcppERGM::PND(eta_mv[i, 3:4], gy, freq)$probability
}
post_pred_prob = sum(pnd * posterior)
cat("Posterior predictive P(Y in C):", round(post_pred_prob, 3), "\n")

# KL divergence: binned HMC posterior vs true grid posterior loaded from
# results/posterior-exact-<prior>.RDS. Each true-posterior file is a length-
# 4,414,201 vector of DENSITIES on the same expand.grid(ete = seq(0,21,by=byf),
# etk = seq(0,105,by=byf*105/21)) ordering as `posterior`, `grd`, and
# `eta_mv`. We renormalize it to a probability MASS so it lives on the
# same scale as the binned `posterior` (which sums to 1).
true_post_file = ifelse(prior_type == "ndg",
                        paste0("results/posterior-exact-ndg-p", sprintf("%g", ndg_p), ".RDS"),
                        paste0("results/posterior-exact-", prior_type, ".RDS"))
if (file.exists(true_post_file)) {
  true_post = readRDS(true_post_file)
  stopifnot(length(true_post) == length(posterior))   # cell-count must match

  # Normalize to probability mass (sums to 1, same units as `posterior`).
  pos = true_post / sum(true_post)

  # Block for zeros: KL is finite only where BOTH distributions have mass.
  # Cells with pos == 0 contribute 0 * log(0/q) = 0 by convention -- safe
  # to drop. Cells with pos > 0 but posterior == 0 would give pos * log(pos/0)
  # = +Inf (true posterior puts mass where our finite HMC sample didn't
  # visit). After dropping those cells we renormalize each side to sum to
  # 1 on the intersection, so we are comparing the two CONDITIONAL distri-
  # butions on a common support. This guarantees KL >= 0 and gives a clear
  # interpretation: "given we're in the intersection, how different are
  # the true and HMC posteriors?"
  both_nz = pos != 0 & posterior != 0
  p_n     = pos[both_nz]       / sum(pos[both_nz])
  q_n     = posterior[both_nz] / sum(posterior[both_nz])
  kl_div  = sum(p_n * log(p_n / q_n))

  # Diagnostics: how much of pos's mass is excluded? If this is large the
  # KL is being computed on only a fraction of the true posterior's support.
  excluded_mass = sum(pos[!both_nz])
  cat(sprintf("KL diagnostics: %d / %d cells in support intersection (%.2f%%)\n",
              sum(both_nz), length(pos), 100 * sum(both_nz) / length(pos)))
  cat(sprintf("                pos mass excluded (no HMC samples): %.4f\n",
              excluded_mass))
  cat(sprintf("KL divergence (true || HMC): %.6f\n", kl_div))
} else {
  cat(sprintf("KL divergence: (skipped -- %s not found)\n", true_post_file))
}

# Print all metrics together
cat("\n--- Summary ---\n")
cat(sprintf("  Target:              (%.2f, %.2f)\n", target[1], target[2]))
cat(sprintf("  Posterior mean:      (%.4f, %.4f)\n", posterior_mean[1], posterior_mean[2]))
cat(sprintf("  Posterior median:    (%.4f, %.4f)\n", posterior_median[1], posterior_median[2]))
cat(sprintf("  PMSE (mu1, mu2):     (%.3f, %.3f)\n", pmse_mu1, pmse_mu2))
cat(sprintf("  Post. pred. P(Y in C): %.3f\n", post_pred_prob))
if(exists("kl_div")) cat(sprintf("  KL divergence:       %.6f\n", kl_div))

##############################
#                            #
#   Mu-space posterior plot  #
#                            #
##############################

library(ggplot2)

# Points to overlay
points_df = data.frame(
  x     = c(target[1], posterior_median[1]),
  y     = c(target[2], posterior_median[2]),
  label = c("Target", "Median")
)


  png(
    paste0("results/", filename, "-mu-overlay.png"),
    width     = 6,
    height    = 5,
    units     = "in",
    res       = 300,
  )
  ggplot(data.frame(samples_mu), aes(x = mu1, y = mu2)) +
    geom_hex(bins = 500, aes(fill = after_stat(count / sum(count)))) +
    scale_fill_distiller(palette = "Blues",
                         direction = 1,
                         name = "Density") +
    geom_point(data = points_df, aes(x = x, y = y, colour = label, shape = label),
               inherit.aes = FALSE, size = 2) +
    scale_colour_manual(values = c("Target" = "red", "Median" = "green3"),
                        name = "") +
    scale_shape_manual(values = c("Target" = 16, "Median" = 16),
                       name = "") +
    coord_cartesian(xlim = c(0, 15), ylim = c(0, 50)) +
    labs(
      x = expression(mu[1] ~ ": Edges"),
      y = expression(mu[2] ~ ": Two-Stars")
    ) +
    theme_bw() +
    theme(
      panel.grid.minor = element_blank(),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10),
      legend.title = element_text(size = 11),
      legend.text = element_text(size = 9),
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white"),
      legend.position = c(.85, .24)
    )
  dev.off()

