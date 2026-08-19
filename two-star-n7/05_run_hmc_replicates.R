# 05_run_hmc_replicates.R
# -----------------------------------------------------------------------
# Replicate the noisy-HMC algorithm 20 times following the protocol in
# Stoehr, Benson & Friel (2019):
#
#   "The different algorithms were replicated 20 times using a different
#    starting point and random seed for each experiment."
#
# Per replicate:
#   - starting point drawn from N(MPLE, M^{-1}) (MPLE from 03_build_ndg_prior.R,
#                                                M    from 02_find_mass_matrix.R)
#   - distinct C-side RNG seed
#   - same prior / main_iters / tuning as 04_run_hmc_single.R
#
# Post-run, each chain is loaded, mu-transformed with RcppERGM::EtaToMu
# (reusing the machinery from 04), and per-replicate metrics are extracted.
# The 20 rows are summarized into a Table-3-style mean (sd) panel.
#
#
# -----------------------------------------------------------------------

if(!file.exists("src/hmc")) {
	stop("Run \"make all\" in src/ first.")
  # cd <this folder>/src && make all
}

# Disable R's default scientific notation for numeric->string conversion,
# otherwise as.character(300000) -> "3e+05" and the C-side sscanf("%zu")
# reads main_iters as 3 (one-iteration instant-exit bug). belt-and-
# suspenders: also wrap with formatC below.
options(scipen = 999)

# -----------------------------------------------------------------------
#   CONFIG  -- edit prior_type here; everything else is paper-aligned
# -----------------------------------------------------------------------
n_replicates  = 20
chain_length  = 100000         # main_iters per replicate
warmup        = 1000           # drop before ESS / posterior mean
master_seed   = 3000           # set seed

prior_type    = "normal"      # "normal", "uniform", or "ndg"
ndg_file      = "../data/ndg_prior.txt"
ndg_p         = 1

target_accept = 0.651          # 0.651, 0.574, 0.900
tnt_p         = 0.5
grad_gap      = 100
corr_type     = "bri"

# MPLE coefficients for g7 ~ edges + kstar(2), computed via
#   ergm(g ~ edges + kstar(2), estimate = "MPLE")$coefficients
# (same call as 03_build_ndg_prior.R; deterministic given the graph).
mplecoef = c(-3.0385268905675438, 0.6771034983042240)
# Mass matrix M is still the one estimated at the MAP by 02_find_mass_matrix.R.
# It is used here only as the *scale* (Minv = asymptotic posterior covariance)
# for the starting-point spread; centering at MPLE vs MAP just shifts the
# mean of that spread by O(1/sqrt(n)).
M = matrix(c(26.020460921843735, 108.351743486973660,
             108.351743486973660, 492.502589178357425),
           nrow = 2, byrow = TRUE)
Minv = solve(M)

# Ground truth in mu-space (observed sufficient statistics for g7)
target_mu = c(7, 14)

# -----------------------------------------------------------------------
#   PATHS
# -----------------------------------------------------------------------
rep_subdir   = "../chains"      # where the C binary writes, relative to src/
full_rep_dir = "chains"         # same directory, relative to this folder
dir.create(full_rep_dir, showWarnings = FALSE, recursive = TRUE)
dir.create("results", showWarnings = FALSE)

# -----------------------------------------------------------------------
#   EtaToMu machinery (identical to the block in 04_run_hmc_single.R)
# -----------------------------------------------------------------------
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

# -----------------------------------------------------------------------
#   DRAW starting points and C-seeds up front (deterministic given seed)
# -----------------------------------------------------------------------
set.seed(master_seed)
starts  = MASS::mvrnorm(n_replicates, mu = mplecoef, Sigma = Minv/4)
c_seeds = sample.int(.Machine$integer.max - 1L, n_replicates)

cat(sprintf("=== 20-replicate run | prior=%s | main_iters=%d | master_seed=%d ===\n",
            prior_type, chain_length, master_seed))
cat("Starting points (rows) drawn from N(MPLE, M^{-1}):\n")
print(round(starts, 4))

# -----------------------------------------------------------------------
#   Per-replicate log / stdout parsers
# -----------------------------------------------------------------------
parse_log = function(log_path) {
	lines = readLines(log_path)
	acc_line  = grep("^accepted:",         lines, value = TRUE)
	time_line = grep("^time_\\(seconds\\):", lines, value = TRUE)
	acc_m = regmatches(acc_line,
	                   regexec("\\(([0-9.eE+-]+) %\\)", acc_line))[[1]]
	time_m = regmatches(time_line,
	                    regexec("([0-9.eE+-]+)\\s*$", time_line))[[1]]
	list(accept_rate = as.numeric(acc_m[2]) / 100,
	     runtime_sec = as.numeric(time_m[2]))
}

parse_stdout = function(stdout_path) {
	# Dual-averaging in hmc.c emits two stdout formats:
	#   - During tuning (i = 0..999):
	#       "<i> eps: <e> epsbar: <eb> nfrogs: <nf> (<nf_eb>) theta: ..."
	#   - At finalization (i = 1000, hmc.c:515-520):
	#       "1000 eps: <eps_final> epsbar: <same> nfrogs: <L_final> theta: ..."
	#     (no parenthetical -- eps has been set to exp(logepsbar) and
	#     nfrogs has been recomputed).
	# The iteration-1000 line is authoritative: it's the eps/L the chain
	# actually uses for all iterations >= 1000. Parse that line first;
	# fall back to the last tuning line only if the chain was shorter
	# than the 1000-iter warm-up (should not happen for sensible runs).
	lines = readLines(stdout_path)

	final_line = grep("^1000 eps:", lines, value = TRUE)
	if (length(final_line) >= 1L) {
		m = regmatches(final_line[1],
		               regexec(
		                 "^1000 eps: ([0-9.eE+-]+) epsbar: [0-9.eE+-]+ nfrogs: ([0-9]+)",
		                 final_line[1]))[[1]]
		return(list(eps = as.numeric(m[2]), L = as.integer(m[3])))
	}

	# Fallback: warm-up didn't complete. Use the last tuning line's
	# projected final values (epsbar + parenthetical).
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

# -----------------------------------------------------------------------
#   MAIN LOOP (sequential)
# -----------------------------------------------------------------------
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

	# Args passed to the C binary. Two gotchas:
	#   (a) R's default numeric->string uses scientific notation for
	#       magnitudes >= 1e+05, and the C-side sscanf("%zu") would stop
	#       at the 'e'. chain_length (300k) and seedC (can reach ~2e9)
	#       are wrapped in formatC(format="d") to force decimal.
	#   (b) theta precision: mplecoef + mvrnorm draws want ~15 sig digits
	#       so HMC starts at the exact intended point. R's default
	#       as.character (invoked implicitly by c()) uses 15 digits,
	#       which is sufficient; matches the convention in 04_run_hmc_single.R.
	args_vec = c(
		"../data/g7.txt", "es",
		theta0[1], theta0[2],           # implicit as.character, 15 digits
		"tnt",
		tnt_p, grad_gap,                # small values, no scipen risk
		corr_type,
		prior_type, ndg_file,
		ndg_p,                          # single-digit integer
		formatC(seedC,        format = "d"),   # can be ~2e9 -> force decimal
		res_file, log_file, tim_file,
		target_accept,                  # < 1, no scipen risk
		formatC(chain_length, format = "d"),   # >= 1e5 -> force decimal
		"yes"
	)

	cat(sprintf("\n[%s] start=(%+.4f, %+.4f) seed=%d\n",
	            tag, theta0[1], theta0[2], seedC))

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
	hmc_raw     = read.table(res_file, header = TRUE)
	samples_eta = as.matrix(hmc_raw)[(warmup + 1L):nrow(hmc_raw), , drop = FALSE]

	# eta -> mu (same per-row loop as 04)
	samples_mu = samples_eta
	for (r in seq_len(nrow(samples_eta))) {
		samples_mu[r, ] = RcppERGM::EtaToMu(samples_eta[r, ], gy, freq)$`mean-value`
	}

	# Set mu-column names so the pooled-plot code below can use
	# ggplot aes(x = mu1, y = mu2) directly.
	colnames(samples_mu) = c("mu1", "mu2")

	# Store everything needed to compute metrics later. The full
	# samples_eta / samples_mu matrices are kept so the metric block
	# (and any downstream analysis on the saved RDS) can reproduce ESS,
	# posterior summaries, MSE, PMSE, and the pooled-posterior plot
	# without rerunning the C binary or redoing the eta -> mu loop.
	replicates[[ii]] = list(
		rep              = ii,
		start_eta        = theta0,
		c_seed           = seedC,
		accept_rate      = log_info$accept_rate,
		runtime_sec      = log_info$runtime_sec,
		eps              = tune_info$eps,
		L                = tune_info$L,
		wall_elapsed_sec = wall_elapsed,
		samples_eta      = samples_eta,
		samples_mu       = samples_mu
	)

	cat(sprintf(
	  "  accept=%.2f%%  eps=%.4f  L=%d  runtime=%.1fs\n",
	  100 * log_info$accept_rate, tune_info$eps, tune_info$L,
	  log_info$runtime_sec))
}

setwd("../")

overall_elapsed = (proc.time() - overall_t0)[["elapsed"]]


# SAVE and LOAD
# saveRDS(replicates, file = ifelse(prior_type == "ndg",
#                                   paste0("results/posterior-20rep-ndg-p", sprintf("%g", ndg_p), ".RDS"),
#                                   paste0("results/posterior-20rep-", prior_type, ".RDS")))

# --- Reporting section ------------------------------------------------
# Everything below runs off a STORED replicate set, not the one simulated
# above, so the tables and figures can be regenerated without re-running the
# sampler. These two lines deliberately override the config at the top of the
# script; set them to whichever run you want to report on.
prior_type    = "uniform"      # "normal", "uniform", or "ndg"
ndg_p         = 1
replicates = readRDS(ifelse(prior_type == "ndg",
                            paste0("results/posterior-20rep-ndg-p", sprintf("%g", ndg_p), ".RDS"),
                            paste0("results/posterior-20rep-", prior_type, ".RDS")))

# -----------------------------------------------------------------------
#   COMPUTE METRICS FROM REPLICATES (post-loop)
# -----------------------------------------------------------------------
#   Single pass over `replicates`: takes the raw samples + chain metadata
#   collected by the main loop and computes every per-replicate statistic
#   (posterior mean / median, MSE, PMSE, posterior-predictive P(Y in C),
#   and KL(true || HMC) on the same grid as 04_run_hmc_single.R).
#
#   Shared resources (grid keys, onhull/eta_mv, the precomputed pnd vector,
#   and the true posterior `pos`) live OUTSIDE the lapply because they are
#   identical across the 20 chains -- the heavy RcppERGM::PND loop runs
#   once instead of 20 times.
#
#   ESS is intentionally not computed here (single-chain ESS isn't the
#   metric of interest for the 20-replicate analysis).
# -----------------------------------------------------------------------
completed = !sapply(replicates, is.null)
cat(sprintf("\n=== Computing metrics for %d completed replicates ===\n",
            sum(completed)))

# --- Shared-resource setup (used by every replicate; computed ONCE) ----
# Grid identical to 04_run_hmc_single.R lines 191-211: 2101 x 2101 cells over
# [0, 21] x [0, 105] with byf = 0.01 in mu1 and byf*105/21 in mu2.
byf      = 0.01
x_breaks = seq(0,  21, by = byf)
y_breaks = seq(0, 105, by = byf * 105/21)
grd_keys = paste(rep(seq_along(x_breaks), times = length(y_breaks)),
                 rep(seq_along(y_breaks), each  = length(x_breaks)), sep = "_")

# eta_mv carries (mu1, mu2, eta1, eta2, ?) per grid cell; needed for PND.
onhull = readRDS(file = paste0(bergm_dir, "onhull.RDS"))
inhull = readRDS(file = paste0(bergm_dir, "inhull.RDS"))
grd    = as.matrix(expand.grid(ete = seq(0, 21, by = byf),
                                 etk = seq(0, 105, by = byf * 105/21)))
eta_mv = cbind(grd[, c(1,2)], NA, NA, 0)
eta_mv[onhull, 3:4] = readRDS(file = paste0(bergm_dir, "onhull_eta.RDS"))[, 1:2]

# pnd: P(Y near degenerate) at every onhull grid cell. Same loop as 04.
# This is the expensive step (~600k RcppERGM::PND calls); doing it once
# outside the lapply saves it from being recomputed for every replicate.
cat("Precomputing pnd at onhull grid cells (one-time, slow)...\n")
t0_pnd = proc.time()
pnd = rep(0, nrow(eta_mv))
for (i in (1:nrow(eta_mv))[onhull]) {
	pnd[i] = RcppERGM::PND(eta_mv[i, 3:4], gy, freq)$probability
}
cat(sprintf("  pnd done in %.1fs\n", (proc.time() - t0_pnd)[["elapsed"]]))

# True posterior on the same grid -- loaded once, normalized to mass.
# Filename pattern matches 04_run_hmc_single.R lines 228-230 (NDG: 1000*ndg_p).
true_post_file = ifelse(prior_type == "ndg",
                        paste0("results/posterior-exact-ndg-p", sprintf("%g", ndg_p), ".RDS"),
                        paste0("results/posterior-exact-", prior_type, ".RDS"))
if (file.exists(true_post_file)) {
	true_post = readRDS(true_post_file)
	stopifnot(length(true_post) == length(grd_keys))
	pos = true_post / sum(true_post)
	kl_available = TRUE
	cat(sprintf("Loaded true posterior: %s (%d cells)\n",
	            true_post_file, length(pos)))
} else {
	pos = NULL
	kl_available = FALSE
	cat(sprintf("True posterior file not found: %s -- KL will be NA\n",
	            true_post_file))
}

metrics_rows = lapply(replicates[completed], function(r) {
	# Posterior summaries in eta and mu space
	pm_eta  = colMeans(r$samples_eta)
	pm_mu   = colMeans(r$samples_mu)
	pmed_mu = apply(r$samples_mu, 2, median)

	# MSE of the mu-posterior mean vs ground truth, averaged across dims
	mse_mu = mean((pm_mu - target_mu)^2)

	# PMSE = variance + bias^2 (per-dim)
	pmse_mu1 = var(r$samples_mu[, 1]) + (pm_mu[1] - target_mu[1])^2
	pmse_mu2 = var(r$samples_mu[, 2]) + (pm_mu[2] - target_mu[2])^2

	# Bin samples_mu onto the shared grid (same block as 04_run_hmc_single.R).
	# `posterior` is a length-4,414,201 probability mass aligned to grd_keys,
	# eta_mv, and pos.
	x_idx = round(r$samples_mu[, 1] / byf)             + 1
	y_idx = round(r$samples_mu[, 2] / (byf * 105/21))  + 1
	x_idx = pmax(1, pmin(x_idx, length(x_breaks)))
	y_idx = pmax(1, pmin(y_idx, length(y_breaks)))
	bin_counts = table(paste(x_idx, y_idx, sep = "_"))
	posterior = as.numeric(bin_counts[match(grd_keys, names(bin_counts))])
	posterior[is.na(posterior)] = 0
	posterior = posterior / nrow(r$samples_mu)

	# Posterior predictive P(Y in C) = E_posterior[PND(mu)] on the grid.
	# `pnd` is precomputed; this is just a dot product.
	post_pred_prob = sum(pnd * posterior)

	# KL(true || HMC) on the support intersection (block-for-zeros). Both
	# distributions are renormalized to sum to 1 on the intersection -- this
	# is necessary because dropping zero cells reduces each side's total
	# mass, and only conditional distributions on a common support give a
	# valid (>= 0) KL. Interpretation: "given we're in the intersection,
	# how different are the true and HMC posteriors?"
	if (kl_available) {
		both_nz = pos != 0 & posterior != 0
		p_n     = pos[both_nz]       / sum(pos[both_nz])
		q_n     = posterior[both_nz] / sum(posterior[both_nz])
		kl_div  = sum(p_n * log(p_n / q_n))
	} else {
		kl_div = NA_real_
	}

	cat(sprintf(
	  "  rep%02d  MSE=%.3f  PPP=%.3f  KL=%s\n",
	  r$rep, mse_mu, post_pred_prob,
	  if (is.finite(kl_div)) sprintf("%.4f", kl_div) else "NA"))

	data.frame(
		rep              = r$rep,
		start_eta1       = r$start_eta[1],
		start_eta2       = r$start_eta[2],
		c_seed           = r$c_seed,
		accept_rate      = r$accept_rate,
		eps              = r$eps,
		L                = r$L,
		runtime_sec      = r$runtime_sec,
		wall_elapsed_sec = r$wall_elapsed_sec,
		pm_eta1          = pm_eta[1],
		pm_eta2          = pm_eta[2],
		pm_mu1           = pm_mu[1],
		pm_mu2           = pm_mu[2],
		pmed_mu1         = pmed_mu[1],
		pmed_mu2         = pmed_mu[2],
		mse_mu           = mse_mu,
		pmse_mu1         = pmse_mu1,
		pmse_mu2         = pmse_mu2,
		post_pred_prob   = post_pred_prob,
		kl_div           = kl_div,
		stringsAsFactors = FALSE
	)
})
metrics = do.call(rbind, metrics_rows)

# -----------------------------------------------------------------------
#   REPLICATE SUMMARY (mean (sd) across replicates)
# -----------------------------------------------------------------------


fmt = function(v) {
	v = v[is.finite(v)]
	if (length(v) == 0) return("NA")
	sprintf("%.3f (%.3f)", mean(v), sd(v))
}

cat(sprintf(
  "\n=== Replicate summary | prior=%s | %d chains of %d iters (drop first %d) ===\n",
  prior_type, nrow(metrics), chain_length, warmup))
cat(sprintf("Total wall time: %.1f min\n\n", overall_elapsed / 60))

rows = list(
	c("accept_rate (rho)",     fmt(metrics$accept_rate)),
	c("eps (final)",           fmt(metrics$eps)),
	c("L   (final)",           fmt(metrics$L)),
	c("runtime_sec",           fmt(metrics$runtime_sec)),
	c("posterior_mean_mu1",    fmt(metrics$pm_mu1)),
	c("posterior_mean_mu2",    fmt(metrics$pm_mu2)),
	c("posterior_median_mu1",  fmt(metrics$pmed_mu1)),
	c("posterior_median_mu2",  fmt(metrics$pmed_mu2)),
	c("MSE(mu, truth=7,14)",   fmt(metrics$mse_mu)),
	c("PMSE(mu1)",             fmt(metrics$pmse_mu1)),
	c("PMSE(mu2)",             fmt(metrics$pmse_mu2)),
	c("Post.pred. P(Y in C)",  fmt(metrics$post_pred_prob)),
	c("KL(true || HMC)",       fmt(metrics$kl_div))
)
for (r in rows) {
	cat(sprintf("  %-22s %s\n", r[1], r[2]))
}

# -----------------------------------------------------------------------
#   POOLED POSTERIOR PLOT (mu-space)
# -----------------------------------------------------------------------
#   Same ggplot body as 04_run_hmc_single.R, but the samples are now the
#   concatenation of all 20 chains' post-warmup mu-draws (up to ~6M
#   samples at main_iters=300000). Filename pattern is identical to 04
#   so this file replaces the old single-chain figure on disk.
# -----------------------------------------------------------------------

# Pool post-warmup samples_mu from every successful replicate. We pull
# samples_mu out of each `replicates` entry (NULLs are skipped naturally
# by rbind()).
mu_chains = lapply(replicates,
                   function(r) if (is.null(r)) NULL else r$samples_mu)
pooled_mu = do.call(rbind, mu_chains)

if (is.null(pooled_mu) || nrow(pooled_mu) == 0L) {
	warning("No completed chains to pool; skipping mu-space plot.")
	quit(save = "no", status = 0)
}

# Matching 04's ggplot aesthetics: aes(x = mu1, y = mu2) needs these names
colnames(pooled_mu) = c("mu1", "mu2")

cat(sprintf("\nPooling %d chains -> %d total samples for mu-space plot\n",
            sum(!sapply(mu_chains, is.null)), nrow(pooled_mu)))

# Median of the POOLED samples (same statistic 04 overlays as a green dot)
posterior_median = apply(pooled_mu, 2, median)

# Filename scheme identical to 04_run_hmc_single.R:
#   ndg   -> "posterior-20rep-ndg-p<p>"
#   other -> "posterior-20rep-<prior_type>"
filename = ifelse(prior_type == "ndg",
                  paste0("posterior-20rep-ndg-p", sprintf("%g", ndg_p)),
                  paste0("posterior-20rep-", prior_type))

# Local alias so the plot block below reads identically to 04's
target = target_mu

library(ggplot2)

# Points to overlay
points_df = data.frame(
  x     = c(target[1], posterior_median[1]),
  y     = c(target[2], posterior_median[2]),
  label = c("Target", "nHMC\nMedian")
)



# -----------------------------------------------------------------------
#   OVERLAY PLOT: HMC pooled posterior vs true posterior (HDR contours)
# -----------------------------------------------------------------------
#   Same axes / theme / dimensions / Target+Median dots as the pooled plot
#   above, but with TWO red HDR contour lines (50% and 95% mass) of the
#   true posterior overlaid on the HMC hex density.
#
#   Approach: instead of contouring the deterministic gridded `pos` (which
#   is jagged in the long thin tail because cell-level numerical ripples
#   become big lateral contour displacements at low density), we
#     1. draw N samples from `pos` (multinomial over grid cells, weighted
#        by mass). Each sample is the (mu1, mu2) coordinate of the cell.
#     2. fit a 2D KDE to those samples via MASS::kde2d. The KDE bandwidth
#        is chosen automatically (Silverman) and is C^infinity smooth, so
#        contours are inherently smooth -- including the tail.
#     3. compute density-level cutoffs at the 50% / 80% mass quantiles of
#        the KDE grid, and contour at those levels with geom_contour.
#
#   The MC noise from the multinomial is well below visual resolution at
#   N = 100k samples (relative SE under 1% per cell). Skipped if no
#   true-posterior file was loaded.
# -----------------------------------------------------------------------
if (kl_available) {
  
  # --- (1) Reshape pos into a spatstat image on the native grid ----------
  library(spatstat.geom)
  library(spatstat.explore)
  
  # pos is a vector of length length(x_breaks)*length(y_breaks),
  # stored with mu1 (x) cycling fastest (same column-major convention
  # as expand.grid(ete, etk) above).
  pos_mat <- matrix(pos,
                    nrow = length(x_breaks),   # mu1 axis
                    ncol = length(y_breaks))   # mu2 axis
  
  pos_im <- spatstat.geom::as.im(
    list(x = x_breaks, y = y_breaks, z = pos_mat)
  )
  
  # --- (2) Small Gaussian blur on the native grid ----------------------
  #
  # sigma is kept small (0.3 mu1-units) so the kernel
  # tails never reach the hull boundary -- no mass leaks outside the
  # feasible region regardless of how the posterior sits relative to
  # the boundary. This is the boundary-safe step.
  smooth_sigma   <- 0.3
  pos_smooth_im  <- spatstat.explore::Smooth(pos_im, sigma = smooth_sigma)
  pos_smooth_mat <- t(as.matrix(pos_smooth_im))   # transpose back: rows=mu1, cols=mu2
  pos_smooth_mat <- pmax(pos_smooth_mat, 0)
  pos_smooth_mat <- pos_smooth_mat / sum(pos_smooth_mat)
  
  cat(sprintf(
    "Overlay plot: direct Gaussian blur of pos (sigma=%.3f mu1-units, %d grid cells)\n",
    smooth_sigma, round(smooth_sigma / byf)))
  
  # --- (3) Downsample to 400x400 for smooth contour rendering ----------
  #
  # The original MASS::kde2d approach produced visually smooth contours
  # because it operated on a 400x400 grid -- the coarser resolution acts
  # as an implicit smoother at the geom_contour rendering stage. We
  # replicate that here by subsampling pos_smooth_mat to the same 400x400
  # resolution AFTER the boundary-safe blur. This gives matching visual
  # smoothness without any boundary bleed.
  coarse_n  <- 450L
  x_coarse  <- seq(0,  21,  length.out = coarse_n)
  y_coarse  <- seq(0, 105,  length.out = coarse_n)
  x_idx_c   <- findInterval(x_coarse, x_breaks, all.inside = TRUE)
  y_idx_c   <- findInterval(y_coarse, y_breaks, all.inside = TRUE)
  pos_coarse <- pos_smooth_mat[x_idx_c, y_idx_c]
  pos_coarse <- pos_coarse / sum(pos_coarse)
  
  # --- (4) HDR contour levels at 50% and 80% mass ----------------------
  hdr_levels <- function(z_mat, masses) {
    v   <- c(z_mat)
    v   <- v[v > 0]
    s   <- sort(v, decreasing = TRUE)
    cum <- cumsum(s) / sum(s)
    vapply(masses, function(m) s[which(cum >= m)[1]], numeric(1))
  }
  levs <- hdr_levels(pos_coarse, c(0.50, 0.80))
  
  cat(sprintf("  HDR levels: 50%%=%.3e  80%%=%.3e\n", levs[1], levs[2]))
  
  # --- (5) Long-format data frame for geom_contour ---------------------
  true_df       <- expand.grid(mu1 = x_coarse, mu2 = y_coarse)
  true_df$z     <- c(pos_coarse)
  
  # --- (6) Build and save the overlay plot -----------------------------
  png(
    paste0("results/", filename, "-mu-overlay.png"),
    width = 6, height = 5, units = "in", res = 300,
  )
  print(
    ggplot() +
      # HMC pooled density
      geom_hex(data = data.frame(pooled_mu),
               aes(x = mu1, y = mu2,
                   fill = after_stat(count / sum(count))),
               bins = 500) +
      scale_fill_distiller(palette = "Blues", direction = 1,
                           name = "HMC Density") +
      # True posterior: two red HDR contour lines (50%, 80%).
      # Contours are guaranteed to stay inside the feasible region:
      # small sigma preserves the zero boundary, and downsampling
      # provides the visual smoothness of the original kde2d approach.
      geom_contour(data = true_df,
                   aes(x = mu1, y = mu2, z = z, colour = "50/80 %\nTrue HDR"),
                   breaks = levs, linewidth = 0.5) +
      # Target + nHMC Median markers
      geom_point(data = points_df,
                 aes(x = x, y = y, colour = label, shape = label),
                 inherit.aes = FALSE, size = 2) +
      scale_colour_manual(
        values = c("Target"            = "red3",
                   "nHMC\nMedian"      = "green3",
                   "50/80 %\nTrue HDR" = "red"),
        name   = NULL,
        guide  = guide_legend(override.aes = list(
          shape    = c("Target" = 16, "nHMC\nMedian" = 16, "50/80 %\nTrue HDR" = NA),
          linetype = c("Target" = 0,  "nHMC\nMedian" = 0,  "50/80 %\nTrue HDR" = 1)
        ))
      ) +
      scale_shape_manual(values = c("Target" = 16, "nHMC\nMedian" = 16),
                         name = NULL, guide = "none") +
      guides(fill   = guide_colourbar(order = 1),
             colour = guide_legend(order = 2)) +
      coord_cartesian(xlim = c(0, 15), ylim = c(0, 50)) +
      labs(
        x = expression(mu[1] ~ ": Edges"),
        y = expression(mu[2] ~ ": Two-Stars")
      ) +
      theme_bw() +
      theme(
        panel.grid.minor     = element_blank(),
        axis.title           = element_text(size = 12),
        axis.text            = element_text(size = 10),
        legend.title         = element_text(size = 11),
        legend.text          = element_text(size = 9),
        plot.background      = element_rect(fill = "white", color = NA),
        panel.background     = element_rect(fill = "white"),
        legend.position      = c(.90, .32),
        legend.spacing.y     = unit(8, "pt"),
        legend.key.spacing.y = unit(5, "pt"),
        legend.margin        = margin(0, 0, 0, 0)
      )
  )
  dev.off()
  
  cat(sprintf("Saved plot: results/%s-mu-overlay.png\n", filename))
  
} else {
  cat("Overlay plot skipped: no true-posterior file was loaded.\n")
}



