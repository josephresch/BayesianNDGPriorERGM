/*
	prior.c
	=======

	Prior distributions for ERGM parameters: Normal, Uniform, and the
	adaptive Non-Degeneracy (NDG) prior.

	Overview
	--------
	  PRIOR_NORMAL :  N(mean, diag(var)) -- closed form, no MCMC state.
	  PRIOR_UNIFORM:  Uniform on [lower0,upper0] x [lower1,upper1].
	  PRIOR_NDG    :  Adaptive, data-driven prior that penalises theta
	                  values under which the ERGM produces sufficient
	                  statistics near the boundary of their lattice.

	The NDG prior in one paragraph
	------------------------------
	Define Phat(theta) as the probability (under the ERGM at parameter
	theta) that the sufficient statistic g(Y) lies strictly inside the
	empirical convex hull C of previously observed statistics. Informally,
	Phat(theta) is near 1 when theta produces "healthy" networks and
	crashes toward 0 when theta pushes the distribution toward degenerate
	regions (empty / full / star-like networks that sit on the hull
	boundary). The NDG prior is

	    log p_NDG(theta)  =  p_strength * log Phat(theta).

	Larger p_strength means a sharper penalty on degeneracy. p_strength=0
	collapses the prior to a flat improper prior (no penalty). Multiplying
	by p is how the paper's strength exponent enters; q is fixed at 0 in
	this build so the (1 - Phat) factor is not used.

	The self-normalised importance-sampling estimator
	-------------------------------------------------
	Computing Phat(theta) exactly would require enumerating every
	achievable network and summing the ERGM unnormalised density over
	the set whose statistics land in int(C) and over the complement --
	both of which are intractable. We estimate it by importance sampling.

	Given an auxiliary pool of m networks with statistics g_1, ..., g_m
	DRAWN FROM THE ERGM AT A REFERENCE PARAMETER theta_tilde, the
	self-normalised IS estimator is

	                  sum_i w_i * 1{g_i in int(C)}
	    Phat(theta) = ------------------------------------,
	                            sum_i w_i

	    w_i = exp( (theta - theta_tilde)^T g_i ).

	Why this is correct:
	    Under the ERGM at theta_tilde, the expectation of any function
	    h(g) is E_{theta_tilde}[h(g)] = sum_g h(g) * p_{theta_tilde}(g).
	    The IS reweighting factor w_i / E_{theta_tilde}[w] converts
	    samples drawn at theta_tilde into an estimate of E_{theta}[h(g)].
	    For h = 1{g in int(C)} we get exactly P(g in int(C) | theta) =
	    Phat(theta). The common normalising constant Z(theta_tilde) in
	    w_i cancels between numerator and denominator, which is why the
	    self-normalised form is free of the intractable partition
	    function. See the paper's Eq. (14).

	Why pick theta_tilde = theta (when possible)?
	    The variance of the IS estimator blows up when theta is far from
	    theta_tilde because a few pool samples dominate the weight mass.
	    The minimum-variance case is theta == theta_tilde: all weights
	    equal exp(0) = 1, and Phat collapses to the empirical proportion
	    (# pool samples in int(C)) / m. hmc.c exploits this by REFRESHING
	    the pool at theta_cur and theta_new at the leapfrog endpoints so
	    both log p_cur and log p_new are evaluated on pools centred at
	    their own theta.

	Gradient of log Phat (analytical, not FD)
	-----------------------------------------
	Differentiating log Phat(theta) = log N(theta) - log D(theta) where
	    N(theta) = sum_{i: in_hull} w_i,   D(theta) = sum_{all i} w_i
	gives, for each coordinate d in {0, 1}:

	    d/d theta_d log w_i  =  g_{i,d}
	    d/d theta_d log N     =  [sum_{i: in_hull} w_i g_{i,d}] / N
	    d/d theta_d log D     =  [sum_{all i}      w_i g_{i,d}] / D

	so

	    d/d theta_d log Phat  =  Ng_d / N  -  Dg_d / D,

	and the prior-gradient contribution is p_strength times this
	vector. This is what ndg_log_gradient computes in one pass.

	Numerical stability: log-sum-exp shift
	--------------------------------------
	If theta is far from theta_tilde, some w_i may overflow double and
	others underflow to zero. We shift by z_max = max_i (theta -
	theta_tilde)^T g_i before exponentiating: every stored weight is then
	exp(z_i - z_max) which is in [0, 1]. Because z_max is SUBTRACTED
	EVERYWHERE (numerator and denominator), the ratio Phat is invariant
	under this shift. Crucially, z_max must be the max over ALL pool
	samples (not just the interior ones) or the numerator and
	denominator would be scaled differently and the cancellation would
	break.

	Pool lifetime
	-------------
	The pool is REPLACED wholesale on every leapfrog step via
	prior_ndg_set_pool. The hull, by contrast, GROWS monotonically --
	every pool sample we have ever seen stays in the hull. This
	asymmetry matters because (a) we want the hull to be as tight and
	stable as possible (it's defining the "non-degenerate" region), and
	(b) we want the pool to be centred at the current theta so the IS
	estimator has minimum variance.

	No ground-truth knowledge
	-------------------------
	This module never reads a precomputed hull indicator or enumeration.
	It operates purely on the integer sufficient statistics handed to it
	via prior_ndg_set_pool or prior_ndg_warm_start.
*/

#include "prior.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define NEG_INF (-1.0/0.0)
#define POS_INF ( 1.0/0.0)

/* ================================================================== */
/*  NDG static helpers (estimator kernels)                            */
/* ================================================================== */

/*
	ndg_log_density: compute log p_NDG(theta) from the current pool.

	Implements

	                    sum_i w_i * 1{in_hull_i}
	    Phat(theta) = ----------------------------,
	                          sum_i w_i

	    w_i = exp( (theta - theta_tilde)^T g_i ),

	    log p_NDG(theta) = p_strength * log Phat(theta).

	Returns NEG_INF if:
	  - the pool is empty (ndg_m == 0), or
	  - every pool sample is OUTSIDE the hull (num == 0, so Phat == 0
	    and log Phat == -inf). In the accept/reject step hmc.c treats
	    NEG_INF as "proposal rejected", which is the correct behavior:
	    the IS estimate says this theta has zero non-degeneracy
	    probability under the current pool.

	p_strength == 0 short-circuits to log density = 0 (a flat improper
	prior): no need to touch the pool at all.

	Two-pass structure
	------------------
	Pass 1 computes z_max over ALL samples. Pass 2 accumulates weights.
	We need two passes because the shift must be known before the
	individual w_i = exp(z_i - z_max) can be formed.

	Cost: O(m) per call. For the typical g7 pool (m = grad_ndraws = 10)
	that's 20 multiply-adds plus 10 exp() calls per evaluation.
*/
static double ndg_log_density(PRIOR *prior, double *theta)
{
	if(prior->ndg_m <= 0) return NEG_INF;       /* no pool -> estimate undefined */
	if(prior->ndg_p == 0.0) return 0.0;         /* strength 0 -> flat prior */

	double d0 = theta[0] - prior->ndg_tref[0];
	double d1 = theta[1] - prior->ndg_tref[1];

	/* ---- Pass 1: find the shift z_max over the whole pool. ---- */
	/* Must be over ALL samples (interior and exterior) so the shift
	   cancels exactly in both sums in pass 2. A shift computed only
	   over interior points would bias the denominator. */
	double z_max = NEG_INF;
	int i;
	for(i = 0; i < prior->ndg_m; ++i)
	{
		double z = d0 * (double)prior->ndg_g1[i]
		         + d1 * (double)prior->ndg_g2[i];
		if(z > z_max) z_max = z;
	}

	/* ---- Pass 2: accumulate shifted weight sums. ---- */
	/* Every w_i = exp(z_i - z_max) is in (0, 1] because z_i <= z_max
	   by construction. This avoids overflow regardless of how far
	   theta has drifted from theta_tilde. */
	double num_shifted   = 0.0;   /* sum over samples with in_hull[i] == 1 */
	double denom_shifted = 0.0;   /* sum over all samples */
	for(i = 0; i < prior->ndg_m; ++i)
	{
		double z = d0 * (double)prior->ndg_g1[i]
		         + d1 * (double)prior->ndg_g2[i];
		double w = exp(z - z_max);
		denom_shifted += w;
		if(prior->ndg_in_hull[i])
			num_shifted += w;
	}

	/* num == 0: Phat == 0, log Phat = -inf. denom == 0 shouldn't
	   happen because at least one w_i = exp(0 - ...) = exp of a
	   non-negative number is positive, but we guard anyway. */
	if(num_shifted   <= 0.0) return NEG_INF;
	if(denom_shifted <= 0.0) return NEG_INF;

	/* log Phat = log(num / denom) = log(num) - log(denom). The z_max
	   shift vanishes here: if we'd left it in, both log-terms would
	   have an extra z_max addend that cancels on subtraction. */
	double log_phat = log(num_shifted) - log(denom_shifted);
	return prior->ndg_p * log_phat;
}

/*
	ndg_log_gradient: analytical gradient of log p_NDG w.r.t. theta.

	Derived from d/d theta_d log Phat = Ng_d/N - Dg_d/D where
	    N    = sum_{i: in_hull} w_i               (interior mass)
	    D    = sum_i            w_i               (total mass)
	    Ng_d = sum_{i: in_hull} w_i * g_{i,d}     (interior first moment)
	    Dg_d = sum_i            w_i * g_{i,d}     (total first moment)

	Multiply by p_strength to get the gradient of log p_NDG.

	One-pass accumulation
	---------------------
	All four running sums plus a fifth (denom_shifted to bookkeep) can
	be accumulated in a single pass over the pool after we know z_max.
	That makes the gradient cost only marginally more than the density
	cost (2 more mul-adds per sample), which matters because hmc.c
	calls the gradient at every leapfrog step.

	Degenerate guard: when N == 0 or D == 0 the ratio Ng/N or Dg/D is
	undefined. We set grad to (0, 0) and let the caller treat the
	density as -inf (the MH step will reject). Returning non-zero
	garbage here would corrupt the leapfrog integrator.

	Consistency: verified against central finite differences on
	ndg_log_density in test_prior.c (Test 4) and against a pure-R
	re-implementation in test_prior.R (Test C). Both pass at double
	precision.
*/
static void ndg_log_gradient(PRIOR *prior, double *theta, double *grad_out)
{
	/* Default output so error paths leave valid state. */
	grad_out[0] = 0.0;
	grad_out[1] = 0.0;

	if(prior->ndg_m   <= 0)   return;
	if(prior->ndg_p   == 0.0) return;

	double d0 = theta[0] - prior->ndg_tref[0];
	double d1 = theta[1] - prior->ndg_tref[1];

	/* Pass 1: z_max over the whole pool (same reason as in the density). */
	double z_max = NEG_INF;
	int i;
	for(i = 0; i < prior->ndg_m; ++i)
	{
		double z = d0 * (double)prior->ndg_g1[i]
		         + d1 * (double)prior->ndg_g2[i];
		if(z > z_max) z_max = z;
	}

	/* Pass 2: accumulate the five shifted moments we need.
	       N    = sum_{interior}  w_i
	       D    = sum_{all}       w_i
	       N_g0 = sum_{interior}  w_i g_{i,0}
	       N_g1 = sum_{interior}  w_i g_{i,1}
	       D_g0 = sum_{all}       w_i g_{i,0}
	       D_g1 = sum_{all}       w_i g_{i,1}
	   The z_max shift cancels in every ratio N/D, Ng/N, Dg/D below. */
	double N = 0.0, D = 0.0;
	double N_g0 = 0.0, N_g1 = 0.0;
	double D_g0 = 0.0, D_g1 = 0.0;

	for(i = 0; i < prior->ndg_m; ++i)
	{
		double g0 = (double)prior->ndg_g1[i];
		double g1 = (double)prior->ndg_g2[i];
		double z  = d0 * g0 + d1 * g1;
		double w  = exp(z - z_max);

		D    += w;
		D_g0 += w * g0;
		D_g1 += w * g1;

		if(prior->ndg_in_hull[i])
		{
			N    += w;
			N_g0 += w * g0;
			N_g1 += w * g1;
		}
	}

	/* If the estimator is degenerate, keep grad = (0, 0) so the caller
	   doesn't propagate NaN or inf into the leapfrog momentum update. */
	if(N <= 0.0 || D <= 0.0) return;

	/* Final assembly: scale by the strength exponent p. */
	grad_out[0] = prior->ndg_p * (N_g0 / N - D_g0 / D);
	grad_out[1] = prior->ndg_p * (N_g1 / N - D_g1 / D);
}

/* ================================================================== */
/*  Constructors / destructor                                         */
/* ================================================================== */

PRIOR *prior_create_normal(double mean0, double mean1, double var0, double var1)
{
	PRIOR *p = calloc(1, sizeof(PRIOR));
	if(!p) return NULL;
	p->type = PRIOR_NORMAL;
	p->normal_mean[0] = mean0;
	p->normal_mean[1] = mean1;
	p->normal_var[0]  = var0;
	p->normal_var[1]  = var1;
	return p;
}

PRIOR *prior_create_uniform(double lo0, double hi0, double lo1, double hi1)
{
	PRIOR *p = calloc(1, sizeof(PRIOR));
	if(!p) return NULL;
	p->type = PRIOR_UNIFORM;
	p->uniform_lower[0] = lo0;
	p->uniform_upper[0] = hi0;
	p->uniform_lower[1] = lo1;
	p->uniform_upper[1] = hi1;
	return p;
}

/*
	prior_create_ndg: build an empty NDG prior.

	Initial state:
	  - Bounding box set (s1_max, s2_max) so the hull can size its
	    bitmaps. For an n-node undirected ERGM with edges + kstar(2):
	        s1_max = n*(n-1)/2      (max # edges)
	        s2_max = n*(n-1)*(n-2)/2 (max # 2-stars)
	  - p_strength stored so every subsequent density/gradient call
	    knows the sharpness. Typical choice: 1.0.
	  - Pool buffers allocated with small initial capacity (16), will
	    grow on first prior_ndg_set_pool if m > 16.
	  - Hull is empty (n_observed == 0). Must be populated either by
	    prior_ndg_warm_start (pre-drawn samples from a file) or by
	    repeated prior_ndg_set_pool calls during HMC burn-in.

	The NDG prior is useless without some hull coverage: at an empty
	hull, in_hull[] is all zero and Phat == 0 for every theta. The
	warm-start file exists precisely to give a thick initial hull so
	early HMC iterations don't get stuck at log Phat = -inf.
*/
PRIOR *prior_create_ndg(int s1_max, int s2_max, double p_strength)
{
	PRIOR *p = calloc(1, sizeof(PRIOR));
	if(!p) return NULL;

	p->type  = PRIOR_NDG;
	p->ndg_p = p_strength;

	p->ndg_hull = hull_create(s1_max, s2_max);
	if(!p->ndg_hull)
	{
		free(p);
		return NULL;
	}

	/* Initial pool capacity is 16; prior_ndg_set_pool grows as needed. */
	p->ndg_capacity = 16;
	p->ndg_g1      = malloc((size_t)p->ndg_capacity * sizeof(int));
	p->ndg_g2      = malloc((size_t)p->ndg_capacity * sizeof(int));
	p->ndg_in_hull = malloc((size_t)p->ndg_capacity * sizeof(int));

	if(!p->ndg_g1 || !p->ndg_g2 || !p->ndg_in_hull)
	{
		prior_destroy(p);
		return NULL;
	}

	p->ndg_m       = 0;
	p->ndg_tref[0] = 0.0;
	p->ndg_tref[1] = 0.0;

	fprintf(stdout,
		"Created NDG prior: bbox=[0,%d]x[0,%d], p_strength=%.6f\n",
		s1_max, s2_max, p_strength);

	return p;
}

void prior_destroy(PRIOR *prior)
{
	if(!prior) return;
	if(prior->type == PRIOR_NDG)
	{
		hull_destroy(prior->ndg_hull);
		free(prior->ndg_g1);
		free(prior->ndg_g2);
		free(prior->ndg_in_hull);
	}
	free(prior);
}

/* ================================================================== */
/*  NDG: warm start                                                   */
/* ================================================================== */

/*
	prior_ndg_warm_start: seed the hull from a file of pre-drawn samples.

	Purpose
	-------
	Before HMC starts, we want the hull to already be a reasonable
	approximation of the achievable-stats region, so that early
	iterations don't suffer from Phat == 0 (log Phat == -inf) at every
	plausible theta. Running a one-off simulation at the MPLE and
	dumping a few times 10^5 auxiliary statistics to disk gives us a
	thick initial hull at zero marginal runtime cost.

	File format (see 07_GenerateNDGPrior.R):
	    line 1     : N                                (unsigned integer)
	    lines 2..  : "<s1> <s2>\n"                    (integer pair per row)

	Only the (s1, s2) columns are used. Any extra fields a caller's
	file might contain (e.g. a pre-computed hull indicator or reference
	theta) are IGNORED by design: the warm start must carry no
	ground-truth information. This contract is exactly why the parser
	uses fscanf("%lf %lf", ...) rather than fscanf("%lf %lf %lf %lf %lf", ...).

	Error handling
	--------------
	  - Missing / unreadable file: logged and returns 1. Caller decides
	    whether to abort or continue with an empty hull.
	  - Header row missing or non-numeric: logged and returns 1.
	  - A row with non-integer (s1, s2): skipped with a bump of
	    n_skipped; we keep going because a malformed row is more likely
	    a single typo than a corrupt file.
	  - Points out of the hull's bounding box: silently ignored by
	    hull_add, not counted as skipped. This is OK because they can't
	    be interior of any hull inside the box.

	Returns 0 on success (including "file parsed but had 0 rows"),
	nonzero on hard failure.
*/
int prior_ndg_warm_start(PRIOR *prior, const char *filename)
{
	if(!prior) return 1;
	if(prior->type != PRIOR_NDG) return 0;  /* no-op for Normal/Uniform */
	if(!filename) return 1;

	FILE *f = fopen(filename, "r");
	if(!f)
	{
		fprintf(stderr, "ERROR: cannot open NDG warm-start file: %s\n", filename);
		return 1;
	}

	size_t nrows = 0;
	if(fscanf(f, "%zu", &nrows) != 1)
	{
		fprintf(stderr, "ERROR: invalid NDG warm-start header in %s\n", filename);
		fclose(f);
		return 1;
	}
	if(nrows == 0)
	{
		fprintf(stderr, "WARNING: NDG warm-start %s has 0 rows\n", filename);
		fclose(f);
		return 0;
	}

	size_t i;
	size_t n_added = 0;
	size_t n_skipped = 0;
	for(i = 0; i < nrows; ++i)
	{
		/* Read two whitespace-separated doubles per row; coerce to int
		   below. Using %lf rather than %d lets us tolerate files that
		   were written with trailing ".0" (e.g. R's default for
		   integer-valued floats) without a format mismatch. */
		double s1_d, s2_d;
		int ret = fscanf(f, "%lf %lf", &s1_d, &s2_d);
		if(ret != 2)
		{
			fprintf(stderr,
				"ERROR: NDG warm-start parse error at row %zu of %s "
				"(expected 2 fields, got %d)\n",
				i, filename, ret);
			fclose(f);
			return 1;
		}

		/* Require exact integer values. Skipping (rather than rejecting
		   the whole file) is the tolerant default: one bad row
		   shouldn't invalidate a 10^5-row warm start. */
		int s1 = (int)s1_d;
		int s2 = (int)s2_d;
		if((double)s1 != s1_d || (double)s2 != s2_d)
		{
			n_skipped++;
			continue;
		}

		/* hull_add is idempotent, so duplicate (s1, s2) rows are fine
		   and just bump n_added without actually growing the hull. */
		hull_add(prior->ndg_hull, s1, s2);
		n_added++;
	}

	fclose(f);

	/* The hull_n_vertices call triggers one rebuild, so the diagnostic
	   line below reports the true post-warm-start vertex count. */
	fprintf(stdout,
		"NDG warm start: read %zu samples from %s (%zu skipped)\n"
		"  hull after warm start: %zu unique points, %d vertices\n",
		n_added, filename, n_skipped,
		hull_n_observed(prior->ndg_hull),
		hull_n_vertices(prior->ndg_hull));

	return 0;
}

/* ================================================================== */
/*  NDG: pool refresh                                                 */
/* ================================================================== */

/*
	prior_ndg_set_pool: replace the IS pool with m new samples.

	Called by hmc.c at every leapfrog step that needs a fresh
	minimum-variance estimate of log p or its gradient. The semantics:

	  1. Copy the m (s1_i, s2_i) pairs into ndg_g1 / ndg_g2.
	  2. Add every sample to the hull via hull_add. This may grow the
	     hull (new points push out the polygon) and flips the dirty
	     bit so the first interior query below triggers a rebuild.
	  3. Recompute in_hull[i] for every i AGAINST THE FINAL HULL STATE.
	     This ordering matters: we don't want the first few samples to
	     be classified against a stale hull that hasn't seen the later
	     samples yet.
	  4. Record theta_tilde as the reference parameter for subsequent
	     weight computations until the next prior_ndg_set_pool.

	Why classify against the post-add hull?
	    Because the hull IS the set of observed points (plus warm
	    start). Any sample that was just added IS in seen[] by the
	    time we classify, so a sample that happens to fall inside the
	    hull will correctly be classified as interior even if it was
	    the one that extended the hull.

	Post-condition: ndg_m == m (or truncated to ndg_capacity if realloc
	failed), ndg_tref == theta_tilde, and in_hull[0..m-1] correctly
	reflects the current hull state.

	Cost: O(m) for the copy + m hull_add calls (each O(1)) + one hull
	rebuild (O(|seen|)) on the first interior query. The rebuild cost
	is amortised because interior[] is then valid until the next
	hull_add batch.

	No-op for non-NDG priors -- safe to call unconditionally from HMC
	so the control flow stays uniform across prior types.
*/
void prior_ndg_set_pool(PRIOR        *prior,
                        const int    *s1_samples,
                        const int    *s2_samples,
                        int           m,
                        const double *theta_tilde)
{
	if(!prior || prior->type != PRIOR_NDG) return;
	if(!s1_samples || !s2_samples || !theta_tilde) return;
	if(m <= 0) { prior->ndg_m = 0; return; }

	/* ---- Grow pool buffers if the new batch exceeds our capacity. ---- */
	/* Capacity doubling matches the amortised O(1) growth pattern used
	   elsewhere in this codebase. If any realloc fails we keep the
	   old buffers (whichever reallocs did succeed are reassigned) and
	   truncate the batch to what we can safely hold. */
	if(m > prior->ndg_capacity)
	{
		int newcap = prior->ndg_capacity > 0 ? prior->ndg_capacity : 16;
		while(newcap < m) newcap *= 2;

		int *new_g1 = realloc(prior->ndg_g1,      (size_t)newcap * sizeof(int));
		int *new_g2 = realloc(prior->ndg_g2,      (size_t)newcap * sizeof(int));
		int *new_ih = realloc(prior->ndg_in_hull, (size_t)newcap * sizeof(int));

		if(new_g1) prior->ndg_g1      = new_g1;
		if(new_g2) prior->ndg_g2      = new_g2;
		if(new_ih) prior->ndg_in_hull = new_ih;
		if(new_g1 && new_g2 && new_ih) prior->ndg_capacity = newcap;

		/* If any realloc failed, cap m at our real capacity to avoid
		   writing past the end of any of the three parallel arrays. */
		if(m > prior->ndg_capacity) m = prior->ndg_capacity;
	}

	/* ---- Copy samples into the pool and register them with the hull. ---- */
	/* Every sample contributes to the hull's monotone growth. Samples
	   that were already in seen[] are no-ops; new ones flip the dirty
	   bit. */
	int i;
	for(i = 0; i < m; ++i)
	{
		prior->ndg_g1[i] = s1_samples[i];
		prior->ndg_g2[i] = s2_samples[i];
		hull_add(prior->ndg_hull, s1_samples[i], s2_samples[i]);
	}

	/* ---- Classify every sample against the FINAL hull state. ---- */
	/* The first hull_is_interior call on a dirty hull triggers a
	   rebuild; subsequent calls in this loop hit the cached
	   interior[] bitmap in O(1) each. */
	for(i = 0; i < m; ++i)
	{
		prior->ndg_in_hull[i] =
			hull_is_interior(prior->ndg_hull, prior->ndg_g1[i], prior->ndg_g2[i]);
	}

	/* ---- Record the reference parameter and pool size. ---- */
	prior->ndg_tref[0] = theta_tilde[0];
	prior->ndg_tref[1] = theta_tilde[1];
	prior->ndg_m       = m;
}

/* ================================================================== */
/*  Log density                                                       */
/* ================================================================== */

/*
	prior_log_density: dispatch log p(theta) to the right estimator.

	For Normal/Uniform: closed-form evaluation, no MCMC state required.
	For NDG: delegates to ndg_log_density, which uses the CURRENT pool
	and theta_tilde (set by the most recent prior_ndg_set_pool).

	Callers are responsible for setting the pool before querying the
	NDG density. hmc.c does this at every leapfrog endpoint and caches
	the result in log_p_cur_cache / log_p_new_cache for the eventual
	accept/reject decision.
*/
double prior_log_density(PRIOR *prior, double *theta)
{
	if(!prior) return 0.0;

	switch(prior->type)
	{
		case PRIOR_NORMAL:
		{
			/* log N(theta | mean, diag(var))  (up to a constant that
			   cancels in every ratio we care about, so we drop it). */
			double lp = 0.0;
			int d;
			for(d = 0; d < 2; ++d)
			{
				double z = theta[d] - prior->normal_mean[d];
				lp -= 0.5 * z * z / prior->normal_var[d];
			}
			return lp;
		}

		case PRIOR_UNIFORM:
		{
			/* Indicator of the box: -inf outside, 0 (= log 1) inside,
			   dropping the constant -log(area). */
			if(theta[0] < prior->uniform_lower[0] ||
			   theta[0] > prior->uniform_upper[0] ||
			   theta[1] < prior->uniform_lower[1] ||
			   theta[1] > prior->uniform_upper[1])
				return NEG_INF;
			return 0.0;
		}

		case PRIOR_NDG:
			return ndg_log_density(prior, theta);

		default:
			return 0.0;
	}
}

/* ================================================================== */
/*  Log prior ratio: log p(theta_new) - log p(theta_cur)              */
/* ================================================================== */

/*
	prior_log_ratio: compute log p(theta_new) - log p(theta_cur).

	Used inside the MH accept/reject step. The legacy (Normal/Uniform)
	implementations compute the difference analytically so we don't
	accumulate floating-point noise; the NDG branch is a convenience
	wrapper for callers that don't manage their own caching.

	NOTE for NDG: hmc.c prefers to bypass this helper. It calls
	prior_log_density TWICE -- once right after refreshing the pool at
	theta_cur (minimum-variance estimate of log p_cur), once right
	after refreshing the pool at theta_new (minimum-variance estimate
	of log p_new) -- and subtracts. Using prior_log_ratio here would
	evaluate BOTH densities against the SAME (most recent) pool, which
	is a higher-variance IS estimate because one of the two thetas is
	off-reference.
*/
double prior_log_ratio(PRIOR *prior, double *theta_cur, double *theta_new)
{
	if(!prior) return 0.0;

	switch(prior->type)
	{
		case PRIOR_NORMAL:
		{
			/* Analytic difference: 0.5 * [ (t_cur - mu)^2 - (t_new - mu)^2 ] / var.
			   Equivalent to prior_log_density(new) - prior_log_density(cur)
			   up to numerical noise; we write it this way to match the
			   legacy exchange driver's arithmetic bit-for-bit. */
			double ratio = 0.0;
			int d;
			for(d = 0; d < 2; ++d)
			{
				double z  = theta_cur[d] - prior->normal_mean[d];
				double zp = theta_new[d] - prior->normal_mean[d];
				ratio += 0.5 * (z*z - zp*zp) / prior->normal_var[d];
			}
			return ratio;
		}

		case PRIOR_UNIFORM:
		{
			/* Only theta_new is checked: theta_cur is assumed to be a
			   previously accepted state, hence inside the box. If
			   theta_new is outside, the ratio is -inf (guaranteed
			   reject); otherwise 0. This preserves bit-for-bit parity
			   with the legacy behaviour. */
			int new_inside =
				(theta_new[0] >= prior->uniform_lower[0]) &&
				(theta_new[0] <= prior->uniform_upper[0]) &&
				(theta_new[1] >= prior->uniform_lower[1]) &&
				(theta_new[1] <= prior->uniform_upper[1]);
			if(!new_inside) return NEG_INF;
			return 0.0;
		}

		case PRIOR_NDG:
		{
			/* Fallback path: evaluate both log densities against
			   whatever pool is currently set. This will usually be
			   the pool at theta_new (because hmc.c refreshed there
			   last), so the log p_cur estimate here has off-reference
			   IS weights and is noisier than what hmc.c achieves with
			   its own caching. */
			double lp_new = prior_log_density(prior, theta_new);
			if(lp_new == NEG_INF) return NEG_INF;
			double lp_cur = prior_log_density(prior, theta_cur);
			if(lp_cur == NEG_INF) return POS_INF;  /* pathological: guaranteed accept */
			return lp_new - lp_cur;
		}

		default:
			return 0.0;
	}
}

/* ================================================================== */
/*  Gradient of log prior                                             */
/* ================================================================== */

/*
	prior_log_gradient: write the gradient of log p(theta) into grad_out.

	  Normal : -(theta - mean) / var, per coordinate.
	  Uniform: 0 everywhere (flat log density inside the box; the
	           boundary is a measure-zero set that HMC will avoid under
	           any sensible stepsize).
	  NDG    : delegated to ndg_log_gradient; uses the current pool
	           and theta_tilde.

	Gradient correctness has been verified three ways:
	  - C-side against central finite differences (test_prior.c Test 4).
	  - R-side against a pure-R reimplementation of the estimator
	    (test_prior.R Test C, analytical).
	  - R-side against numDeriv::grad on the pure-R log_p (Test C, FD).
*/
void prior_log_gradient(PRIOR *prior, double *theta, double *grad_out)
{
	if(!prior) { grad_out[0] = 0.0; grad_out[1] = 0.0; return; }

	switch(prior->type)
	{
		case PRIOR_NORMAL:
		{
			/* d/d theta_d [ -0.5 * (theta_d - mu_d)^2 / var_d ]
			     = (mu_d - theta_d) / var_d
			   (Equivalently -(theta - mu)/var, up to sign choice.) */
			int d;
			for(d = 0; d < 2; ++d)
				grad_out[d] = (prior->normal_mean[d] - theta[d]) / prior->normal_var[d];
			break;
		}

		case PRIOR_UNIFORM:
		{
			/* Flat inside the box, so gradient is zero there. We
			   don't special-case the boundary; HMC simulating
			   continuous dynamics with a finite stepsize will never
			   hit a measure-zero set exactly. */
			grad_out[0] = 0.0;
			grad_out[1] = 0.0;
			break;
		}

		case PRIOR_NDG:
			ndg_log_gradient(prior, theta, grad_out);
			break;

		default:
			grad_out[0] = 0.0;
			grad_out[1] = 0.0;
			break;
	}
}
