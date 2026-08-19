/*
	prior.c
	=======

	Prior distributions for ERGM parameters: Normal, Uniform, and the
	adaptive Non-Degeneracy (NDG) prior.

	Overview
	--------
	  PRIOR_NORMAL :  N(mean, diag(var)) -- closed form, no MCMC state.
	  PRIOR_UNIFORM:  Uniform on the box [lower_i, upper_i] per dim.
	  PRIOR_NDG    :  Adaptive, data-driven prior that penalises theta
	                  values under which the ERGM produces sufficient
	                  statistics near the boundary of their achievable
	                  region.

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
	gives, for each coordinate d in {0, 1, 2}:

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
	exp(z_i - z_max) which is in (0, 1]. Because z_max is SUBTRACTED
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
	It operates purely on the sufficient statistics handed to it via
	prior_ndg_set_pool or prior_ndg_warm_start.
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
	    NEG_INF as "proposal rejected", which is the correct behaviour:
	    the IS estimate says this theta has zero non-degeneracy
	    probability under the current pool.

	p_strength == 0 short-circuits to log density = 0 (a flat improper
	prior): no need to touch the pool at all.

	Two-pass structure
	------------------
	Pass 1 computes z_max over ALL samples. Pass 2 accumulates weights.
	We need two passes because the shift must be known before the
	individual w_i = exp(z_i - z_max) can be formed.

	Cost: O(m) per call for a 3D pool (3 multiply-adds per sample in
	each pass, plus one exp() call in pass 2). For the typical Kapferer
	pool (m = grad_ndraws = 10) that's 60 mul-adds plus 10 exp() calls
	per evaluation.
*/
static double ndg_log_density(PRIOR *prior, double *theta)
{
	if(prior->ndg_m <= 0) return NEG_INF;       /* no pool -> estimate undefined */
	if(prior->ndg_p == 0.0) return 0.0;         /* strength 0 -> flat prior */

	double d0 = theta[0] - prior->ndg_tref[0];
	double d1 = theta[1] - prior->ndg_tref[1];
	double d2 = theta[2] - prior->ndg_tref[2];

	/* ---- Pass 1: find the shift z_max over the whole pool. ---- */
	/* Must be over ALL samples (interior and exterior) so the shift
	   cancels exactly in both sums in pass 2. A shift computed only
	   over interior points would bias the denominator. */
	double z_max = NEG_INF;
	int i;
	for(i = 0; i < prior->ndg_m; ++i)
	{
		double z = d0 * prior->ndg_g1[i]
		         + d1 * prior->ndg_g2[i]
		         + d2 * prior->ndg_g3[i];
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
		double z = d0 * prior->ndg_g1[i]
		         + d1 * prior->ndg_g2[i]
		         + d2 * prior->ndg_g3[i];
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
	All eight running sums (N, D, three components of Ng, three
	components of Dg) can be accumulated in a single pass over the pool
	after we know z_max. That makes the gradient cost only marginally
	more than the density cost (6 more mul-adds per sample), which
	matters because hmc.c calls the gradient at every leapfrog step.

	Degenerate guard: when N == 0 or D == 0 the ratio Ng/N or Dg/D is
	undefined. We set grad to (0, 0, 0) and let the caller treat the
	density as -inf (the MH step will reject). Returning non-zero
	garbage here would corrupt the leapfrog integrator.

	Consistency: verified against central finite differences on
	ndg_log_density in test_prior.c (Test 4) and against a pure-R
	re-implementation in test_prior.R (Test C). Both pass at double
	precision.
*/
static void ndg_log_gradient(PRIOR *prior, double *theta, double *grad_out)
{
	/* Default output so error paths leave valid state. Any early
	   return below (empty pool, p_strength == 0, degenerate N/D) then
	   produces a zero gradient -- which is the right behaviour for HMC:
	   zero gradient means the NDG prior contributes no force to the
	   leapfrog momentum update at this theta, so the likelihood and
	   other priors still drive the dynamics correctly. */
	grad_out[0] = 0.0;
	grad_out[1] = 0.0;
	grad_out[2] = 0.0;

	if(prior->ndg_m   <= 0)   return;
	if(prior->ndg_p   == 0.0) return;

	double d0 = theta[0] - prior->ndg_tref[0];
	double d1 = theta[1] - prior->ndg_tref[1];
	double d2 = theta[2] - prior->ndg_tref[2];

	/* Pass 1: z_max over the whole pool (same reason as in the density). */
	double z_max = NEG_INF;
	int i;
	for(i = 0; i < prior->ndg_m; ++i)
	{
		double z = d0 * prior->ndg_g1[i]
		         + d1 * prior->ndg_g2[i]
		         + d2 * prior->ndg_g3[i];
		if(z > z_max) z_max = z;
	}

	/* Pass 2: accumulate the eight shifted moments we need.
	       N    = sum_{interior}  w_i
	       D    = sum_{all}       w_i
	       N_g0 = sum_{interior}  w_i g_{i,0}
	       N_g1 = sum_{interior}  w_i g_{i,1}
	       N_g2 = sum_{interior}  w_i g_{i,2}
	       D_g0 = sum_{all}       w_i g_{i,0}
	       D_g1 = sum_{all}       w_i g_{i,1}
	       D_g2 = sum_{all}       w_i g_{i,2}
	   The z_max shift cancels in every ratio N/D, Ng/N, Dg/D below. */
	double N = 0.0, D = 0.0;
	double N_g0 = 0.0, N_g1 = 0.0, N_g2 = 0.0;
	double D_g0 = 0.0, D_g1 = 0.0, D_g2 = 0.0;

	for(i = 0; i < prior->ndg_m; ++i)
	{
		double g0 = prior->ndg_g1[i];
		double g1 = prior->ndg_g2[i];
		double g2 = prior->ndg_g3[i];
		double z  = d0 * g0 + d1 * g1 + d2 * g2;
		double w  = exp(z - z_max);

		D    += w;
		D_g0 += w * g0;
		D_g1 += w * g1;
		D_g2 += w * g2;

		if(prior->ndg_in_hull[i])
		{
			N    += w;
			N_g0 += w * g0;
			N_g1 += w * g1;
			N_g2 += w * g2;
		}
	}

	/* If the estimator is degenerate, keep grad = 0 so the caller
	   doesn't propagate NaN or inf into the leapfrog momentum update. */
	if(N <= 0.0 || D <= 0.0) return;

	/* Final assembly: scale by the strength exponent p. */
	grad_out[0] = prior->ndg_p * (N_g0 / N - D_g0 / D);
	grad_out[1] = prior->ndg_p * (N_g1 / N - D_g1 / D);
	grad_out[2] = prior->ndg_p * (N_g2 / N - D_g2 / D);
}

/* ================================================================== */
/*  Constructors / destructor                                         */
/* ================================================================== */

PRIOR *prior_create_normal(int d, const double *mean, const double *var)
{
	if(d <= 0 || d > MAXSTATS) return NULL;
	if(!mean || !var) return NULL;
	PRIOR *p = calloc(1, sizeof(PRIOR));
	if(!p) return NULL;
	p->type = PRIOR_NORMAL;
	p->d    = d;
	int i;
	for(i = 0; i < d; ++i)
	{
		p->normal_mean[i] = mean[i];
		p->normal_var[i]  = var[i];
	}
	return p;
}

PRIOR *prior_create_uniform(int d, const double *lower, const double *upper)
{
	if(d <= 0 || d > MAXSTATS) return NULL;
	if(!lower || !upper) return NULL;
	PRIOR *p = calloc(1, sizeof(PRIOR));
	if(!p) return NULL;
	p->type = PRIOR_UNIFORM;
	p->d    = d;
	int i;
	for(i = 0; i < d; ++i)
	{
		p->uniform_lower[i] = lower[i];
		p->uniform_upper[i] = upper[i];
	}
	return p;
}

/*
	prior_create_ndg: build an empty 3D NDG prior.

	Initial state:
	  - A 3D hull is created with the supplied tolerance; it starts
	    empty (no observations, no seed tetrahedron).
	  - p_strength stored so every subsequent density/gradient call
	    knows the sharpness. Typical choice: 1.0.
	  - Pool buffers allocated with small initial capacity (16); they
	    grow on first prior_ndg_set_pool if m > 16.
	  - Hull is empty (n_observed == 0). Must be populated either by
	    prior_ndg_warm_start (pre-drawn samples from a file) or by
	    repeated prior_ndg_set_pool calls during HMC burn-in.

	The NDG prior is useless without some hull coverage: at an empty
	hull, in_hull[] is all zero and Phat == 0 for every theta. The
	warm-start file exists precisely to give a thick initial hull so
	early HMC iterations don't get stuck at log Phat = -inf.
*/
PRIOR *prior_create_ndg(double hull_eps, double p_strength)
{
	PRIOR *p = calloc(1, sizeof(PRIOR));
	if(!p) return NULL;

	p->type  = PRIOR_NDG;
	p->d     = 3;
	p->ndg_p = p_strength;

	p->ndg_hull = hull3d_create(hull_eps);
	if(!p->ndg_hull)
	{
		free(p);
		return NULL;
	}

	/* Initial pool capacity is 16; prior_ndg_set_pool grows as needed. */
	p->ndg_capacity = 16;
	p->ndg_g1      = malloc((size_t)p->ndg_capacity * sizeof(double));
	p->ndg_g2      = malloc((size_t)p->ndg_capacity * sizeof(double));
	p->ndg_g3      = malloc((size_t)p->ndg_capacity * sizeof(double));
	p->ndg_in_hull = malloc((size_t)p->ndg_capacity * sizeof(int));

	if(!p->ndg_g1 || !p->ndg_g2 || !p->ndg_g3 || !p->ndg_in_hull)
	{
		prior_destroy(p);
		return NULL;
	}

	p->ndg_m       = 0;
	p->ndg_tref[0] = 0.0;
	p->ndg_tref[1] = 0.0;
	p->ndg_tref[2] = 0.0;

	fprintf(stdout,
		"Created NDG prior (3D): hull_eps=%.3e, p_strength=%.6f\n",
		hull_eps > 0 ? hull_eps : 1e-9, p_strength);

	return p;
}

void prior_destroy(PRIOR *prior)
{
	if(!prior) return;
	if(prior->type == PRIOR_NDG)
	{
		hull3d_destroy(prior->ndg_hull);
		free(prior->ndg_g1);
		free(prior->ndg_g2);
		free(prior->ndg_g3);
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
	plausible theta. Running a one-off simulation at the MAP and
	dumping a few 10^4 auxiliary statistics to disk gives us a thick
	initial hull at zero marginal runtime cost.

	File format (see 07_GenerateNDGPrior.R):
	    line 1     : N                                (unsigned integer)
	    lines 2..  : "<s1> <s2> <s3>\n"               (three doubles per row)

	Only the (s1, s2, s3) columns are used. Any extra fields a caller's
	file might contain (e.g. a pre-computed hull indicator or reference
	theta) are IGNORED by design: the warm start must carry no
	ground-truth information.

	Error handling
	--------------
	  - Missing / unreadable file: logged and returns 1. Caller decides
	    whether to abort or continue with an empty hull.
	  - Header row missing or non-numeric: logged and returns 1.
	  - Row with fewer than 3 fields: logged and returns 1 (we can't
	    skip because subsequent rows would desync; better to abort).

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
	for(i = 0; i < nrows; ++i)
	{
		double s1, s2, s3;
		int ret = fscanf(f, "%lf %lf %lf", &s1, &s2, &s3);
		if(ret != 3)
		{
			fprintf(stderr,
				"ERROR: NDG warm-start parse error at row %zu of %s "
				"(expected 3 fields, got %d)\n",
				i, filename, ret);
			fclose(f);
			return 1;
		}
		hull3d_add(prior->ndg_hull, s1, s2, s3);
		n_added++;
	}

	fclose(f);

	fprintf(stdout,
		"NDG warm start: read %zu samples from %s\n"
		"  hull after warm start: %zu observations, %d vertices, %d faces\n",
		n_added, filename,
		hull3d_n_observed(prior->ndg_hull),
		hull3d_n_vertices(prior->ndg_hull),
		hull3d_n_faces(prior->ndg_hull));

	return 0;
}

/* ================================================================== */
/*  NDG: pool refresh                                                 */
/* ================================================================== */

/*
	prior_ndg_set_pool: replace the IS pool with m new samples.

	Called by hmc.c at every leapfrog step that needs a fresh
	minimum-variance estimate of log p or its gradient. The semantics:

	  1. Copy the m (s1_i, s2_i, s3_i) triples into ndg_g1 / ndg_g2 / ndg_g3.
	  2. Add every sample to the hull via hull3d_add. This may extend
	     the hull (new points push out the polytope).
	  3. Recompute in_hull[i] for every i AGAINST THE FINAL HULL STATE.
	     This ordering matters: we don't want the first few samples to
	     be classified against a stale hull that hasn't seen the later
	     samples yet.
	  4. Record theta_tilde as the reference parameter for subsequent
	     weight computations until the next prior_ndg_set_pool.

	Why classify against the post-add hull?
	    Because the hull IS the set of observed points (plus warm
	    start). Any sample that was just added IS part of the polytope
	    by the time we classify, so a sample that happens to fall
	    inside the hull will correctly be classified as interior even
	    if it was the one that just extended the hull.

	Post-condition: ndg_m == m (or truncated to ndg_capacity if realloc
	failed), ndg_tref == theta_tilde, and in_hull[0..m-1] correctly
	reflects the current hull state.

	Cost: O(m) for the copy + m incremental hull3d_add operations
	(each amortised O(polylog n)) + m interior queries (each O(F)
	linear in the number of faces). For the typical Kapferer pool
	(m = 10, F ~ 50 vertices) that's a few hundred mul-adds.

	No-op for non-NDG priors -- safe to call unconditionally from HMC
	so the control flow stays uniform across prior types.
*/
void prior_ndg_set_pool(PRIOR        *prior,
                        const double *s1_samples,
                        const double *s2_samples,
                        const double *s3_samples,
                        int           m,
                        const double *theta_tilde)
{
	if(!prior || prior->type != PRIOR_NDG) return;
	if(!s1_samples || !s2_samples || !s3_samples || !theta_tilde) return;
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

		double *new_g1 = realloc(prior->ndg_g1,      (size_t)newcap * sizeof(double));
		double *new_g2 = realloc(prior->ndg_g2,      (size_t)newcap * sizeof(double));
		double *new_g3 = realloc(prior->ndg_g3,      (size_t)newcap * sizeof(double));
		int    *new_ih = realloc(prior->ndg_in_hull, (size_t)newcap * sizeof(int));

		if(new_g1) prior->ndg_g1      = new_g1;
		if(new_g2) prior->ndg_g2      = new_g2;
		if(new_g3) prior->ndg_g3      = new_g3;
		if(new_ih) prior->ndg_in_hull = new_ih;
		if(new_g1 && new_g2 && new_g3 && new_ih) prior->ndg_capacity = newcap;

		/* If any realloc failed, cap m at our real capacity to avoid
		   writing past the end of any of the four parallel arrays. */
		if(m > prior->ndg_capacity) m = prior->ndg_capacity;
	}

	/* ---- Copy samples into the pool and register them with the hull. ---- */
	int i;
	for(i = 0; i < m; ++i)
	{
		prior->ndg_g1[i] = s1_samples[i];
		prior->ndg_g2[i] = s2_samples[i];
		prior->ndg_g3[i] = s3_samples[i];
		hull3d_add(prior->ndg_hull, s1_samples[i], s2_samples[i], s3_samples[i]);
	}

	/* ---- Classify every sample against the final hull state. ---- */
	for(i = 0; i < m; ++i)
	{
		prior->ndg_in_hull[i] =
			hull3d_is_interior(prior->ndg_hull,
				prior->ndg_g1[i], prior->ndg_g2[i], prior->ndg_g3[i]);
	}

	/* ---- Record the reference parameter and pool size. ---- */
	prior->ndg_tref[0] = theta_tilde[0];
	prior->ndg_tref[1] = theta_tilde[1];
	prior->ndg_tref[2] = theta_tilde[2];
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
			int k;
			for(k = 0; k < prior->d; ++k)
			{
				double z = theta[k] - prior->normal_mean[k];
				lp -= 0.5 * z * z / prior->normal_var[k];
			}
			return lp;
		}

		case PRIOR_UNIFORM:
		{
			/* Indicator of the box: -inf outside, 0 (= log 1) inside,
			   dropping the constant -log(vol). */
			int k;
			for(k = 0; k < prior->d; ++k)
			{
				if(theta[k] < prior->uniform_lower[k] ||
				   theta[k] > prior->uniform_upper[k])
					return NEG_INF;
			}
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
			int k;
			for(k = 0; k < prior->d; ++k)
			{
				double z  = theta_cur[k] - prior->normal_mean[k];
				double zp = theta_new[k] - prior->normal_mean[k];
				ratio += 0.5 * (z*z - zp*zp) / prior->normal_var[k];
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
			int k;
			for(k = 0; k < prior->d; ++k)
			{
				if(theta_new[k] < prior->uniform_lower[k] ||
				   theta_new[k] > prior->uniform_upper[k])
					return NEG_INF;
			}
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
	int k;
	if(!prior) { for(k = 0; k < MAXSTATS; ++k) grad_out[k] = 0.0; return; }

	switch(prior->type)
	{
		case PRIOR_NORMAL:
		{
			/* d/d theta_k [ -0.5 * (theta_k - mu_k)^2 / var_k ]
			     = (mu_k - theta_k) / var_k
			   (Equivalently -(theta - mu)/var, up to sign choice.) */
			for(k = 0; k < prior->d; ++k)
				grad_out[k] = (prior->normal_mean[k] - theta[k]) / prior->normal_var[k];
			break;
		}

		case PRIOR_UNIFORM:
		{
			/* Flat inside the box, so gradient is zero there. We
			   don't special-case the boundary; HMC simulating
			   continuous dynamics with a finite stepsize will never
			   hit a measure-zero set exactly. */
			for(k = 0; k < prior->d; ++k) grad_out[k] = 0.0;
			break;
		}

		case PRIOR_NDG:
			ndg_log_gradient(prior, theta, grad_out);
			break;

		default:
			for(k = 0; k < prior->d; ++k) grad_out[k] = 0.0;
			break;
	}
}
