#ifndef PRIOR_H
#define PRIOR_H

#include <stdlib.h>
#include "hull3d.h"
#include "graph_2410.h"   /* MAXSTATS */

#define PRIOR_NORMAL  1
#define PRIOR_UNIFORM 2
#define PRIOR_NDG     3

/*
	PRIOR: Normal, Uniform, or adaptive Non-Degeneracy (NDG).

	NDG (adaptive importance-reweighting estimator) -- 3D in this build
	-------------------------------------------------------------------
	The prior probability that the ERGM sufficient statistic g(Y) lies
	in the interior of the empirical convex hull C of observed auxiliary
	samples. Given a pool of m samples {g_i} drawn at reference parameter
	theta_tilde, the self-normalised IS estimator is

	    Phat(theta; theta_tilde)
	      = sum_i w_i * 1{g_i in int(C)}  /  sum_i w_i
	    w_i = exp( (theta - theta_tilde)^T g_i )

	Strength p is applied as

	    log p_NDG(theta) = p * log Phat(theta; theta_tilde).

	(q is fixed at 0 in this build; the (1 - Phat) factor is not used.)

	The pool is REPLACED at every leapfrog step (set via prior_ndg_set_pool).
	The hull, by contrast, GROWS MONOTONICALLY across the whole run: every
	point that has ever been shown to the prior via hull3d_add (indirectly,
	via prior_ndg_set_pool or prior_ndg_warm_start) stays in the hull
	forever. Interior indicators in_hull[] are recomputed each time the
	pool is set, so they always reflect the most recent hull.

	For the Kapferer / egd model the sufficient statistic is
	(edges, gwesp(0.25), gwdsp(0.25)) which mixes integer (edges) and
	real-valued (gwesp, gwdsp) components, so the 3D hull hull3d_* is
	used -- it stores observations as doubles and triangulates the
	polytope in R^3. This module has no access to any ground-truth hull
	or enumeration -- the hull is built purely from samples drawn by
	the TNT sampler during HMC (with an optional warm-start file of
	pre-drawn samples, whose only data used are the statistics
	themselves).
*/

typedef struct {
	int type;

	/* Parameter-space dimension for Normal / Uniform / NDG. */
	int d;

	/* Normal: N(mean, diag(var)) -- var = variance, not sd */
	double normal_mean[MAXSTATS];
	double normal_var[MAXSTATS];

	/* Uniform: [lower, upper] per dimension */
	double uniform_lower[MAXSTATS];
	double uniform_upper[MAXSTATS];

	/* ---- NDG (adaptive IS) -- 3D ---- */

	/* Current importance-sampling pool. Replaced wholesale at every
	   leapfrog step via prior_ndg_set_pool. Stored as doubles because
	   the Kapferer / egd sufficient statistic is real-valued (gwesp,
	   gwdsp). Parallel arrays of length ndg_m. */
	double *ndg_g1;         /* s1 values (edges)                           */
	double *ndg_g2;         /* s2 values (gwesp)                           */
	double *ndg_g3;         /* s3 values (gwdsp)                           */
	int    *ndg_in_hull;    /* 1 if g_i strictly interior to current hull  */
	int     ndg_m;          /* pool size                                   */
	int     ndg_capacity;   /* allocated capacity of the four arrays       */

	/* Reference parameter theta_tilde (the theta at which the pool was
	   drawn). Used to compute IS weights exp((theta - tref)^T g_i). */
	double  ndg_tref[MAXSTATS];

	/* Strength exponent p: log p_NDG(theta) = p * log Phat(theta). */
	double  ndg_p;

	/* Empirical convex hull in R^3. Owned by this PRIOR object; grows
	   monotonically across the entire HMC run. */
	HULL3D *ndg_hull;
} PRIOR;

/* ------------------------------------------------------------------ */
/*  Constructors                                                      */
/* ------------------------------------------------------------------ */

/* Generic d-dimensional constructors (d <= MAXSTATS). */
PRIOR *prior_create_normal(int d, const double *mean, const double *var);
PRIOR *prior_create_uniform(int d, const double *lower, const double *upper);

/* Create an adaptive NDG prior (3D).
     hull_eps   : numerical tolerance for the 3D hull (visibility,
                  coplanarity, and strict-interior tests). Pass 0 for
                  the default (1e-9). With Kapferer / egd statistics of
                  magnitude O(10^2..10^3), the default leaves ~9 digits
                  of relative tolerance.
     p_strength : exponent applied to log Phat (q is fixed at 0). */
PRIOR *prior_create_ndg(double hull_eps, double p_strength);

void prior_destroy(PRIOR *prior);

/* ------------------------------------------------------------------ */
/*  NDG: adaptive machinery                                           */
/* ------------------------------------------------------------------ */

/* Seed the hull from a warm-start file of auxiliary samples.

   File format: line 1 = N (unsigned integer); lines 2..N+1 = one sample
   per line, white-space separated. The first three fields (s1, s2, s3)
   are used; any additional fields (e.g. a pre-computed hull indicator
   or a reference theta) are read and IGNORED, so the warm start carries
   NO information from any ground-truth enumeration.

   Returns 0 on success, nonzero on failure. No-op (returns 0) for
   non-NDG priors. */
int prior_ndg_warm_start(PRIOR *prior, const char *filename);

/* Replace the IS pool with a fresh batch of m samples drawn at
   theta_tilde. Every sample is also added to the hull (hulls grow
   monotonically). After this call, prior_log_density / prior_log_ratio
   / prior_log_gradient use these samples and this reference parameter
   until the next prior_ndg_set_pool.

   No-op for non-NDG priors. */
void prior_ndg_set_pool(PRIOR        *prior,
                        const double *s1_samples,
                        const double *s2_samples,
                        const double *s3_samples,
                        int           m,
                        const double *theta_tilde);

/* ------------------------------------------------------------------ */
/*  Density, ratio, gradient                                          */
/* ------------------------------------------------------------------ */

/* log p(theta). For NDG, uses the current pool and theta_tilde. */
double prior_log_density(PRIOR *prior, double *theta);

/* log p(theta_new) - log p(theta_cur).

   For Normal/Uniform: closed-form difference.
   For NDG: evaluated via two prior_log_density calls against the current
            pool. Callers who want the minimum-variance estimate (weights
            identically 1) should ensure the pool is freshly drawn at
            theta_cur when evaluating log p(theta_cur), and freshly drawn
            at theta_new when evaluating log p(theta_new). hmc.c does
            this via cached evaluations at the appropriate leapfrog
            nodes and calls prior_log_density directly rather than
            this ratio helper. */
double prior_log_ratio(PRIOR *prior, double *theta_cur, double *theta_new);

/* Gradient of log p(theta), written into grad_out[0..d-1]. For NDG,
   uses the current pool. */
void prior_log_gradient(PRIOR *prior, double *theta, double *grad_out);

#endif
