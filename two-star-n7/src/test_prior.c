/*
	test_prior.c

	Self-contained verification of hull.c and prior.c (adaptive NDG).

	Covers the parts of the phase-6 verification battery that don't
	require the old closed-form NDG or a full-pipeline metric match:

	  1. Hull correctness on hand-crafted geometries (triangle, square,
	     collinear segment, single point, square with an interior point,
	     and a known 5-point convex set with one interior sample).
	  2. IS log-density against closed form at theta = theta_tilde
	     (weights == 1, so log Phat = log(#{interior}/m)).
	  3. IS log-density against hand-computed values at
	     theta != theta_tilde on a small pool.
	  4. Analytical gradient vs central finite difference of log_density.
	  5. A machine-readable dump of (pool, theta, theta_tilde, log_density,
	     log_gradient) for the R cross-check in test_prior.R.

	Exit code: 0 if every assertion passes within tolerance; 1 otherwise.
*/

#include "prior.h"
#include "hull.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Tolerances and pass/fail bookkeeping                              */
/* ------------------------------------------------------------------ */

#define TOL_TIGHT   1e-12     /* analytical closed forms             */
#define TOL_LOOSE   1e-6      /* central finite differences          */

static int g_pass = 0;
static int g_fail = 0;

static void rec(const char *tag, int ok)
{
	if(ok) g_pass++; else g_fail++;
	printf("  [%s] %s\n", ok ? "PASS" : "FAIL", tag);
}

static int close_abs(double got, double want, double tol)
{
	if(isinf(want) && isinf(got) && (signbit(want) == signbit(got))) return 1;
	return fabs(got - want) <= tol;
}

static int close_rel(double got, double want, double tol)
{
	if(isinf(want) && isinf(got) && (signbit(want) == signbit(got))) return 1;
	double s = fabs(want) > 1.0 ? fabs(want) : 1.0;
	return fabs(got - want) / s <= tol;
}

/* ------------------------------------------------------------------ */
/*  Test 1: hull geometries                                           */
/* ------------------------------------------------------------------ */

static void test_hull_geometries(void)
{
	printf("\n=== TEST 1: hull geometries ===\n");

	/* 1a: empty hull */
	{
		HULL *h = hull_create(10, 10);
		rec("empty hull has 0 vertices",
		    hull_n_vertices(h) == 0);
		hull_destroy(h);
	}

	/* 1b: single point */
	{
		HULL *h = hull_create(10, 10);
		hull_add(h, 3, 4);
		rec("single point has 1 vertex",
		    hull_n_vertices(h) == 1);
		hull_destroy(h);
	}

	/* 1c: collinear segment -- 3 points on a line */
	{
		HULL *h = hull_create(10, 10);
		hull_add(h, 1, 1);
		hull_add(h, 2, 2);
		hull_add(h, 3, 3);
		/* After collinear-pop: only the two endpoints survive.
		   k < 3 => empty strict interior. */
		rec("collinear 3 points have 2 vertices",
		    hull_n_vertices(h) == 2);
		hull_destroy(h);
	}

	/* 1d: triangle -- 3 non-collinear points */
	{
		HULL *h = hull_create(10, 10);
		hull_add(h, 1, 1);
		hull_add(h, 5, 1);
		hull_add(h, 3, 5);
		rec("triangle has 3 vertices", hull_n_vertices(h) == 3);
		hull_destroy(h);
	}

	/* 1e: square with an interior observed point */
	{
		HULL *h = hull_create(10, 10);
		hull_add(h, 0, 0);
		hull_add(h, 4, 0);
		hull_add(h, 4, 4);
		hull_add(h, 0, 4);
		hull_add(h, 2, 2);  /* interior */
		rec("square has 4 vertices (interior (2,2) does not count)",
		    hull_n_vertices(h) == 4);
		rec("(2,2) classified as interior of the square",
		    hull_is_interior(h, 2, 2) == 1);
		rec("(0,0) not strict interior (vertex / boundary)",
		    hull_is_interior(h, 0, 0) == 0);
		rec("(4,0) not strict interior (vertex / boundary)",
		    hull_is_interior(h, 4, 0) == 0);
		hull_destroy(h);
	}

	/* 1f: monotonic growth -- adding points never shrinks the hull */
	{
		HULL *h = hull_create(20, 20);
		hull_add(h, 5, 5);
		hull_add(h, 15, 5);
		hull_add(h, 10, 15);
		int nv1 = hull_n_vertices(h);

		hull_add(h, 10, 10);  /* interior to existing triangle */
		int nv2 = hull_n_vertices(h);

		hull_add(h, 20, 20);  /* strictly outside */
		int nv3 = hull_n_vertices(h);

		rec("after interior add vertex count unchanged",
		    nv1 == nv2 && nv1 == 3);
		rec("after exterior add vertex count grew to 4",
		    nv3 == 4);
		hull_destroy(h);
	}
}

/* ------------------------------------------------------------------ */
/*  Test 2: IS log-density at theta == theta_tilde                    */
/* ------------------------------------------------------------------ */
/*
	With theta == theta_tilde every IS weight is exactly 1, so

	    Phat = #{interior} / m
	    log p_NDG = p * log(Phat)

	This is the cleanest closed-form check of the density path.
*/

static void test_is_at_reference(void)
{
	printf("\n=== TEST 2: IS density at theta == theta_tilde ===\n");

	PRIOR *p = prior_create_ndg(21, 105, 1.0);

	/* Seed the hull with a square so (2,2) is strictly interior. */
	int seed_s1[] = {0, 4, 4, 0};
	int seed_s2[] = {0, 0, 4, 4};
	double tref[2] = {-1.0, 0.5};
	prior_ndg_set_pool(p, seed_s1, seed_s2, 4, tref);
	/* Reset the pool so we don't count the seeding corners as interior
	   samples of the pool. */

	/* Now build a pool:
	     (2,2) -- interior
	     (2,3) -- interior
	     (3,3) -- interior
	     (0,0) -- boundary (vertex)
	     (4,4) -- boundary (vertex)
	   => 3 interior out of 5 => Phat = 0.6, log p = log(0.6). */
	int pool_s1[] = {2, 2, 3, 0, 4};
	int pool_s2[] = {2, 3, 3, 0, 4};
	prior_ndg_set_pool(p, pool_s1, pool_s2, 5, tref);

	double theta[2] = {-1.0, 0.5};  /* == tref */
	double got = prior_log_density(p, theta);
	double want = log(3.0 / 5.0);
	rec("p=1 log density at theta == tref == log(3/5)",
	    close_abs(got, want, TOL_TIGHT));

	/* Gradient at theta == tref should be exactly zero up to rounding:
	   the ratio (N_{g}/N - D_{g}/D) still depends on g distribution,
	   and weights are all 1, so it becomes
	     mean(g | interior) - mean(g | all).
	   That's generally NOT zero. We check the numeric values against
	   the hand-computed means. */
	double grad[2] = {0.0, 0.0};
	prior_log_gradient(p, theta, grad);

	/* interior pool: (2,2),(2,3),(3,3) -> mean = (7/3, 8/3)
	   all pool:       above + (0,0),(4,4) -> mean = (11/5, 12/5)
	   grad = p * (interior_mean - all_mean) = (7/3 - 11/5, 8/3 - 12/5) */
	double want_g0 = (7.0/3.0) - (11.0/5.0);
	double want_g1 = (8.0/3.0) - (12.0/5.0);
	rec("gradient[0] at theta == tref matches closed form",
	    close_abs(grad[0], want_g0, TOL_TIGHT));
	rec("gradient[1] at theta == tref matches closed form",
	    close_abs(grad[1], want_g1, TOL_TIGHT));

	/* p = 0 trivial case: density flat at 0 regardless of pool. */
	prior_destroy(p);
	p = prior_create_ndg(21, 105, 0.0);
	prior_ndg_set_pool(p, pool_s1, pool_s2, 5, tref);
	got = prior_log_density(p, theta);
	rec("p=0 log density is zero",
	    close_abs(got, 0.0, TOL_TIGHT));

	prior_destroy(p);
}

/* ------------------------------------------------------------------ */
/*  Test 3: IS log-density at theta != theta_tilde                    */
/* ------------------------------------------------------------------ */
/*
	Hand-compute Phat for a 5-sample pool and a specific (theta,tref):

	    w_i  = exp( (theta - tref)^T g_i )
	    N    = sum_{i: interior} w_i
	    D    = sum_i             w_i
	    Phat = N / D

	We then check the C implementation reproduces this via log-sum-exp.
*/

static void test_is_off_reference(void)
{
	printf("\n=== TEST 3: IS density at theta != theta_tilde ===\n");

	PRIOR *p = prior_create_ndg(21, 105, 1.0);

	/* Seed hull so (2,2),(2,3),(3,3) are interior; corners are boundary. */
	int seed_s1[] = {0, 4, 4, 0};
	int seed_s2[] = {0, 0, 4, 4};
	double tref[2] = {-1.0, 0.25};
	prior_ndg_set_pool(p, seed_s1, seed_s2, 4, tref);

	int pool_s1[] = {2, 2, 3, 0, 4};
	int pool_s2[] = {2, 3, 3, 0, 4};
	int in_hull[] = {1, 1, 1, 0, 0};  /* expected interior flags */
	int m = 5;

	prior_ndg_set_pool(p, pool_s1, pool_s2, m, tref);

	/* Read back ndg_in_hull and verify against expectation. */
	int ok_flags = 1;
	int i;
	for(i = 0; i < m; ++i)
		if(p->ndg_in_hull[i] != in_hull[i]) ok_flags = 0;
	rec("in_hull flags match expectation", ok_flags);

	/* Evaluate at a clearly off-reference theta. */
	double theta[2] = {0.25, 0.75};
	double d0 = theta[0] - tref[0];  /* +1.25 */
	double d1 = theta[1] - tref[1];  /* +0.50 */

	/* Hand-compute (use double precision consistently). */
	double z[5], mx = -INFINITY, N = 0.0, D = 0.0;
	for(i = 0; i < m; ++i)
	{
		z[i] = d0 * (double)pool_s1[i] + d1 * (double)pool_s2[i];
		if(z[i] > mx) mx = z[i];
	}
	for(i = 0; i < m; ++i)
	{
		double w = exp(z[i] - mx);
		D += w;
		if(in_hull[i]) N += w;
	}
	double want = log(N) - log(D);

	double got = prior_log_density(p, theta);
	rec("p=1 log density off-ref matches hand-computed Phat",
	    close_abs(got, want, TOL_TIGHT));

	/* Strength scaling: p=2 should double the log density. */
	prior_destroy(p);
	p = prior_create_ndg(21, 105, 2.0);
	prior_ndg_set_pool(p, seed_s1, seed_s2, 4, tref);
	prior_ndg_set_pool(p, pool_s1, pool_s2, m, tref);
	double got2 = prior_log_density(p, theta);
	rec("p=2 log density off-ref is 2x the p=1 value",
	    close_abs(got2, 2.0 * want, TOL_TIGHT));

	prior_destroy(p);
}

/* ------------------------------------------------------------------ */
/*  Test 4: analytical gradient vs central finite difference          */
/* ------------------------------------------------------------------ */

static void fd_grad(PRIOR *prior, double *theta, double h, double *out)
{
	double t[2];
	t[0] = theta[0] + h; t[1] = theta[1];
	double fp0 = prior_log_density(prior, t);
	t[0] = theta[0] - h; t[1] = theta[1];
	double fm0 = prior_log_density(prior, t);
	out[0] = (fp0 - fm0) / (2.0 * h);

	t[0] = theta[0]; t[1] = theta[1] + h;
	double fp1 = prior_log_density(prior, t);
	t[0] = theta[0]; t[1] = theta[1] - h;
	double fm1 = prior_log_density(prior, t);
	out[1] = (fp1 - fm1) / (2.0 * h);
}

static void test_gradient_fd(void)
{
	printf("\n=== TEST 4: gradient vs central FD ===\n");

	PRIOR *p = prior_create_ndg(21, 105, 1.3);

	int seed_s1[] = {0, 4, 4, 0};
	int seed_s2[] = {0, 0, 4, 4};
	double tref[2] = {-1.0, 0.25};
	prior_ndg_set_pool(p, seed_s1, seed_s2, 4, tref);

	int pool_s1[] = {2, 2, 3, 0, 4, 1, 3};
	int pool_s2[] = {2, 3, 3, 0, 4, 2, 1};
	int m = 7;
	prior_ndg_set_pool(p, pool_s1, pool_s2, m, tref);

	/* Evaluate at several theta and compare analytical vs FD. */
	double thetas[][2] = {
		{ -1.0,  0.25},   /* == tref */
		{ -0.5,  0.40},
		{  0.0,  0.00},
		{  0.25, 0.75},
		{ -2.0, -0.25},
	};
	int n_theta = (int)(sizeof(thetas) / sizeof(thetas[0]));

	int k;
	for(k = 0; k < n_theta; ++k)
	{
		double g_ana[2] = {0,0}, g_fd[2] = {0,0};
		prior_log_gradient(p, thetas[k], g_ana);
		fd_grad(p, thetas[k], 1e-5, g_fd);

		char buf[96];
		snprintf(buf, sizeof buf,
		         "theta=(%.2f,%.2f) grad[0] ana=%+.6e fd=%+.6e",
		         thetas[k][0], thetas[k][1], g_ana[0], g_fd[0]);
		rec(buf, close_rel(g_ana[0], g_fd[0], TOL_LOOSE));

		snprintf(buf, sizeof buf,
		         "theta=(%.2f,%.2f) grad[1] ana=%+.6e fd=%+.6e",
		         thetas[k][0], thetas[k][1], g_ana[1], g_fd[1]);
		rec(buf, close_rel(g_ana[1], g_fd[1], TOL_LOOSE));
	}

	prior_destroy(p);
}

/* ------------------------------------------------------------------ */
/*  Test 5: machine-readable dump for the R cross-check               */
/* ------------------------------------------------------------------ */
/*
	Writes /tmp/test_prior_dump.txt in a simple tabular format:

	    POOL
	    m p_strength tref0 tref1
	    s1_0 s2_0
	    ...
	    s1_{m-1} s2_{m-1}
	    HULL
	    nv
	    v1_s1 v1_s2
	    ...
	    THETAS
	    n_theta
	    theta0 theta1 log_density grad0 grad1
	    ...

	test_prior.R reads this, recomputes everything in pure R, and checks
	at 1e-10 tolerance. The hull is cross-checked against R's chull().
*/
static void test_dump_for_r(void)
{
	printf("\n=== TEST 5: dump for R cross-check ===\n");

	PRIOR *p = prior_create_ndg(21, 105, 1.3);

	int seed_s1[] = {0, 4, 4, 0};
	int seed_s2[] = {0, 0, 4, 4};
	double tref[2] = {-1.0, 0.25};
	prior_ndg_set_pool(p, seed_s1, seed_s2, 4, tref);

	int pool_s1[] = {2, 2, 3, 0, 4, 1, 3};
	int pool_s2[] = {2, 3, 3, 0, 4, 2, 1};
	int m = 7;
	prior_ndg_set_pool(p, pool_s1, pool_s2, m, tref);

	double thetas[][2] = {
		{ -1.0,  0.25},
		{ -0.5,  0.40},
		{  0.0,  0.00},
		{  0.25, 0.75},
		{ -2.0, -0.25},
	};
	int n_theta = (int)(sizeof(thetas) / sizeof(thetas[0]));

	FILE *f = fopen("/tmp/test_prior_dump.txt", "w");
	if(!f) { rec("open /tmp/test_prior_dump.txt for write", 0); prior_destroy(p); return; }

	fprintf(f, "POOL\n");
	fprintf(f, "%d %.17g %.17g %.17g\n", m, p->ndg_p, tref[0], tref[1]);
	int i;
	for(i = 0; i < m; ++i)
		fprintf(f, "%d %d\n", pool_s1[i], pool_s2[i]);

	fprintf(f, "HULL\n");
	int nv = hull_n_vertices(p->ndg_hull);
	fprintf(f, "%d\n", nv);
	for(i = 0; i < nv; ++i)
		fprintf(f, "%d %d\n", p->ndg_hull->vert_s1[i],
		                       p->ndg_hull->vert_s2[i]);

	fprintf(f, "THETAS\n");
	fprintf(f, "%d\n", n_theta);
	int k;
	for(k = 0; k < n_theta; ++k)
	{
		double lp = prior_log_density(p, thetas[k]);
		double g[2] = {0,0};
		prior_log_gradient(p, thetas[k], g);
		fprintf(f, "%.17g %.17g %.17g %.17g %.17g\n",
		        thetas[k][0], thetas[k][1], lp, g[0], g[1]);
	}
	fclose(f);

	rec("wrote /tmp/test_prior_dump.txt", 1);

	prior_destroy(p);
}

/* ------------------------------------------------------------------ */

int main(void)
{
	printf("test_prior: self-contained verification of hull.c + prior.c\n");

	test_hull_geometries();
	test_is_at_reference();
	test_is_off_reference();
	test_gradient_fd();
	test_dump_for_r();

	printf("\n=== summary: %d passed, %d failed ===\n", g_pass, g_fail);
	return (g_fail == 0) ? 0 : 1;
}
