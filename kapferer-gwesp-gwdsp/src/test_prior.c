/*
	test_prior.c

	Self-contained verification of hull3d.c and prior.c (adaptive NDG, 3D).

	Covers:
	  1. Hull correctness on hand-crafted 3D geometries (empty, pre-seed
	     caching, tetrahedron seed, cube, interior vs exterior add,
	     monotonic growth).
	  2. IS log-density at theta == theta_tilde (weights == 1, so
	     log Phat = log(#{interior}/m)).
	  3. IS log-density at theta != theta_tilde, cross-checked against a
	     hand-computed log-sum-exp.
	  4. Analytical gradient vs central finite difference of log_density
	     over several (theta, theta_tilde) pairs.
	  5. A machine-readable dump of (pool, hull, thetas, log_density,
	     log_gradient) for the R cross-check in test_prior.R.

	Exit code: 0 if every assertion passes within tolerance; 1 otherwise.
*/

#include "prior.h"
#include "hull3d.h"
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
/*  Test 1: 3D hull geometries                                        */
/* ------------------------------------------------------------------ */

static void test_hull_geometries(void)
{
	printf("\n=== TEST 1: hull3d geometries ===\n");

	/* 1a: empty hull has no seed tetra, no faces, no interior points. */
	{
		HULL3D *h = hull3d_create(0);
		rec("empty hull has 0 faces",         hull3d_n_faces(h)    == 0);
		rec("empty hull has 0 vertices",      hull3d_n_vertices(h) == 0);
		rec("empty hull classifies (0,0,0) as non-interior",
		    hull3d_is_interior(h, 0.0, 0.0, 0.0) == 0);
		hull3d_destroy(h);
	}

	/* 1b: three coplanar points alone cannot seed the tetrahedron. */
	{
		HULL3D *h = hull3d_create(0);
		hull3d_add(h, 0.0, 0.0, 0.0);
		hull3d_add(h, 1.0, 0.0, 0.0);
		hull3d_add(h, 0.0, 1.0, 0.0);
		rec("three coplanar points do not seed tetrahedron",
		    hull3d_n_faces(h) == 0);
		hull3d_destroy(h);
	}

	/* 1c: tetrahedron seed from four affinely-independent points. */
	{
		HULL3D *h = hull3d_create(0);
		hull3d_add(h, 0.0, 0.0, 0.0);
		hull3d_add(h, 1.0, 0.0, 0.0);
		hull3d_add(h, 0.0, 1.0, 0.0);
		hull3d_add(h, 0.0, 0.0, 1.0);
		rec("tetrahedron has 4 faces", hull3d_n_faces(h)    == 4);
		rec("tetrahedron has 4 vertices", hull3d_n_vertices(h) == 4);
		/* Centroid is strictly inside. */
		rec("tetrahedron centroid is strict interior",
		    hull3d_is_interior(h, 0.25, 0.25, 0.25) == 1);
		/* A face vertex is not strict interior. */
		rec("tetrahedron vertex (0,0,0) is not strict interior",
		    hull3d_is_interior(h, 0.0, 0.0, 0.0) == 0);
		/* Exterior point. */
		rec("point outside tetrahedron is not interior",
		    hull3d_is_interior(h, 2.0, 2.0, 2.0) == 0);
		hull3d_destroy(h);
	}

	/* 1d: unit cube [0,1]^3 -- all 8 corners on the hull. */
	{
		HULL3D *h = hull3d_create(0);
		int i, j, k;
		for(i = 0; i <= 1; ++i)
		for(j = 0; j <= 1; ++j)
		for(k = 0; k <= 1; ++k)
			hull3d_add(h, (double)i, (double)j, (double)k);
		rec("unit cube has 8 vertices", hull3d_n_vertices(h) == 8);
		/* A cube triangulated as 2 tris per face has 12 faces. */
		rec("unit cube has 12 triangular faces", hull3d_n_faces(h) == 12);
		rec("cube centroid (0.5,0.5,0.5) is strict interior",
		    hull3d_is_interior(h, 0.5, 0.5, 0.5) == 1);
		rec("cube corner (0,0,0) is not strict interior",
		    hull3d_is_interior(h, 0.0, 0.0, 0.0) == 0);
		rec("outside point (2,0,0) is not interior",
		    hull3d_is_interior(h, 2.0, 0.0, 0.0) == 0);
		hull3d_destroy(h);
	}

	/* 1e: monotonic growth -- interior adds leave the polytope fixed,
	       exterior adds grow it. */
	{
		HULL3D *h = hull3d_create(0);
		hull3d_add(h, 0.0, 0.0, 0.0);
		hull3d_add(h, 1.0, 0.0, 0.0);
		hull3d_add(h, 0.0, 1.0, 0.0);
		hull3d_add(h, 0.0, 0.0, 1.0);
		int nv1 = hull3d_n_vertices(h);
		int nf1 = hull3d_n_faces(h);

		/* Add interior point. */
		hull3d_add(h, 0.25, 0.25, 0.25);
		rec("interior add leaves vertex count unchanged",
		    hull3d_n_vertices(h) == nv1);
		rec("interior add leaves face count unchanged",
		    hull3d_n_faces(h) == nf1);

		/* Add exterior point far away -- must add at least one vertex. */
		hull3d_add(h, 5.0, 5.0, 5.0);
		rec("exterior add grew vertex count",
		    hull3d_n_vertices(h) > nv1);

		hull3d_destroy(h);
	}
}

/* ------------------------------------------------------------------ */
/*  Helpers for building a well-defined 3D test pool                  */
/* ------------------------------------------------------------------ */

/*
	The following tests use a unit cube [0,1]^3 as the hull (seeded from
	all 8 corners) and a 5-sample pool that mixes strict-interior points
	with corner samples (which are on the boundary and therefore NOT
	strict interior). The first three corners are added during the
	warm-seed step so the hull exists before the pool is set.
*/
static void seed_cube(PRIOR *p, const double *tref)
{
	double s1[8], s2[8], s3[8];
	int i = 0, a, b, c;
	for(a = 0; a <= 1; ++a)
	for(b = 0; b <= 1; ++b)
	for(c = 0; c <= 1; ++c)
	{
		s1[i] = (double)a;
		s2[i] = (double)b;
		s3[i] = (double)c;
		++i;
	}
	prior_ndg_set_pool(p, s1, s2, s3, 8, tref);
}

/* ------------------------------------------------------------------ */
/*  Test 2: IS log-density at theta == theta_tilde                    */
/* ------------------------------------------------------------------ */

static void test_is_at_reference(void)
{
	printf("\n=== TEST 2: IS density at theta == theta_tilde ===\n");

	PRIOR *p = prior_create_ndg(0.0, 1.0);
	double tref[3] = {-1.0, 0.5, 0.25};
	seed_cube(p, tref);

	/* 5-sample pool:
	     (0.25, 0.5, 0.5)  -- strict interior
	     (0.5,  0.5, 0.5)  -- strict interior
	     (0.5,  0.25, 0.75)-- strict interior
	     (0.0,  0.0, 0.0)  -- corner (not strict interior)
	     (1.0,  1.0, 1.0)  -- corner (not strict interior)
	   => 3 interior / 5 => Phat = 0.6, log p = log(0.6). */
	double pool_s1[] = {0.25, 0.5, 0.5, 0.0, 1.0};
	double pool_s2[] = {0.5,  0.5, 0.25, 0.0, 1.0};
	double pool_s3[] = {0.5,  0.5, 0.75, 0.0, 1.0};
	int m = 5;
	prior_ndg_set_pool(p, pool_s1, pool_s2, pool_s3, m, tref);

	double theta[3] = {-1.0, 0.5, 0.25};  /* == tref */
	double got = prior_log_density(p, theta);
	double want = log(3.0 / 5.0);
	rec("p=1 log density at theta == tref == log(3/5)",
	    close_abs(got, want, TOL_TIGHT));

	/* Gradient at theta == tref: weights are all 1, so
	     grad = p * (interior_mean - all_mean).
	   interior: (0.25,0.5,0.5), (0.5,0.5,0.5), (0.5,0.25,0.75)
	     mean = (1.25/3, 1.25/3, 1.75/3)
	   all: above + (0,0,0) + (1,1,1)
	     mean = (2.25/5, 2.25/5, 2.75/5). */
	double grad[3] = {0.0, 0.0, 0.0};
	prior_log_gradient(p, theta, grad);

	double want_g0 = (1.25 / 3.0) - (2.25 / 5.0);
	double want_g1 = (1.25 / 3.0) - (2.25 / 5.0);
	double want_g2 = (1.75 / 3.0) - (2.75 / 5.0);
	rec("gradient[0] at theta == tref matches closed form",
	    close_abs(grad[0], want_g0, TOL_TIGHT));
	rec("gradient[1] at theta == tref matches closed form",
	    close_abs(grad[1], want_g1, TOL_TIGHT));
	rec("gradient[2] at theta == tref matches closed form",
	    close_abs(grad[2], want_g2, TOL_TIGHT));

	/* p = 0 trivial case: density flat at 0 regardless of pool. */
	prior_destroy(p);
	p = prior_create_ndg(0.0, 0.0);
	seed_cube(p, tref);
	prior_ndg_set_pool(p, pool_s1, pool_s2, pool_s3, m, tref);
	got = prior_log_density(p, theta);
	rec("p=0 log density is zero",
	    close_abs(got, 0.0, TOL_TIGHT));

	prior_destroy(p);
}

/* ------------------------------------------------------------------ */
/*  Test 3: IS log-density at theta != theta_tilde                    */
/* ------------------------------------------------------------------ */

static void test_is_off_reference(void)
{
	printf("\n=== TEST 3: IS density at theta != theta_tilde ===\n");

	PRIOR *p = prior_create_ndg(0.0, 1.0);
	double tref[3] = {-1.0, 0.25, 0.0};
	seed_cube(p, tref);

	double pool_s1[]  = {0.25, 0.5,  0.5,  0.0, 1.0};
	double pool_s2[]  = {0.5,  0.5,  0.25, 0.0, 1.0};
	double pool_s3[]  = {0.5,  0.5,  0.75, 0.0, 1.0};
	int in_hull_want[]= {1,    1,    1,    0,   0  };
	int m = 5;
	prior_ndg_set_pool(p, pool_s1, pool_s2, pool_s3, m, tref);

	/* Read back ndg_in_hull and verify. */
	int ok_flags = 1;
	int i;
	for(i = 0; i < m; ++i)
		if(p->ndg_in_hull[i] != in_hull_want[i]) ok_flags = 0;
	rec("in_hull flags match expectation", ok_flags);

	/* Evaluate at a clearly off-reference theta. */
	double theta[3] = {0.25, 0.75, 1.0};
	double d0 = theta[0] - tref[0];  /* +1.25 */
	double d1 = theta[1] - tref[1];  /* +0.50 */
	double d2 = theta[2] - tref[2];  /* +1.00 */

	double z[5], mx = -INFINITY, N = 0.0, D = 0.0;
	for(i = 0; i < m; ++i)
	{
		z[i] = d0 * pool_s1[i] + d1 * pool_s2[i] + d2 * pool_s3[i];
		if(z[i] > mx) mx = z[i];
	}
	for(i = 0; i < m; ++i)
	{
		double w = exp(z[i] - mx);
		D += w;
		if(in_hull_want[i]) N += w;
	}
	double want = log(N) - log(D);

	double got = prior_log_density(p, theta);
	rec("p=1 log density off-ref matches hand-computed Phat",
	    close_abs(got, want, TOL_TIGHT));

	/* Strength scaling: p=2 should double the log density. */
	prior_destroy(p);
	p = prior_create_ndg(0.0, 2.0);
	seed_cube(p, tref);
	prior_ndg_set_pool(p, pool_s1, pool_s2, pool_s3, m, tref);
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
	double t[3];
	int k;
	for(k = 0; k < 3; ++k)
	{
		t[0] = theta[0]; t[1] = theta[1]; t[2] = theta[2];
		t[k] = theta[k] + h;
		double fp = prior_log_density(prior, t);
		t[k] = theta[k] - h;
		double fm = prior_log_density(prior, t);
		out[k] = (fp - fm) / (2.0 * h);
	}
}

static void test_gradient_fd(void)
{
	printf("\n=== TEST 4: gradient vs central FD ===\n");

	PRIOR *p = prior_create_ndg(0.0, 1.3);
	double tref[3] = {-1.0, 0.25, 0.0};
	seed_cube(p, tref);

	double pool_s1[] = {0.25, 0.5, 0.5,  0.0, 1.0, 0.1, 0.7};
	double pool_s2[] = {0.5,  0.5, 0.25, 0.0, 1.0, 0.3, 0.2};
	double pool_s3[] = {0.5,  0.5, 0.75, 0.0, 1.0, 0.4, 0.6};
	int m = 7;
	prior_ndg_set_pool(p, pool_s1, pool_s2, pool_s3, m, tref);

	double thetas[][3] = {
		{ -1.0,  0.25, 0.00 },   /* == tref */
		{ -0.5,  0.40, 0.10 },
		{  0.0,  0.00, 0.00 },
		{  0.25, 0.75, 0.50 },
		{ -2.0, -0.25, 0.20 },
	};
	int n_theta = (int)(sizeof(thetas) / sizeof(thetas[0]));

	int k;
	for(k = 0; k < n_theta; ++k)
	{
		double g_ana[3] = {0,0,0}, g_fd[3] = {0,0,0};
		prior_log_gradient(p, thetas[k], g_ana);
		fd_grad(p, thetas[k], 1e-5, g_fd);

		char buf[128];
		int d;
		for(d = 0; d < 3; ++d)
		{
			snprintf(buf, sizeof buf,
			         "theta=(%.2f,%.2f,%.2f) grad[%d] ana=%+.6e fd=%+.6e",
			         thetas[k][0], thetas[k][1], thetas[k][2],
			         d, g_ana[d], g_fd[d]);
			rec(buf, close_rel(g_ana[d], g_fd[d], TOL_LOOSE));
		}
	}

	prior_destroy(p);
}

/* ------------------------------------------------------------------ */
/*  Test 5: machine-readable dump for the R cross-check               */
/* ------------------------------------------------------------------ */
/*
	Writes /tmp/test_prior_dump.txt in this format:

	    POOL
	    m p_strength tref0 tref1 tref2
	    s1_0 s2_0 s3_0
	    ...
	    HULL_OBS
	    n_obs
	    x y z
	    ... (insertion-order observed points)
	    HULL_VERT
	    n_vert
	    idx
	    ... (indices into HULL_OBS)
	    HULL_FACES
	    n_faces
	    a b c
	    ... (triples of indices into HULL_OBS)
	    THETAS
	    n_theta
	    theta0 theta1 theta2 log_density grad0 grad1 grad2
	    ...

	test_prior.R reads this and cross-checks the hull against
	geometry::convhulln, recomputes Phat and the gradient in pure R, and
	verifies numDeriv::grad on the R log-density as well.
*/
static void test_dump_for_r(void)
{
	printf("\n=== TEST 5: dump for R cross-check ===\n");

	PRIOR *p = prior_create_ndg(0.0, 1.3);
	double tref[3] = {-1.0, 0.25, 0.0};
	seed_cube(p, tref);

	double pool_s1[] = {0.25, 0.5, 0.5,  0.0, 1.0, 0.1, 0.7};
	double pool_s2[] = {0.5,  0.5, 0.25, 0.0, 1.0, 0.3, 0.2};
	double pool_s3[] = {0.5,  0.5, 0.75, 0.0, 1.0, 0.4, 0.6};
	int m = 7;
	prior_ndg_set_pool(p, pool_s1, pool_s2, pool_s3, m, tref);

	double thetas[][3] = {
		{ -1.0,  0.25, 0.00 },
		{ -0.5,  0.40, 0.10 },
		{  0.0,  0.00, 0.00 },
		{  0.25, 0.75, 0.50 },
		{ -2.0, -0.25, 0.20 },
	};
	int n_theta = (int)(sizeof(thetas) / sizeof(thetas[0]));

	FILE *f = fopen("/tmp/test_prior_dump.txt", "w");
	if(!f) { rec("open /tmp/test_prior_dump.txt for write", 0); prior_destroy(p); return; }

	/* ---- POOL ---- */
	fprintf(f, "POOL\n");
	fprintf(f, "%d %.17g %.17g %.17g %.17g\n",
	        m, p->ndg_p, tref[0], tref[1], tref[2]);
	int i;
	for(i = 0; i < m; ++i)
		fprintf(f, "%.17g %.17g %.17g\n", pool_s1[i], pool_s2[i], pool_s3[i]);

	/* ---- HULL ---- */
	/* Dump the hull via hull3d_dump into a temp file, read it back in
	   verbatim, and append to our composite dump. Simpler alternative:
	   iterate the HULL3D struct directly. We do the direct walk so the
	   dump is a single fopen. */
	HULL3D *hull = p->ndg_hull;

	fprintf(f, "HULL_OBS\n");
	fprintf(f, "%zu\n", hull3d_n_observed(hull));
	{
		size_t j;
		for(j = 0; j < hull3d_n_observed(hull); ++j)
			fprintf(f, "%.17g %.17g %.17g\n",
			        hull->obs_x[j], hull->obs_y[j], hull->obs_z[j]);
	}

	fprintf(f, "HULL_VERT\n");
	int nv = hull3d_n_vertices(hull);
	fprintf(f, "%d\n", nv);
	{
		int j;
		for(j = 0; j < nv; ++j)
			fprintf(f, "%d\n", hull->vert_idx[j]);
	}

	fprintf(f, "HULL_FACES\n");
	int nf = hull3d_n_faces(hull);
	fprintf(f, "%d\n", nf);
	{
		int j;
		for(j = 0; j < nf; ++j)
			fprintf(f, "%d %d %d\n",
			        hull->face_a[j], hull->face_b[j], hull->face_c[j]);
	}

	/* ---- THETAS ---- */
	fprintf(f, "THETAS\n");
	fprintf(f, "%d\n", n_theta);
	int k;
	for(k = 0; k < n_theta; ++k)
	{
		double lp = prior_log_density(p, thetas[k]);
		double g[3] = {0,0,0};
		prior_log_gradient(p, thetas[k], g);
		fprintf(f, "%.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
		        thetas[k][0], thetas[k][1], thetas[k][2],
		        lp, g[0], g[1], g[2]);
	}
	fclose(f);

	rec("wrote /tmp/test_prior_dump.txt", 1);

	prior_destroy(p);
}

/* ------------------------------------------------------------------ */

int main(void)
{
	printf("test_prior: self-contained verification of hull3d.c + prior.c (3D)\n");

	test_hull_geometries();
	test_is_at_reference();
	test_is_off_reference();
	test_gradient_fd();
	test_dump_for_r();

	printf("\n=== summary: %d passed, %d failed ===\n", g_pass, g_fail);
	return (g_fail == 0) ? 0 : 1;
}
