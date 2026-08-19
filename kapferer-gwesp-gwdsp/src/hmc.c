#include "hmc.h"
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <math.h>
#include <time.h>

#define INF 1.0/0.0
#define NINF -1.0/0.0

/*
	automatic tuning of the hmc stepsize
	adj = 1/(nprops^phi)
	sigma(t) = sigma(t-1)*exp(adj*(current_accept - target_accept))
*/
double hmc_autotune(double current_eps, double target_accept, size_t naccepts, size_t nprops, double phi)
{
	double new_eps;
	double current_accept = (double)naccepts/(double)nprops;
	double adj = (1.0/pow((double)nprops,phi));
	new_eps = current_eps*exp(adj*(current_accept-target_accept));

	return new_eps;
}

double hmc_autotune2(double *Hbar, double *logepsbar, double delta, double logalpha, double m, double mu, double t0, double kappa, double lambda)
{
	double alpha = (logalpha >= 0.0) ? 1 : exp(logalpha);
	double logeps;

	*Hbar = (1.0 - 1.0/(m+t0))*(*Hbar) + (1.0/(m+t0))*(delta - alpha);
	logeps = mu - (sqrt(m)/lambda)*(*Hbar);
	*logepsbar = pow(m, -kappa)*logeps + (1.0 - pow(m, -kappa))*(*logepsbar);

	return logeps;
}

double hmc_autotune3(double current_eps, double target_accept, double logalpha, size_t nprops, double phi)
{
	double new_eps;
	double adj = pow((double)nprops,-phi);
	double alpha = (logalpha >= 0.0) ? 1.0 : exp(logalpha);
	new_eps = current_eps*exp(adj*(alpha-target_accept));

	return new_eps;
}

/* convert a timespec to a number of seconds */
static double get_time_seconds(struct timespec* ts)
{
    return (double)ts->tv_sec + (double)ts->tv_nsec/1000000000.0;
}

/* add numbers safely log scale */
/* checked 15/10/2016 */
double logaddexp(double x, double y)
{
	/* 	if x is max and y is min
			log(exp(x) + exp(y)) = log(exp(x)(1 + exp(y-x))) = x + log(1+exp(y-x) */

	return (x > y) ? (x + log1p(exp(y-x))) : (y + log1p(exp(x-y)));
}

/* get the hmc acceptance ratio
   log_prior_ratio is supplied by the caller. For NDG this is typically
   log_p_new_cache - log_p_cur_cache, each computed against a pool that
   was freshly set at the corresponding leapfrog node (minimum-variance
   IS estimate). For Normal/Uniform it's just prior_log_ratio.

   Momentum ratio uses a general d*d Minv (row-major, symmetric):
       (1/2) (p^T Minv p - evol_p^T Minv evol_p)
   Written as a single double loop so it works for any dimension. */
double get_logalphahmc_massmatrix(SAMPLER2 *sampler, double *theta, double *evol_theta, double *p, double *evol_p, double *Minv, double logzratio, double log_prior_ratio)
{
	double logalphahmc = 0.0;
	double logmomentumratio = 0.0;
	size_t d, e, nd;

	nd = sampler->nstats;

	for(d = 0; d < nd; ++d)
	{
		logalphahmc += (evol_theta[d]-theta[d])*(sampler->stats_orig[d]); 										/* likelihood ratio */
	}

	/* prior ratio (precomputed by caller) */
	logalphahmc += log_prior_ratio;

	for(d = 0; d < nd; ++d)
		for(e = 0; e < nd; ++e)
			logmomentumratio += Minv[d*nd + e] * (p[d]*p[e] - evol_p[d]*evol_p[e]);
	logmomentumratio *= 0.5;

	logalphahmc += logmomentumratio;

	logalphahmc += logzratio;

	return logalphahmc;
}

/* get the logzratio */
/* partially checked */
double get_log_zratio(SAMPLER2 *sampler, double *theta_denom, double *theta_numer)
{
	double log_zratio;
	size_t d;

	if(sampler->ndraws == 1)
	{
		log_zratio = 0.0;
		for(d = 0; d < sampler->nstats; ++d)
			log_zratio += (theta_numer[d] - theta_denom[d])*sampler->samp_stats[0][d];
	}
	else
	{
		size_t i;
		double temp;
		log_zratio = NINF;

		for(i = 0; i < sampler->ndraws; ++i)
		{
			temp = 0.0;
			for(d = 0; d < sampler->nstats; ++d)
			{
				temp += (theta_numer[d] - theta_denom[d])*sampler->samp_stats[i][d];
			}
			log_zratio = logaddexp(log_zratio, temp);
		}

		log_zratio -= log((double)sampler->ndraws);
		//log_zratio -= log((double)sampler->ndraws -1000.0);

	}

	return log_zratio;
}

/* get the logzratio */
/* partially checked */
double get_log_zratio_end(SAMPLER2 *sampler, double *theta_denom, double *theta_numer, size_t nterms)
{
	double log_zratio;
	size_t d;

	if(sampler->ndraws == 1)
	{
		log_zratio = 0.0;
		for(d = 0; d < sampler->nstats; ++d)
			log_zratio += (theta_numer[d] - theta_denom[d])*sampler->samp_stats[0][d];
	}
	else
	{
		size_t i;
		double temp;
		log_zratio = NINF;

		for(i = 0; i < sampler->ndraws; ++i)
		{
			if(i < (sampler->ndraws - nterms))
			{
				continue;
			}

			temp = 0.0;
			for(d = 0; d < sampler->nstats; ++d)
			{
				temp += (theta_numer[d] - theta_denom[d])*sampler->samp_stats[i][d];
			}
			log_zratio = logaddexp(log_zratio, temp);
		}

		log_zratio -= log((double)nterms);

	}

	return log_zratio;
}

/* get the loggrad at theta */
void get_loggrad(SAMPLER2 *sampler, double *theta, PRIOR *prior, double *loggrad)
{
	size_t d;
	double prior_grad[MAXSTATS];
	for(d = 0; d < sampler->nstats; ++d) prior_grad[d] = 0.0;
	prior_log_gradient(prior, theta, prior_grad);

	if(sampler->ndraws == 1)
	{
		for(d = 0; d < sampler->nstats; ++d)
			loggrad[d] = (double)sampler->stats_orig[d] - (double)sampler->samp_stats[0][d] + prior_grad[d];
	}
	else
	{
		size_t i;

		for(d = 0; d < sampler->nstats; ++d)
		{
			loggrad[d] = 0.0;
			for(i = 0; i < sampler->ndraws; ++i)
			{
				loggrad[d] -= (double)sampler->samp_stats[i][d];
			}
			loggrad[d] /= (double)sampler->ndraws;
			loggrad[d] += (double)sampler->stats_orig[d];
			loggrad[d] += prior_grad[d];
		}
	}

	return;
}

/* do cholesky decomposition */
int local_cholesky_decomp(gsl_matrix * A)
{
  const size_t M = A->size1;

  size_t j;

  for (j = 0; j < M; ++j)
  {
    double ajj;
    gsl_vector_view v = gsl_matrix_subcolumn(A, j, j, M - j); /* A(j:n,j) */

    if (j > 0)
    {
      gsl_vector_view w = gsl_matrix_subrow(A, j, 0, j);           /* A(j,1:j-1)^T */
      gsl_matrix_view m = gsl_matrix_submatrix(A, j, 0, M - j, j); /* A(j:n,1:j-1) */

      gsl_blas_dgemv(CblasNoTrans, -1.0, &m.matrix, &w.vector, 1.0, &v.vector);
    }

    ajj = gsl_matrix_get(A, j, j);

    if (ajj <= 0.0)
    {
      GSL_ERROR("matrix is not positive definite", GSL_EDOM);
    }

    ajj = sqrt(ajj);
    gsl_vector_scale(&v.vector, 1.0 / ajj);
  }

	return GSL_SUCCESS;
}

int local_ran_mvgaussian(const gsl_rng * r, const gsl_vector * mu, const gsl_matrix * L, gsl_vector * result)
{
  const size_t M = L->size1;

  size_t i;

  for (i = 0; i < M; ++i)
    gsl_vector_set(result, i, gsl_ran_ugaussian(r));

  gsl_blas_dtrmv(CblasLower, CblasNoTrans, CblasNonUnit, L, result);
  gsl_vector_add(result, mu);

  return GSL_SUCCESS;
}


void hmc_algorithm_massmatrix(size_t iterations, SAMPLER2 *sampler, double *theta_cur, size_t nfrogs, double hmc_eps, size_t corr_type, size_t endterms, PRIOR *prior, char *results_filename, char *log_filename, char *time_filename, double target_accept, gsl_rng *rng)
{
	size_t i;	/* main loop */
	size_t f;	/* frog loop */
	size_t d;	/* stats dim */
	size_t e;	/* stats dim, inner (matvec) */
	size_t accept_count = 0;
	double logzratio;
	double *p, *evol_p, *evol_theta, *evol_theta_lag, *loggrad;
	double hmc_halfeps = 0.5*hmc_eps;
	double logalphahmc = 0.0;
	double time_temp;
	double clock = 0.0;

	/* open the results file */
	FILE *results_file = fopen(results_filename, "w+");
	if(results_file == NULL) { fprintf(stderr, "ERROR: could not open results file\n"); return; }

	/* open the log file */
	FILE *log_file = fopen(log_filename, "w+");
	if(log_file == NULL) { fprintf(stderr, "ERROR: could not open log file\n"); return; }

	/* open the time file */
	FILE *time_file = fopen(time_filename, "w+");
	if(time_file == NULL) { fprintf(stderr, "ERROR: could not open time file\n"); return; }

	/* create the hmc variables */
	p = malloc(sampler->nstats*sizeof(double));
	evol_p = malloc(sampler->nstats*sizeof(double));
	evol_theta = malloc(sampler->nstats*sizeof(double));
	evol_theta_lag = malloc(sampler->nstats*sizeof(double));
	loggrad = malloc(sampler->nstats*sizeof(double));

	/* Scratch buffers for handing the IS pool (adaptive NDG prior) from
	   the TNT sampler's samp_stats[][] matrix to prior_ndg_set_pool.
	   Allocated once, reused at every tnt_run2 site. For non-NDG priors
	   prior_ndg_set_pool is a no-op but we still populate and pass the
	   arrays so the control flow stays uniform. Doubles because the
	   Kapferer / egd sufficient statistic (edges, gwesp(0.25),
	   gwdsp(0.25)) is real-valued. */
	double *pool_s1 = malloc(sampler->ndraws * sizeof(double));
	double *pool_s2 = malloc(sampler->ndraws * sizeof(double));
	double *pool_s3 = malloc(sampler->ndraws * sizeof(double));

	/* Cached log-prior evaluations for the accept/reject step.
	     log_p_cur_cache is computed right after the very first tnt_run2
	     of each iteration, when the IS pool is fresh at theta_cur.
	     log_p_new_cache is computed right after the final tnt_run2, when
	     the pool is fresh at the trajectory endpoint (theta_new).
	   For NDG this gives the minimum-variance IS estimate (weights == 1
	   at each endpoint). For Normal/Uniform prior_log_density is a cheap
	   closed form and the caching is harmless. */
	double log_p_cur_cache = 0.0;
	double log_p_new_cache = 0.0;

	/* adaptive tune 2 */
	double Hbar = 0.0;
	double t0 = 10.0;
	double kappa = 0.75;
	double lambda = 0.05;
	double logepsbar = log(1.0);
	double mu = log(10.0*hmc_eps);
	double int_time = 1.68291413922398;

	/* Mass matrix -- Kapferer network (39 nodes, edges + gwesp + gwdsp),
	   from 02_FindMassMatrix.R (MAP-simulated stat covariance + prior
	   correction; seed=20398, nsim=500, MAP from 01_FindMAP.R). */
	double M[MAXSTATS*MAXSTATS] = {
	    1216.006849699398572, 1865.081165436773972,  4845.486883567362383,
	    1865.081165436773972, 2886.529283705796843,  7414.617605358649598,
	    4845.486883567362383, 7414.617605358649598, 20077.955452080885152
	};

	fprintf(stdout,"M:\n");
	fprintf(stdout,"  %.15lf %.15lf %.15lf\n", M[0], M[1], M[2]);
	fprintf(stdout,"  %.15lf %.15lf %.15lf\n", M[3], M[4], M[5]);
	fprintf(stdout,"  %.15lf %.15lf %.15lf\n", M[6], M[7], M[8]);

	/* Invert M via GSL on a working copy (original M still needed for the
	   in-place cholesky factorisation used by local_ran_mvgaussian below). */
	double Minv[MAXSTATS*MAXSTATS];
	{
		gsl_matrix *Mcopy = gsl_matrix_alloc(sampler->nstats, sampler->nstats);
		gsl_matrix_view M_view_tmp = gsl_matrix_view_array(M, sampler->nstats, sampler->nstats);
		gsl_matrix_memcpy(Mcopy, &M_view_tmp.matrix);
		gsl_linalg_cholesky_decomp1(Mcopy);
		gsl_linalg_cholesky_invert(Mcopy);
		size_t r, c;
		for(r = 0; r < sampler->nstats; ++r)
			for(c = 0; c < sampler->nstats; ++c)
				Minv[r*sampler->nstats + c] = gsl_matrix_get(Mcopy, r, c);
		gsl_matrix_free(Mcopy);
	}

	gsl_matrix_view M_gsl = gsl_matrix_view_array(M, sampler->nstats, sampler->nstats);
	local_cholesky_decomp(&M_gsl.matrix);
	gsl_vector *p_gsl = gsl_vector_calloc(sampler->nstats);
	gsl_vector *mu_vec = gsl_vector_calloc(sampler->nstats);

	/* print boilerplate to log file and results file */
	fprintf(log_file,"HMC Algorithm for ERGM\n");
	fprintf(log_file,"results_file: %s\n", results_filename);
	fprintf(log_file,"nnodes: %zu\n", sampler->graph_orig->nnodes);
	fprintf(log_file,"iterations: %zu\n", iterations);
	fprintf(log_file,"grad_burnin: %zu\n", sampler->burnin);
	fprintf(log_file,"grad_ndraws: %zu\n", sampler->ndraws);
	fprintf(log_file,"grad_gap: %zu\n", sampler->gap);
	fprintf(log_file,"M: %.15lf %.15lf %.15lf %.15lf %.15lf %.15lf %.15lf %.15lf %.15lf\n",
	        M[0], M[1], M[2], M[3], M[4], M[5], M[6], M[7], M[8]);
	fprintf(log_file,"Minv: %.15lf %.15lf %.15lf %.15lf %.15lf %.15lf %.15lf %.15lf %.15lf\n",
	        Minv[0], Minv[1], Minv[2], Minv[3], Minv[4], Minv[5], Minv[6], Minv[7], Minv[8]);
	fprintf(log_file,"z_z'_correction: %zu\n", corr_type);
	fprintf(results_file,"theta_edges theta_gwesp theta_gwdsp\n");
	fprintf(time_file,"iteration time_seconds\n");

	fprintf(stdout,"M (post-cholesky L):\n");
	fprintf(stdout,"  %.15lf %.15lf %.15lf\n", M[0], M[1], M[2]);
	fprintf(stdout,"  %.15lf %.15lf %.15lf\n", M[3], M[4], M[5]);
	fprintf(stdout,"  %.15lf %.15lf %.15lf\n", M[6], M[7], M[8]);

	/* create and start the timer */
  struct timespec start;
  struct timespec end;
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	
	/* main hmc loop */
	for(i = 0; i < iterations; ++i)
	{
		logzratio = 0.0;

		/* generate new momentum variables and set evolved theta and lagged back to current point */
		local_ran_mvgaussian(rng, mu_vec, &M_gsl.matrix, p_gsl);
		for(d = 0; d < sampler->nstats; ++d)
		{
			evol_theta[d] = theta_cur[d]; 
			evol_theta_lag[d] = theta_cur[d];
			//p[d] = gsl_ran_gaussian_ziggurat(rng, momen_sd[d]);
			p[d] = gsl_vector_get(p_gsl,d);
			evol_p[d] = p[d];
		}

		/* get the gradient at evol_theta (== theta_cur at iteration start) */
		tnt_run2(sampler, evol_theta);

		/* Refresh the NDG IS pool at evol_theta = theta_cur, and cache
		   log p(theta_cur) while the pool is fresh at this point. */
		{
			size_t j;
			for(j = 0; j < sampler->ndraws; ++j)
			{
				pool_s1[j] = sampler->samp_stats[j][0];
				pool_s2[j] = sampler->samp_stats[j][1];
				pool_s3[j] = sampler->samp_stats[j][2];
			}
			prior_ndg_set_pool(prior, pool_s1, pool_s2, pool_s3, (int)sampler->ndraws, evol_theta);
			log_p_cur_cache = prior_log_density(prior, evol_theta);
		}

		get_loggrad(sampler, evol_theta, prior, loggrad);
		tnt_restore(sampler);

		/* evolve p forward half a step */
		for(d = 0; d < sampler->nstats; ++d)
			evol_p[d] += hmc_halfeps*loggrad[d];

		/* leap frogs */
		for(f = 0; f < (nfrogs-1); ++f)
		{
			/* step theta forward one full step: evol_theta += hmc_eps * Minv @ evol_p */
			for(d = 0; d < sampler->nstats; ++d)
			{
				double step = 0.0;
				for(e = 0; e < sampler->nstats; ++e)
					step += Minv[d*sampler->nstats + e] * evol_p[e];
				evol_theta[d] += hmc_eps*step;
			}

			/* get the gradient at evol_theta */
			tnt_run2(sampler, evol_theta);

			/* Refresh the NDG IS pool at this leapfrog node. The hull
			   absorbs every sample cumulatively; the pool is replaced
			   wholesale. */
			{
				size_t j;
				for(j = 0; j < sampler->ndraws; ++j)
				{
					pool_s1[j] = sampler->samp_stats[j][0];
					pool_s2[j] = sampler->samp_stats[j][1];
					pool_s3[j] = sampler->samp_stats[j][2];
				}
				prior_ndg_set_pool(prior, pool_s1, pool_s2, pool_s3, (int)sampler->ndraws, evol_theta);
			}

			get_loggrad(sampler, evol_theta, prior, loggrad);

			/* add the logzratio contribution */
			if(corr_type == 3)
			{
				logzratio += get_log_zratio(sampler, evol_theta, evol_theta_lag);
			}

			/* restore the sampler back to observed graph */
			tnt_restore(sampler);

			/* step p forward one full step */
			for(d = 0; d < sampler->nstats; ++d)
			{
				evol_p[d] += hmc_eps*loggrad[d];
			}

			/* update the lagged values */
			for(d = 0; d < sampler->nstats; ++d)
				evol_theta_lag[d] = evol_theta[d];

		}/* end leapfrog loop */

		/* step theta forward one final full step: evol_theta += hmc_eps * Minv @ evol_p */
		for(d = 0; d < sampler->nstats; ++d)
		{
			double step = 0.0;
			for(e = 0; e < sampler->nstats; ++e)
				step += Minv[d*sampler->nstats + e] * evol_p[e];
			evol_theta[d] += hmc_eps*step;
		}

		/* get the gradient at evol_theta (trajectory endpoint == theta_new) */
		tnt_run2(sampler, evol_theta);

		/* Refresh the NDG IS pool at the trajectory endpoint and cache
		   log p(theta_new) while the pool is fresh at this point. */
		{
			size_t j;
			for(j = 0; j < sampler->ndraws; ++j)
			{
				pool_s1[j] = sampler->samp_stats[j][0];
				pool_s2[j] = sampler->samp_stats[j][1];
				pool_s3[j] = sampler->samp_stats[j][2];
			}
			prior_ndg_set_pool(prior, pool_s1, pool_s2, pool_s3, (int)sampler->ndraws, evol_theta);
			log_p_new_cache = prior_log_density(prior, evol_theta);
		}

		get_loggrad(sampler, evol_theta, prior, loggrad);

		/* add the logzratio contribution */
		if(corr_type == 2)
		{
			//logzratio = get_log_zratio(sampler, evol_theta, theta_cur);
			logzratio = get_log_zratio_end(sampler, evol_theta, theta_cur, endterms);
		}
		else if(corr_type == 3)
		{
			logzratio += get_log_zratio(sampler, evol_theta, evol_theta_lag);
		}

		/* restore the sampler back to observed graph */			
		tnt_restore(sampler);

		/* evolve p forward half a step */
		for(d = 0; d < sampler->nstats; ++d)
			evol_p[d] += hmc_halfeps*loggrad[d];

		/* get the hmc acceptance ratio.
		   log_prior_ratio uses the two cached evaluations, each taken
		   when the IS pool was fresh at the corresponding leapfrog node
		   (theta_cur for log_p_cur_cache, theta_new for log_p_new_cache).
		   For Normal/Uniform priors these are just the closed-form log
		   densities and the cache is irrelevant. */
		double log_prior_ratio = log_p_new_cache - log_p_cur_cache;
		logalphahmc = get_logalphahmc_massmatrix(sampler, theta_cur, evol_theta, p, evol_p, Minv, logzratio, log_prior_ratio);

		/* calculate the log acceptance probability */
		if(logalphahmc >= 0.0 || log(gsl_rng_uniform_pos(rng)) < logalphahmc)
		{
			for(d = 0; d < sampler->nstats; ++d)
				theta_cur[d] = evol_theta[d];
			++accept_count;
		}

		/* adaptively tune the stepsize */	
		if(i < 1000)
		{
			//hmc_eps = hmc_autotune(hmc_eps, 0.651, accept_count, i+1, 1.00);
			//nfrogs = (size_t) (int_time/hmc_eps);
			//if(nfrogs == 0) nfrogs=1;
			//hmc_eps = hmc_autotune3(hmc_eps, 0.80, logalphahmc, i+1, 1.00);
			//hmc_halfeps = 0.5*hmc_eps;
		}
		

		/* adaptively tune the stepsize */
		if(i < 1000)
		{
			hmc_eps = exp(hmc_autotune2(&Hbar, &logepsbar, target_accept, logalphahmc, 1.0+i, mu, t0, kappa, lambda));
			hmc_halfeps = 0.5*hmc_eps;
			nfrogs = (size_t) (int_time/hmc_eps);
			if(nfrogs == 0) nfrogs=1;
			if(nfrogs > 50) { nfrogs = 50; hmc_eps = int_time/50.0; hmc_halfeps = 0.5*hmc_eps; }
			fprintf(stdout, "%zu eps: %.15lf epsbar: %.15lf nfrogs: %zu (%zu) theta: %.15lf %.15lf %.15lf\n",i, hmc_eps, exp(logepsbar), nfrogs, (size_t)(int_time/exp(logepsbar)), theta_cur[0], theta_cur[1], theta_cur[2]);
		}
		if(i == 1000)
		{
			hmc_eps = exp(logepsbar);
			hmc_halfeps = 0.5*hmc_eps;
			nfrogs = (size_t) (int_time/hmc_eps);
			if(nfrogs == 0) nfrogs=1;
			if(nfrogs > 50) { nfrogs = 50; hmc_eps = int_time/50.0; hmc_halfeps = 0.5*hmc_eps; }
			fprintf(stdout, "%zu eps: %.15lf epsbar: %.15lf nfrogs: %zu theta: %.15lf %.15lf %.15lf\n",i, hmc_eps, hmc_eps, nfrogs, theta_cur[0], theta_cur[1], theta_cur[2]);
		}

		fprintf(results_file, "%+.16lf %+.16lf %+.16lf\n", theta_cur[0], theta_cur[1], theta_cur[2]);
		fflush(results_file);

		/* stop the code every 1000th iteration and log the time */
		if((i % 100) == 0)
		{
			clock_gettime(CLOCK_MONOTONIC_RAW, &end);
			time_temp = get_time_seconds(&end) - get_time_seconds(&start);
			clock += time_temp;
			fprintf(time_file,"%zu %.16lf\n", i, clock);
			fprintf(stdout, "stepsize %.15lf halfstepsize %.15lf L %zu\n", hmc_eps, hmc_halfeps, nfrogs);
			clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		}
	}

	/* stop the timer */
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	time_temp = get_time_seconds(&end) - get_time_seconds(&start);
	clock += time_temp;

	fprintf(log_file, "accepted: %zu out of %zu (%.3lf %%)\n", accept_count, iterations, (100.0*accept_count)/(double)iterations);
	fprintf(log_file, "time_(seconds): %.15lf\n", clock);

	free(p);
	free(evol_p);
	free(evol_theta);
	free(evol_theta_lag);
	free(loggrad);
	free(pool_s1);
	free(pool_s2);
	free(pool_s3);

	fclose(results_file);
	fclose(log_file);
	fclose(time_file);

	gsl_vector_free(p_gsl);
	gsl_vector_free(mu_vec);

	return;
}
