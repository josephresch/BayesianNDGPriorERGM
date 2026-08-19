#include "hmc.h"
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#define INF 1.0/0.0
#define NINF -1.0/0.0

int main(int nargs, char **args)
{
	/*
		Expected arguments (20 total = program name + 19):
		args[1]  = data file
		args[2]  = model ("egd": edges + gwesp(0.25) + gwdsp(0.25))
		args[3]  = theta_cur[0]   (edges)
		args[4]  = theta_cur[1]   (gwesp)
		args[5]  = theta_cur[2]   (gwdsp)
		args[6]  = sampler ("tnt")
		args[7]  = tnt_p
		args[8]  = grad_gap
		args[9]  = correction type ("una", "end", "bri")
		args[10] = prior type ("normal", "uniform", "ndg")
		args[11] = ndg warm-start file (or "none" -- used only by ndg prior)
		args[12] = ndg strength p (ignored unless prior_type == "ndg")
		args[13] = seed
		args[14] = results file
		args[15] = log file
		args[16] = time file
		args[17] = main_iters
		args[18] = target_accept (dual-averaging target acceptance rate, e.g. 0.651)
		args[19] = "yes"
	*/
	if(nargs!=20)
	{
		fprintf(stderr, "Usage: hmc data model theta0 theta1 theta2 sampler tnt_p grad_gap correction prior_type ndg_file ndg_p seed results log time main_iters target_accept yes\n");
		return EXIT_FAILURE;
	}

	/* variables */
	size_t main_iters=0, nfrogs=0, grad_burnin=0, grad_ndraws=0, grad_gap=0, rng_seed=0, corr_type=0, endterms=0;
	double *theta_cur;
	double tntp;
	double hmc_eps = 0.0;
	double ndg_p_strength = 0.0;
	double target_accept = 0.0;
	int invalidmodel = 1;

	char inputline[1024];
	char results_filename[1024];
	char log_filename[1024];
	char time_filename[1024];
	char model_string[16];
	char prior_type[16];
	char ndg_file[1024];

	double int_time = 2.38/sqrt(2.0);

	gsl_rng *main_rng;

	GRAPH		*graph_orig;
	SAMPLER2	*tntsampler;
	PRIOR		*prior = NULL;

	/* logo */
	FILE *logo_file;
	char logo_line[4096];
	if((logo_file = fopen("logo_file", "r+")) != NULL)
	{
		while(fgets(logo_line, 4096, logo_file) != NULL) {
			printf("%s", logo_line);
		}
		fclose(logo_file);
	}

	/* first read the graph data */
	do
	{
		sscanf(args[1], "%s", inputline);
		if(access(inputline, R_OK) != 0) printf("\t   ERROR: Check file exists!\n");
	}while(access(inputline, R_OK) != 0);

	graph_orig = loadGRAPH(inputline);

	/* check the model type */
	while(invalidmodel)
	{
		sscanf(args[2], "%s", model_string);

		if(strcmp(model_string, "egd") == 0)
		{
			invalidmodel = 0;

			theta_cur = malloc(3*sizeof(double));

			sscanf(args[3], "%lf", &theta_cur[0]);
			sscanf(args[4], "%lf", &theta_cur[1]);
			sscanf(args[5], "%lf", &theta_cur[2]);
		}
		else
		{
			fprintf(stderr, "ERROR: Unsupported model (expected 'egd': edges + gwesp + gwdsp)\n");
			destroyGRAPH(graph_orig);
			return EXIT_FAILURE;
		}
	}

	/* sampler type */
	sscanf(args[6], "%s", inputline);

	if(strcmp(inputline, "tnt") == 0)
	{
		do{
			sscanf(args[7], "%lf", &tntp);
		}while((tntp <= 0.0) || (tntp >= 1.0));
	}
	else
	{
		fprintf(stderr, "ERROR: Unsupported sampler\n");
		destroyGRAPH(graph_orig);
		return EXIT_FAILURE;
	}

	/* read iterations and check */
	sscanf(args[17], "%zu", &main_iters);
	while(main_iters == 0)	{printf("\t Iterations (main): "); scanf("%zu", &main_iters);}
	grad_burnin = 500;  /* Kapferer has 39 nodes (741 dyads) -- adjust as needed */
	grad_ndraws = 10;
	while(grad_burnin == 0) 	{printf("\t Burn in (gradient): "); scanf("%zu", &grad_burnin);}
	while(grad_ndraws == 0) 	{printf("\t Number of draws (gradient): ");	scanf("%zu", &grad_ndraws);}
	sscanf(args[8], "%zu", &grad_gap);
	hmc_eps = 1.0;
	while(hmc_eps <= 0.0) {printf("\t HMC step size: "); scanf("%lf", &hmc_eps);}
	nfrogs = (size_t) int_time/hmc_eps;
	while(nfrogs == 0.0) {printf("\t HMC leapfrogs: "); scanf("%zu", &nfrogs);}

	/* correction type */
	sscanf(args[9], "%s", inputline);
	if(strcmp(inputline, "una") == 0 ) {
		corr_type = 1;
	}
	else if(strcmp(inputline, "end") == 0) {
		printf("\t\tHow many terms: "); scanf("%zu", &endterms);
		corr_type = 2;
	}
	else if(strcmp(inputline, "bri") == 0) {
		corr_type = 3;
	}
	else {
		fprintf(stderr, "ERROR: Unsupported correction\n");
		destroyGRAPH(graph_orig);
		return EXIT_FAILURE;
	}

	/* prior type */
	sscanf(args[10], "%s", prior_type);
	sscanf(args[11], "%s", ndg_file);
	sscanf(args[12], "%lf", &ndg_p_strength);

	if(strcmp(prior_type, "normal") == 0)
	{
		/* Caimo & Friel flat prior: N(0, diag(100)) on edges+gwesp+gwdsp. */
		double mean[3] = {0.0, 0.0, 0.0};
		double var[3]  = {100.0, 100.0, 100.0};
		prior = prior_create_normal(3, mean, var);
		fprintf(stdout, "Prior: Normal(0, diag(100))  (edges+gwesp+gwdsp)\n");
	}
	else if(strcmp(prior_type, "uniform") == 0)
	{
		/* Locked bounds from migration plan:
		     edges ∈ [-8,   2  ]
		     gwesp ∈ [-2,   5  ]
		     gwdsp ∈ [-0.2, 0.05] */
		double lo[3] = {-8.0, -2.0, -0.20};
		double hi[3] = { 2.0,  5.0,  0.05};
		prior = prior_create_uniform(3, lo, hi);
		fprintf(stdout,
			"Prior: Uniform([-8,2] x [-2,5] x [-0.2,0.05])\n");
	}
	else if(strcmp(prior_type, "ndg") == 0)
	{
		/* Adaptive non-degeneracy prior in 3D (edges + gwesp + gwdsp).
		   hull_eps = 0 -> default 1e-9 tolerance from hull3d_create. */
		prior = prior_create_ndg(0.0, ndg_p_strength);
		if(prior == NULL)
		{
			fprintf(stderr, "ERROR: prior_create_ndg failed\n");
			destroyGRAPH(graph_orig);
			free(theta_cur);
			return EXIT_FAILURE;
		}
		if(strcmp(ndg_file, "none") != 0)
		{
			if(prior_ndg_warm_start(prior, ndg_file) != 0)
			{
				fprintf(stderr, "ERROR: could not load NDG warm-start file '%s'\n", ndg_file);
				prior_destroy(prior);
				destroyGRAPH(graph_orig);
				free(theta_cur);
				return EXIT_FAILURE;
			}
			fprintf(stdout, "Prior: NDG (p=%.6lf) warm-started from '%s'\n",
			        ndg_p_strength, ndg_file);
		}
		else
		{
			fprintf(stdout, "Prior: NDG (p=%.6lf) -- no warm start, hull grows from HMC samples\n",
			        ndg_p_strength);
		}
	}
	else
	{
		fprintf(stderr, "ERROR: Unsupported prior type: %s (use normal, uniform, or ndg)\n", prior_type);
		destroyGRAPH(graph_orig);
		free(theta_cur);
		return EXIT_FAILURE;
	}

	/* seed */
	sscanf(args[13], "%zu", &rng_seed);

	/* filenames */
	sscanf(args[14], "%s", results_filename);
	sscanf(args[15], "%s", log_filename);
	sscanf(args[16], "%s", time_filename);

	/* dual-averaging target acceptance rate */
	sscanf(args[18], "%lf", &target_accept);
	while(target_accept <= 0.0 || target_accept >= 1.0) {printf("\t HMC target acceptance rate (0,1): "); scanf("%lf", &target_accept);}

	/* run the hmc algorithm */
	sscanf(args[19],"%s", inputline);
	if(strcmp(inputline, "yes") == 0)
	{
		/* Algorithm clear to run */

		/* allocate and seed the random number generator */
		main_rng = gsl_rng_alloc(gsl_rng_mt19937);
		gsl_rng_set(main_rng, rng_seed);

		/* setup sampler and attach graph */
		tntsampler = tnt_create2(grad_burnin, grad_ndraws, grad_gap, main_rng);
		tnt_attach2(tntsampler, graph_orig, model_string, tntp);

		/* run the hmc */
		hmc_algorithm_massmatrix(main_iters, tntsampler, theta_cur, nfrogs, hmc_eps, corr_type, endterms, prior, results_filename, log_filename, time_filename, target_accept, main_rng);

		tnt_detach2(tntsampler);
		tnt_destroy2(tntsampler);
		gsl_rng_free(main_rng);
	}
	else
	{
		printf("Algorithm not run\n");
	}

	free(theta_cur);
	prior_destroy(prior);
	destroyGRAPH(graph_orig);

	return EXIT_SUCCESS;
}
