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
		Expected arguments (19 total = program name + 18):
		args[1]  = data file
		args[2]  = model ("es")
		args[3]  = theta_cur[0]
		args[4]  = theta_cur[1]
		args[5]  = sampler ("tnt")
		args[6]  = tnt_p
		args[7]  = grad_gap
		args[8]  = correction type ("una", "end", "bri")
		args[9]  = prior type ("normal", "uniform", "ndg")
		args[10] = ndg warm-start file (or "none" -- used only by ndg prior)
		args[11] = ndg strength p (ignored unless prior_type == "ndg")
		args[12] = seed
		args[13] = results file
		args[14] = log file
		args[15] = time file
		args[16] = target accept rate (delta for dual-averaging eps tuning)
		args[17] = main_iters (total HMC iterations; 0 triggers interactive prompt)
		args[18] = "yes"
	*/
	if(nargs!=19)
	{
		fprintf(stderr, "Usage: hmc data model theta0 theta1 sampler tnt_p grad_gap correction prior_type ndg_file ndg_p seed results log time target_accept main_iters yes\n");
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

		if(strcmp(model_string, "es") == 0)
		{
			invalidmodel = 0;

			theta_cur = malloc(2*sizeof(double));

			sscanf(args[3], "%lf", &theta_cur[0]);
			sscanf(args[4], "%lf", &theta_cur[1]);
		}
		else
		{
			fprintf(stderr, "ERROR: Unsupported model\n");
			destroyGRAPH(graph_orig);
			return EXIT_FAILURE;
		}
	}

	/* sampler type */
	sscanf(args[5], "%s", inputline);

	if(strcmp(inputline, "tnt") == 0)
	{
		do{
			sscanf(args[6], "%lf", &tntp);
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
	grad_burnin = 500;  /* reduced from 5000: g7 has 7 nodes (21 dyads) vs karate 34 nodes (561 dyads) */
	grad_ndraws = 10;
	while(grad_burnin == 0) 	{printf("\t Burn in (gradient): "); scanf("%zu", &grad_burnin);}
	while(grad_ndraws == 0) 	{printf("\t Number of draws (gradient): ");	scanf("%zu", &grad_ndraws);}
	sscanf(args[7], "%zu", &grad_gap);
	hmc_eps = 1.0;
	while(hmc_eps <= 0.0) {printf("\t HMC step size: "); scanf("%lf", &hmc_eps);}
	nfrogs = (size_t) int_time/hmc_eps;
	while(nfrogs == 0.0) {printf("\t HMC leapfrogs: "); scanf("%zu", &nfrogs);}

	/* correction type */
	sscanf(args[8], "%s", inputline);
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
	sscanf(args[9], "%s", prior_type);
	sscanf(args[10], "%s", ndg_file);
	sscanf(args[11], "%lf", &ndg_p_strength);

	if(strcmp(prior_type, "normal") == 0)
	{
		/* N(0, diag(10)) -- variance = 10 per dimension */
		prior = prior_create_normal(0.0, 0.0, 10.0, 10.0);
		fprintf(stdout, "Prior: Normal(0, diag(10))\n");
	}
	else if(strcmp(prior_type, "uniform") == 0)
	{
		/* Uniform on [-4, 2] x [-0.05, 1] */
		prior = prior_create_uniform(-4.0, 2.0, -0.05, 1.0);
		fprintf(stdout, "Prior: Uniform([-4,2] x [-0.05,1])\n");
	}
	else if(strcmp(prior_type, "ndg") == 0)
	{
		/* Integer bounding box for (s1 = edges, s2 = 2-stars) on an
		   n-node undirected graph:
		     s1_max = n*(n-1)/2
		     s2_max = n*(n-1)*(n-2)/2
		   Derived solely from the graph size -- no ground-truth info. */
		int nn     = graph_orig->nnodes;
		int s1_max = nn * (nn - 1) / 2;
		int s2_max = nn * (nn - 1) * (nn - 2) / 2;

		prior = prior_create_ndg(s1_max, s2_max, ndg_p_strength);
		if(!prior)
		{
			fprintf(stderr, "ERROR: Failed to create NDG prior (bbox=[0,%d]x[0,%d], p=%g)\n",
			        s1_max, s2_max, ndg_p_strength);
			destroyGRAPH(graph_orig);
			free(theta_cur);
			return EXIT_FAILURE;
		}
		if(strcmp(ndg_file, "none") != 0)
		{
			if(prior_ndg_warm_start(prior, ndg_file) != 0)
			{
				fprintf(stderr, "ERROR: Failed to warm-start NDG prior from %s\n", ndg_file);
				prior_destroy(prior);
				destroyGRAPH(graph_orig);
				free(theta_cur);
				return EXIT_FAILURE;
			}
			fprintf(stdout, "Prior: NDG adaptive (p=%g), warm-start from %s\n",
			        ndg_p_strength, ndg_file);
		}
		else
		{
			fprintf(stdout, "Prior: NDG adaptive (p=%g), no warm-start\n",
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
	sscanf(args[12], "%zu", &rng_seed);

	/* filenames */
	sscanf(args[13], "%s", results_filename);
	sscanf(args[14], "%s", log_filename);
	sscanf(args[15], "%s", time_filename);

	/* target accept rate (delta for Nesterov dual-averaging eps tuning) */
	sscanf(args[16], "%lf", &target_accept);

	/* run the hmc algorithm */
	sscanf(args[18],"%s", inputline);
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
		hmc_algorithm_massmatrix(main_iters, tntsampler, theta_cur, nfrogs, hmc_eps, target_accept, corr_type, endterms, prior, results_filename, log_filename, time_filename, main_rng);

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
