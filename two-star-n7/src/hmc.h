/*
	hmc.h
	Hamiltonian Monte Carlo for ERGM
*/

#include <gsl/gsl_rng.h>									/* for rng */
#include "graph_2410.h"			/* graph object */
#include "prior.h"				/* prior distributions */
#include <stdlib.h>												/* for size_t */

void hmc_algorithm_massmatrix(size_t iterations, SAMPLER2 *sampler, double *theta_cur, size_t nfrogs, double hmc_eps, double target_accept, size_t corr_type, size_t endterms, PRIOR *prior, char *results_filename, char *log_filename, char *time_filename, gsl_rng *rng);
