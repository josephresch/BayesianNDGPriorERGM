#ifndef GRAPH_2410_H
#define GRAPH_2410_H

#include <gsl/gsl_rng.h>
#include <stdlib.h> /* for size_t */

#define MAXSTATS 3

/* GWESP / GWDSP decay parameter (alpha). Fixed for this build. */
#define ALPHA_GWESP 0.25
#define ALPHA_GWDSP 0.25

/*
	DYAD OBJECT - DEFINITION
*/
	typedef struct dyad
	{
		size_t 	i;			/* row index of dyad - 0 to n-1, where n = nnodes */
		size_t 	j;			/* col index of dyad - 0 to n-1, where n = nnodes */
		size_t	index;	/* index of the dyad - 0 to N, where N = ndyads */
		int 		edge;		/* is this an edge? */
	} DYAD;

/*
	GRAPH OBJECT - DEFINITION
*/
	typedef struct graph
	{
		size_t 	nnodes;			/* number of nodes */
		size_t 	ndyads;			/* number of dyads */
		size_t 	nedges;			/* number of edges */
		size_t 	*offset;		/* offset vector (to index rows quickly) */
		size_t 	*degree;		/* degree vector (degree of each node) */
		DYAD		**DYADlist;	/* pointers to pointers to dyads (an array of dyad pointers) */
		int 		status;			/* a variable to indicated the status of graph (e.g. ready, not ready, needs updating...) */
	} GRAPH;

/*
	GRAPH OBJECT - LOAD - This function loads a graph from a file
*/
	GRAPH* loadGRAPH(char *filename);

/*
	GRAPH OBJECT - DUPLICATE2 - This function returns an exact copy of the graph
*/
	GRAPH *dupGRAPH2(GRAPH *graph);

/*
	GRAPH OBJECT - DESTROY - This function frees a graph
*/
	void destroyGRAPH(GRAPH *graph);

/*
	SAMPLER2 OBJECT - DEFINITION
*/
	typedef struct sampler2
	{
		/* graph variables */
		int 		graph_attached;
		GRAPH 	*graph_orig;			/* this graph remains unchanged */	
		GRAPH		*graph_copy;			/* this graph is disposable */

		/* sampler variables */
		int 		type; 				/* 1=GIBBS 2=MGIBBS 3=TNT */
		gsl_rng *samprng;			/* rng for the sampler */
		size_t	burnin;				/* how long to run the sampler for before taking draws */
		size_t 	ndraws;				/* the number of draws to take */
		size_t	gap;					/* the gap between draws */

		/* likelihood variables */
		size_t 	nstats;
		double	*stats_orig;	/* stats (won't change) — real-valued for gwesp/gwdsp */
		double 	*stats_copy;	/* stats (will change) */
		double 	*changestats;	/* per-toggle change in each stat — real-valued */

		/* integer edge-count mirror of graph_orig / graph_copy nedges,
		   kept in sync alongside stats_*; used for TNT proposal accounting
		   (array indexing into logproposal_del, edgescale divisor, etc.) */
		size_t	nedges_orig;
		size_t	nedges_copy;

		/* extra rng details */
		unsigned long int samprng_range;			/* rng range */
		unsigned long int samprng_dyadscale;	/* the scale for random dyads */
		unsigned long int samprng_edgescale;	/* the scale for random edges (TNT only) */
		
		/* metropolis variables */
		double logproposal;										/* log proposal ratio */
		double logalpha;											/* log metropolis hastings ratio */
		
		/* dyad / edges indices */
		size_t index;
		size_t *edges_orig;										/* edge indices of graph (won't change) */
		size_t *edges_copy;										/* edge indices of graph (will change) */
		size_t edge_index;										/* location of edge in the edges vector */
		
		/* extra TNT variables */				
		int tnt_case;													/* the executing case of the TNT sampler */
		double p;															/* the tnt parameter p */
		double *logproposal_add;							/* log proposal ratios - adding */
		double *logproposal_del;							/* log proposal ratios - deleting */

		/* model indicator */
		int model[MAXSTATS];									/* edge star triangle gwesp gwdsp ... */

		/* results storage */
		double **samp_stats;									/* a matrix of stats sampled — real-valued */
	}	SAMPLER2;

/*
	SAMPLER OBJECT - CREATE - This function returns a pointer to the sampler
*/
	SAMPLER2 *tnt_create2(size_t burnin, size_t ndraws, size_t gap, gsl_rng *samprng);

/*
	SAMPLER OBJECT - ATTACH - This function attaches a graph to the sampler
*/
	void tnt_attach2(SAMPLER2 *tntsampler, GRAPH *graph, char *model, double p);

/*
	SAMPLER OBJECT - DETACH - Detach the graph from the sampler
*/
	void tnt_detach2(SAMPLER2 *tntsampler);

/*
	SAMPLER OBJECT - RUN - This function runs the sampler
*/
	void tnt_run2(SAMPLER2 *tntsampler, double *theta);

/*
	SAMPLER OBJECT - RESTORE - Restore graph_copy / stats_copy / edges_copy from _orig
*/
	void tnt_restore(SAMPLER2 *tntsampler);

/*
	SAMPLER OBJECT - DESTROY - This funciton frees a sampler
*/
	void tnt_destroy2(SAMPLER2 *tntsampler);

/*
	get_dyad - gets the value of a dyad (edge=1,empty=0)
*/
	int get_dyad(GRAPH *graph, size_t i, size_t j);

/*
	get_edges_change - gets the change in the number of edges by toggling dyad at index
*/
	int get_edges_change(GRAPH *graph, size_t index);

/*
	get_SP - number of shared partners of dyad (i,j): |{k : (i,k) and (j,k) both edges}|
*/
	size_t get_SP(GRAPH *graph, size_t i, size_t j);

/*
	get_gwesp / get_gwdsp - full-graph geometrically-weighted (E|D)SP statistics.
	  f(L, alpha) = exp(alpha) * (1 - (1 - exp(-alpha))^L)
	  gwesp(G, alpha) = sum_{(i,j) : edge} f(SP(i,j), alpha)
	  gwdsp(G, alpha) = sum_{i<j}           f(SP(i,j), alpha)
*/
	double get_gwesp(GRAPH *graph, double alpha);
	double get_gwdsp(GRAPH *graph, double alpha);

/*
	get_gwesp_change / get_gwdsp_change - change in gwesp / gwdsp from toggling dyad at index.
	  Computed on-the-fly (no DSP matrix maintained): O(n) per toggle.
*/
	double get_gwesp_change(GRAPH *graph, size_t index, double alpha);
	double get_gwdsp_change(GRAPH *graph, size_t index, double alpha);

/*
	tnt_accept_change2 - makes all adjustments to accept the change by the TNT sampler
*/
	void tnt_accept_change2(SAMPLER2 *tntsampler, size_t index, size_t edge_index);

#endif /* GRAPH_2410_H */
