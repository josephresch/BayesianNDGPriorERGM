#include "graph_2410.h"
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_sf_gamma.h>		/* for the n choose m function */
#include <gsl/gsl_randist.h>		/* for the graph with random edges */

#define INF 1.0/0.0
#define MAXSTATS 3

/*
	GRAPH OBJECT - CREATE - This function returns a pointer to a graph, if makerandom is set then edge placement will be randomised
*/
	GRAPH* createGRAPH(size_t nnodes, size_t nedges, int makerandom, gsl_rng *rng)
	{
		#if DEBUG
		if(nnodes > 1000)
		{
			fprintf(stderr, "WARNING: Large graph, recompile to remove this warning!\n");
			return NULL;
		}
		#endif

		GRAPH *graph = malloc(sizeof(GRAPH)); 	/* allocate memory for one GRAPH */
		size_t i,j,index;

		/* set the number of nodes and dyads */
		graph->nnodes = nnodes;
		graph->ndyads = (size_t)gsl_sf_choose(nnodes,2);

		/* check nedges <= ndyads */
		if(nedges > graph->ndyads) {
				fprintf(stderr, "ERROR: Edges greater than number of dyads\n");
				free(graph);
				return NULL;
		}
		graph->nedges = nedges;	

		/* create the offset vector 
			 the first offset is 0, then follows the recursion: offset[k] = offset[k-1] + (nnodes-k) */
		graph->offset = malloc((nnodes-1)*sizeof(size_t));
		graph->offset[0] = 0;
		for(i = 1; i < (nnodes-1); ++i)
			graph->offset[i] = graph->offset[i-1] + (nnodes-i);

		/* create the degree vector */
		graph->degree = calloc(nnodes, sizeof(size_t));

		/* allocate memory for the dyad list */
		graph->DYADlist = malloc(graph->ndyads*sizeof(DYAD *));

		if(makerandom)
		{
			/* Spread the nedges randomly across all dyads */
		
			/* allocate a vector that holds dyads available for selection */			
			size_t *available_dyads = malloc(graph->ndyads*sizeof(size_t));
			for(i = 0; i < graph->ndyads; ++i)
				available_dyads[i] = i;
			
			/* randomly select nedges edges from dyads */
			size_t *selected_edges = malloc(graph->nedges*sizeof(size_t));
			gsl_ran_choose(rng, selected_edges, nedges, available_dyads, graph->ndyads, sizeof(size_t));
			free(available_dyads);

			index = 0;
			size_t next_edge_index = 0, edge_count = 0;
			if(nedges > 0)
				next_edge_index = selected_edges[0];

			for(i = 0; i < nnodes; ++i)
			{
				for(j = i+1; j < nnodes; ++j)
				{
					graph->DYADlist[index] = malloc(sizeof(DYAD));
					graph->DYADlist[index]->index = index;
					graph->DYADlist[index]->i = i;
					graph->DYADlist[index]->j = j;
					if(nedges > 0 && (index == next_edge_index))
					{
						graph->DYADlist[index]->edge = 1;
						graph->degree[i]++;
						graph->degree[j]++;
						edge_count++;
						if(edge_count < nedges)
							next_edge_index = selected_edges[edge_count];
					}
					else
					{
						graph->DYADlist[index]->edge = 0;
					}
					index++;
				}/* end of nodej loop */
			}/* end of nodei loop */
			free(selected_edges);
		}/* end of makerandom check */
		else
		{
			/* set the first nedges as edges */
			index = 0;
			for(i = 0; i < nnodes; ++i)
			{
				for(j = i+1; j < nnodes; ++j)
				{
					graph->DYADlist[index] = malloc(sizeof(DYAD));
					graph->DYADlist[index]->index = index;
					graph->DYADlist[index]->i = i;
					graph->DYADlist[index]->j = j;

					if(nedges) {
						graph->DYADlist[index]->edge = 1;
						graph->degree[i]++;
						graph->degree[j]++;
						nedges--;
					}
					else {
						graph->DYADlist[index]->edge = 0;
					}

					index++;
				}/* end of nodej loop */
			}/* end of nodei loop */
		}/* end of makerandom check */

		/* set the status */
		graph->status = 0;

		return graph;
	}

/*
	GRAPH OBJECT - LOAD - This function loads a graph from a file
*/
	GRAPH* loadGRAPH(char *filename)
	{
		GRAPH *graph;
		FILE *graph_file;

		/* check if file exists (read access) */
		if(access(filename, R_OK) != 0)
		{
			fprintf(stderr, "ERROR: graph file (%s) cannot be read!\n", filename);
			return NULL;
		}

		graph_file = fopen(filename, "r");

		char line[4096];
		size_t i, j, index, nnodes;
		int val, format;

		/* read the format of the file */
		if(fgets(line, sizeof(line), graph_file)!=NULL)
		{
			sscanf(line, "%*s %d", &format);
		}
		else
		{
			fprintf(stderr, "ERROR: cannot read graph format!\n");
			fclose(graph_file);
			return NULL;
		}

		/* read nodes */
		if(fgets(line, sizeof(line), graph_file)!=NULL)
		{
			sscanf(line, "%*s %zu", &nnodes);
		}
		else
		{
			fprintf(stderr, "ERROR: cannot read nnodes!\n");
			fclose(graph_file);
			return NULL;
		}

		/* allocate memory for graph and set attributes */
		graph = malloc(sizeof(GRAPH));
		graph->nnodes = nnodes;		
		graph->nedges = 0;
		graph->ndyads = (size_t) gsl_sf_choose(nnodes,2);

		/* create the offset vector 
			 the first offset is 0, then follows the recursion: offset[k] = offset[k-1] + (nnodes-k) */
		graph->offset = malloc((nnodes-1)*sizeof(size_t));
		graph->offset[0] = 0;
		for(i = 1; i < (nnodes-1); ++i)
			graph->offset[i] = graph->offset[i-1] + (nnodes-i);

		/* create the degree vector */
		graph->degree = calloc(nnodes, sizeof(size_t));

		if(format != 1)
		{
			fprintf(stderr, "ERROR: graph format not supported!\n");
			fclose(graph_file);
			free(graph);
			return NULL;
		}

		/* read in graph */
		graph->DYADlist = malloc(graph->ndyads*sizeof(DYAD *));
		index = 0;
		for(i = 0; i < graph->nnodes; ++i)
		{
			if(fgets(line, sizeof(line), graph_file)!=NULL)
			{
				for(j = i+1; j < graph->nnodes; ++j)
				{
					sscanf(&line[2*j], "%d", &val); /* 2 skips over spaces */
					graph->DYADlist[index] = malloc(sizeof(DYAD));
					graph->DYADlist[index]->index = index;
					graph->DYADlist[index]->i = i;
					graph->DYADlist[index]->j = j;
					graph->DYADlist[index]->edge = val;
	
					if(val!=0 && val!=1)
					{
						fprintf(stderr, "ERROR: adjacency matrix is not binary!\n");
					}
				
					if(val)
					{
						graph->nedges++;
						graph->degree[i]++;
						graph->degree[j]++;
					}
					index++;
				}/* nodej loop end */
			}/* line check end */
			else
			{
				fprintf(stderr, "ERROR: cannot read adjaceny matrix!\n");
				destroyGRAPH(graph);
				fclose(graph_file);
				return NULL;
			}
		}/* nodei loop end */

		/* set the status */
		graph->status = 0;

		fclose(graph_file);

		return graph;
	}

/*
	GRAPH OBJECT - SAVE - This function saves a graph to a file
*/
	void saveGRAPH(GRAPH *graph, char *filename)
	{
		FILE *graph_file;

		/* attempt to open file for writing */
		if((graph_file = fopen(filename, "w+"))==NULL)
		{
			fprintf(stderr, "ERROR: cannot open file for saving!\n");
			return;
		}

		size_t i, j;

		/* print format of file */ /* TODO support other formats */
		fprintf(graph_file, "graphformat 1\n");

		/* print nnodes */
		fprintf(graph_file, "nnodes %zu\n", graph->nnodes);

		/* print the adjacency matrix */
		for(i = 0; i < graph->nnodes; ++i)
		{
			for(j = 0; j < graph->nnodes; ++j)
			{
				if(i == j)
					fprintf(graph_file, "0");
				else
					fprintf(graph_file, "%d", get_dyad(graph, i, j));
				
				if(j < (graph->nnodes-1))
					fprintf(graph_file, " "); /* print a space except for last element in a line */
			}/* end nodej loop */
			fprintf(graph_file,"\n");
		}/* end nodei loop */

		fclose(graph_file);

		return;

	}

/*
	GRAPH OBJECT - DUPLICATE2 - This function returns an exact copy of the graph
*/
	GRAPH *dupGRAPH2(GRAPH *graph)
	{
		GRAPH *graphcopy = malloc(sizeof(GRAPH));		/* allocate memory for the graph */
		size_t i, index;

		*graphcopy = *graph; /* do a shallow copy, this will copy everything (including pointers, but not copy what they point to) */

		/* offset deep copy */
		graphcopy->offset = malloc((graphcopy->nnodes-1)*sizeof(size_t));
		for(i = 0; i < (graphcopy->nnodes-1); ++i)
			graphcopy->offset[i] = graph->offset[i];

		/* degree deep copy */
		graphcopy->degree = malloc(graphcopy->nnodes*sizeof(size_t));
		for(i = 0; i < graphcopy->nnodes; ++i)
			graphcopy->degree[i] = graph->degree[i];

		/* dyadlist deep copy */
		graphcopy->DYADlist = malloc(graphcopy->ndyads*sizeof(DYAD *));
		for(index = 0; index < graphcopy->ndyads; ++index)
		{
			graphcopy->DYADlist[index] = malloc(sizeof(DYAD));
			graphcopy->DYADlist[index]->index = graph->DYADlist[index]->index;
			graphcopy->DYADlist[index]->i = graph->DYADlist[index]->i;
			graphcopy->DYADlist[index]->j = graph->DYADlist[index]->j;
			graphcopy->DYADlist[index]->edge = graph->DYADlist[index]->edge;
		}
		
		return graphcopy;
	}

/*
	GRAPH OBJECT - DESTROY - This function frees a graph
*/
	void destroyGRAPH(GRAPH *graph)
	{
		size_t i;
		for(i = 0; i < graph->ndyads; ++i)
			free(graph->DYADlist[i]);
		free(graph->DYADlist);

		free(graph->offset);	/* offset vector bye bye */
		free(graph->degree);	/* degree vector bye bye */
		free(graph);
	
		return;
	}

/*
	SAMPLER OBJECT - CREATE - This function returns a pointer to the sampler
*/
	SAMPLER *tnt_create(size_t burnin, size_t ndraws, size_t gap, gsl_rng *samprng)
	{
		SAMPLER *tntsampler = malloc(sizeof(SAMPLER));	/* allocate memory for one SAMPLER */

		/* set the type and graph attached */
		tntsampler->graph_attached = 0;		
		tntsampler->type = 3; /* 3 for TNT */

		/* random number gen */
		tntsampler->samprng = samprng;
		tntsampler->samprng_range = samprng->type->max - samprng->type->min;

		/* iterations etc. */
		tntsampler->burnin	= burnin;
		tntsampler->ndraws	= ndraws;
		tntsampler->gap			= gap;

		return tntsampler;
	}

	SAMPLER2 *tnt_create2(size_t burnin, size_t ndraws, size_t gap, gsl_rng *samprng)
	{
		SAMPLER2 *tntsampler = malloc(sizeof(SAMPLER2));	/* allocate memory for one SAMPLER */

		/* set the type and set graph attached to 0 */
		tntsampler->graph_attached = 0;		
		tntsampler->type = 3; /* 3 for TNT */

		/* random number gen */
		tntsampler->samprng = samprng;
		tntsampler->samprng_range = samprng->type->max - samprng->type->min;

		/* iterations etc. */
		tntsampler->burnin	= burnin;
		tntsampler->ndraws	= ndraws;
		tntsampler->gap			= gap;

		return tntsampler;
	}

/*
	SAMPLER OBJECT - ATTACH - This function attaches a graph to the sampler
*/
	void tnt_attach(SAMPLER *tntsampler, GRAPH *graph, char *model, double p)
	{
		size_t i;
		double q, odds, Nodds;

		if(tntsampler->graph_attached)
		{
			fprintf(stderr, "ERROR: trying to attach a graph to sampler which already has a graph attached!\n");
			return;
		}
		if(p <= 0 || p >= 1.0)
		{
			fprintf(stderr, "ERROR: TNT probability must be 0 < p < 1\n");
			return;
		}

		/* set the pointer to the graph */
		tntsampler->graph = graph;
		tntsampler->graph_attached = 1;

		/* parse the model */
		memset(tntsampler->model, 0, MAXSTATS*sizeof(int));
		tntsampler->nstats = 0;
		for(i = 0; i < strlen(model); ++i)
		{
			switch(model[i])
			{
				case 'e':
					if(tntsampler->model[0])
					{
						fprintf(stderr, "ERROR: edge selected more than once\n");
					}
					else
					{
						tntsampler->model[0] = 1;
						tntsampler->nstats++;
					}
					break;
				case 's':
					if(tntsampler->model[1])
					{
						fprintf(stderr, "ERROR: 2star selected more than once\n");
					}
					else
					{
						tntsampler->model[1] = 1;
						tntsampler->nstats++;
					}
					break;
				case 't':
					if(tntsampler->model[2])
					{
						fprintf(stderr, "ERROR: triangle selected more than once\n");
					}
					else
					{
						tntsampler->model[2] = 1;
						tntsampler->nstats++;
					}
					break;

				default:
					fprintf(stderr, "ERROR: unrecognised stat selected\n");
			}
		}

		/* set up the rng dyadscale */
		tntsampler->samprng_dyadscale	= tntsampler->samprng_range/graph->ndyads;

		/* allocate memory for the stats and changestats */
		tntsampler->stats = malloc(tntsampler->nstats*sizeof(size_t));
		tntsampler->changestats = calloc(tntsampler->nstats, sizeof(int));

		/* calculate the stats (this part is HARDCODED at the moment) */
		tntsampler->stats[0] = graph->nedges; /* TODO - Hardcoded */
		tntsampler->stats[1] = get_stars(graph, 2);

		/* create the edges vector */
		tntsampler->edge_index = 0;
		tntsampler->edges = malloc(graph->ndyads*sizeof(size_t));
		for(size_t j = 0; j < graph->ndyads; ++j)
			if(graph->DYADlist[j]->edge)
				tntsampler->edges[(tntsampler->edge_index)++] = j;

		/* set the log proposals */
		tntsampler->p = p;
		q = 1.0 - p;
		odds = p/q;
		Nodds = graph->ndyads*odds;

		tntsampler->logproposal_add = malloc((graph->ndyads+1)*sizeof(double));
		tntsampler->logproposal_del = malloc((graph->ndyads+1)*sizeof(double));

		/* E = 0 ADD is allowed but DEL is not */
		tntsampler->logproposal_add[0] = log(graph->ndyads*p + q);
		tntsampler->logproposal_del[0] = 0; /* set to 0 or INF, won't be used anyway */

		/* E = 1 */
		tntsampler->logproposal_del[1] = -log(graph->ndyads*p + q);

		/* E = N-1 */
		tntsampler->logproposal_add[graph->ndyads-1] = -log(q);

		/* E = N DEL is allowed but ADD is not */
		tntsampler->logproposal_del[graph->ndyads] = log(q);
		tntsampler->logproposal_add[graph->ndyads] = 0; /* set to 0 or INF, won't be used anyway */

		/* set the add values */
		for(i = 1; i < graph->ndyads-1; ++i)
		{
			tntsampler->logproposal_add[i] = log(Nodds + i + 1.0) - log(i+1.0);
		}

		/* set the delete values */
		for(i = 2; i < graph->ndyads; ++i)
		{
			tntsampler->logproposal_del[i] = log(i) - log(Nodds + i);
		}

		/* set up the samp stats */
		tntsampler->samp_stats = malloc(tntsampler->ndraws*sizeof(size_t *));
		for(size_t i = 0; i < tntsampler->ndraws; ++i)
			tntsampler->samp_stats[i] = malloc(tntsampler->nstats*sizeof(size_t));

		return;
	}

	void tnt_attach2(SAMPLER2 *tntsampler, GRAPH *graph, char *model, double p)
	{
		size_t i;
		double q, odds, Nodds;

		if(tntsampler->graph_attached)
		{
			fprintf(stderr, "ERROR: trying to attach a graph to a sampler which already has a graph attached!\n");
			return;
		}
		if(p <= 0 || p >= 1.0)
		{
			fprintf(stderr, "ERROR: TNT probability must be 0 < p < 1\n");
			return;
		}

		/* set the orig pointer to the graph */
		tntsampler->graph_orig = graph;
		/* make a copy of this graph and put it into the other pointer */
		tntsampler->graph_copy = dupGRAPH2(tntsampler->graph_orig);

		/* set graph attached */
		tntsampler->graph_attached = 1;

		/* parse the model */
		memset(tntsampler->model, 0, MAXSTATS*sizeof(int));
		tntsampler->nstats = 0;
		for(i = 0; i < strlen(model); ++i)
		{
			switch(model[i])
			{
				case 'e':
					if(tntsampler->model[0])
					{
						fprintf(stderr, "ERROR: edge selected more than once\n");
					}
					else
					{
						tntsampler->model[0] = 1;
						tntsampler->nstats++;
					}
					break;
				case 's':
					if(tntsampler->model[1])
					{
						fprintf(stderr, "ERROR: 2star selected more than once\n");
					}
					else
					{
						tntsampler->model[1] = 1;
						tntsampler->nstats++;
					}
					break;
				case 't':
					if(tntsampler->model[2])
					{
						fprintf(stderr, "ERROR: triangle selected more than once\n");
					}
					else
					{
						tntsampler->model[2] = 1;
						tntsampler->nstats++;
					}
					break;

				default:
					fprintf(stderr, "ERROR: unrecognised stat selected\n");
			}
		}

		/* set up the rng dyadscale */
		tntsampler->samprng_dyadscale	= tntsampler->samprng_range/graph->ndyads;

		/* allocate memory for the stats_orig, stats_copy and changestats */
		tntsampler->stats_orig = malloc(tntsampler->nstats*sizeof(size_t));
		tntsampler->stats_copy = malloc(tntsampler->nstats*sizeof(size_t));
		tntsampler->changestats = calloc(tntsampler->nstats, sizeof(int));

		/* calculate the stats (this part is HARDCODED at the moment) */
		tntsampler->stats_orig[0] = tntsampler->stats_copy[0] =  graph->nedges; /* TODO - Hardcoded */
		tntsampler->stats_orig[1] = tntsampler->stats_copy[1] = get_stars(graph, 2);

		/* create the edges_orig and edges_copy vector */
		tntsampler->edge_index = 0;
		tntsampler->edges_orig = malloc(graph->ndyads*sizeof(size_t));
		tntsampler->edges_copy = malloc(graph->ndyads*sizeof(size_t));
		for(size_t j = 0; j < graph->ndyads; ++j)
		{
			if(graph->DYADlist[j]->edge)
			{
				tntsampler->edges_orig[tntsampler->edge_index] = j;
				tntsampler->edges_copy[tntsampler->edge_index] = j;
				(tntsampler->edge_index)++;
			}
		}

		/* set the log proposals */
		tntsampler->p = p;
		q = 1.0 - p;
		odds = p/q;
		Nodds = (graph->ndyads)*odds;

		tntsampler->logproposal_add = malloc((graph->ndyads+1)*sizeof(double));
		tntsampler->logproposal_del = malloc((graph->ndyads+1)*sizeof(double));

		/* E = 0 ADD is allowed but DEL is not */
		tntsampler->logproposal_add[0] = log(graph->ndyads*p + q);
		tntsampler->logproposal_del[0] = 0; /* set to 0 or INF, won't be used anyway */

		/* E = 1 */
		tntsampler->logproposal_del[1] = -log(graph->ndyads*p + q);

		/* E = N-1 */
		tntsampler->logproposal_add[graph->ndyads-1] = -log(q);

		/* E = N DEL is allowed but ADD is not */
		tntsampler->logproposal_del[graph->ndyads] = log(q);
		tntsampler->logproposal_add[graph->ndyads] = 0; /* set to 0 or INF, won't be used anyway */

		/* set the add values */
		for(i = 1; i < graph->ndyads-1; ++i)
		{
			tntsampler->logproposal_add[i] = log(Nodds + i + 1.0) - log(i+1.0);
		}

		/* set the delete values */
		for(i = 2; i < graph->ndyads; ++i)
		{
			tntsampler->logproposal_del[i] = log(i) - log(Nodds + i);
		}

		/* set up the samp stats */
		tntsampler->samp_stats = malloc(tntsampler->ndraws*sizeof(size_t *));
		for(size_t i = 0; i < tntsampler->ndraws; ++i)
			tntsampler->samp_stats[i] = malloc(tntsampler->nstats*sizeof(size_t));

		return;
	}

/*
	SAMPLER OBJECT - DETACH - Detach the graph from the sampler
*/
	void tnt_detach(SAMPLER *tntsampler)
	{
		size_t i;
		tntsampler->graph = NULL;
		free(tntsampler->logproposal_add);
		free(tntsampler->logproposal_del);
		free(tntsampler->stats);
		free(tntsampler->changestats);
		free(tntsampler->edges);

		for(i = 0; i < tntsampler->ndraws; ++i)
			free(tntsampler->samp_stats[i]);
		free(tntsampler->samp_stats);

		tntsampler->graph_attached = 0;
		return;
	}

	void tnt_detach2(SAMPLER2 *tntsampler)
	{
		size_t i;
		tntsampler->graph_orig = NULL;					/* this graph will be free'd elsewhere */
		destroyGRAPH(tntsampler->graph_copy);
		free(tntsampler->logproposal_add);
		free(tntsampler->logproposal_del);
		free(tntsampler->stats_orig);
		free(tntsampler->stats_copy);
		free(tntsampler->changestats);
		free(tntsampler->edges_orig);
		free(tntsampler->edges_copy);

		for(i = 0; i < tntsampler->ndraws; ++i)
			free(tntsampler->samp_stats[i]);
		free(tntsampler->samp_stats);

		tntsampler->graph_attached = 0;
		return;
	}

/*
	SAMPLER OBJECT - COPY - Take a full deep copy of the sampler (including graph if attached)
*/
	SAMPLER* tnt_copy(SAMPLER *sampler)
	{
		SAMPLER *samplercopy = malloc(sizeof(SAMPLER));
		size_t i, j;

		/* do a shallow copy of the sampler first */
		*samplercopy = *sampler;

		/* deep copy graph */
		if(sampler->graph_attached)
		{
			samplercopy->graph = dupGRAPH2(sampler->graph);
		}

		/* deep copy stats and changestats */
		samplercopy->stats = malloc(samplercopy->nstats*sizeof(size_t));
		samplercopy->changestats = calloc(samplercopy->nstats, sizeof(int));
		for(i = 0; i < samplercopy->nstats; ++i)
		{
			samplercopy->stats[i] = sampler->stats[i];
			samplercopy->changestats[i] = sampler->changestats[i];
		}

		/* deep copy edges */
		samplercopy->edges = malloc(samplercopy->graph->ndyads*sizeof(size_t));
		for(i = 0; i < samplercopy->stats[0]; ++i)
			samplercopy->edges[i] = sampler->edges[i];

		/* deep copy log proposal vectors */
		samplercopy->logproposal_add = malloc((samplercopy->graph->ndyads+1)*sizeof(double));
		samplercopy->logproposal_del = malloc((samplercopy->graph->ndyads+1)*sizeof(double));
		for(i = 0; i < samplercopy->graph->ndyads+1; ++i)
		{
			samplercopy->logproposal_add[i] = sampler->logproposal_add[i];
			samplercopy->logproposal_del[i] = sampler->logproposal_del[i];
		}

		/* deep copy samp_stats */
		samplercopy->samp_stats = malloc(samplercopy->ndraws*sizeof(size_t *));
		for(i = 0; i < samplercopy->ndraws; ++i)
		{
			samplercopy->samp_stats[i] = malloc(samplercopy->nstats*sizeof(size_t));
			for(j = 0; j < samplercopy->nstats; ++j)
			{
				samplercopy->samp_stats[i][j] = sampler->samp_stats[i][j];
			}
		}

		return samplercopy;
	}

/*
	SAMPLER OBJECT - COPYGRAPH - Copy the graph from source to destination
*/
	void tnt_copygraph(SAMPLER *sampler_dest, SAMPLER *sampler_source)
	{
		size_t i;
		/* free the graph associated with sampler_dest and copy in the one from source */
		destroyGRAPH(sampler_dest->graph);
		sampler_dest->graph = dupGRAPH2(sampler_source->graph);

		/* copy over the stats */
		for(i = 0; i < sampler_source->nstats; ++i)
		{
			sampler_dest->stats[i] = sampler_source->stats[i];
		}

		/* copy over the edges */
		for(i = 0; i < sampler_source->stats[0]; ++i)
		{
			sampler_dest->edges[i] = sampler_source->edges[i];
		}

		return;
	}

	void tnt_restore(SAMPLER2 *tntsampler)
	{
		size_t k;

		/* restore edges_orig and stats_orig */
		for(k = 0; k < tntsampler->graph_orig->nedges; ++k)
		{
			tntsampler->edges_copy[k] = tntsampler->edges_orig[k]; /* there is only a need to copy the first nedges elements, even though vector is longer */
		}
		for(k = 0; k < tntsampler->nstats; ++k)
		{
			tntsampler->stats_copy[k] = tntsampler->stats_orig[k];
		}

		/* copy the graph, no need to copy offsets */
		/* degree copy */
		for(k = 0; k < tntsampler->graph_orig->nnodes; ++k)
			tntsampler->graph_copy->degree[k] = tntsampler->graph_orig->degree[k];

		/* dyadlist copy */
		for(k = 0; k < tntsampler->graph_orig->ndyads; ++k)
		{
			tntsampler->graph_copy->DYADlist[k]->index = tntsampler->graph_orig->DYADlist[k]->index;
			tntsampler->graph_copy->DYADlist[k]->i = tntsampler->graph_orig->DYADlist[k]->i;
			tntsampler->graph_copy->DYADlist[k]->j = tntsampler->graph_orig->DYADlist[k]->j;
			tntsampler->graph_copy->DYADlist[k]->edge = tntsampler->graph_orig->DYADlist[k]->edge;
		}

		return;
	}

/*
	SAMPLER OBJECT - PRINT - This funciton prints the sampler
*/
	void tnt_print(SAMPLER *tntsampler)
	{
		if(!tntsampler->graph_attached)
		{
			fprintf(stderr, "ERROR: cannot print, no graph attached!\n");
		}
		size_t i;

		printf("type: %d\n", tntsampler->type);
		printf("rng: %s with range %zu\n", gsl_rng_name(tntsampler->samprng), tntsampler->samprng_range);
		printf("burnin: %zu\n", tntsampler->burnin);
		printf("ndraws: %zu\n", tntsampler->ndraws);
		printf("gap: %zu\n", tntsampler->gap);
		printf("nstats: %zu\n", tntsampler->nstats);
		printf("p: %.5lf\n", tntsampler->p);
		printf("model: ");
		for(i = 0; i < MAXSTATS; ++i)
		{
			printf("%d", tntsampler->model[i]);
		}
		printf("\n");
		printf("logproposals\n");
		printf("    add      del\n");
		for(i = 0; i < tntsampler->graph->ndyads+1; ++i)
		{
			printf("%3zu %+.5lf %+.5lf\n", i, tntsampler->logproposal_add[i], tntsampler->logproposal_del[i]);
		}
		printf("stats: ");
		printf("%zu %zu\n", tntsampler->stats[0], tntsampler->stats[1]);

		return;
	}

	void tnt_print2(SAMPLER2 *tntsampler)
	{
		if(!tntsampler->graph_attached)
		{
			fprintf(stderr, "ERROR: cannot print, no graph attached!\n");
		}
		size_t i;

		printf("type: %d\n", tntsampler->type);
		printf("rng: %s with range %zu\n", gsl_rng_name(tntsampler->samprng), tntsampler->samprng_range);
		printf("burnin: %zu\n", tntsampler->burnin);
		printf("ndraws: %zu\n", tntsampler->ndraws);
		printf("gap: %zu\n", tntsampler->gap);
		printf("nstats: %zu\n", tntsampler->nstats);
		printf("p: %.5lf\n", tntsampler->p);
		printf("model: ");
		for(i = 0; i < MAXSTATS; ++i)
		{
			printf("%d", tntsampler->model[i]);
		}
		printf("\n");
		printf("logproposals\n");
		printf("    add      del\n");
		for(i = 0; i < tntsampler->graph_orig->ndyads+1; ++i)
		{
			printf("%3zu %+.5lf %+.5lf\n", i, tntsampler->logproposal_add[i], tntsampler->logproposal_del[i]);
		}
		printf("stats_orig: ");
		printf("%zu %zu\n", tntsampler->stats_orig[0], tntsampler->stats_orig[1]);
		printf("stats_copy: ");
		printf("%zu %zu\n", tntsampler->stats_copy[0], tntsampler->stats_copy[1]);

		return;
	}

/*
	SAMPLER OBJECT - RUN - This function runs the sampler
*/
	void tnt_run(SAMPLER *tntsampler, double *theta)
	{
		size_t draws_taken = 0;
		size_t k;

		/* run the burn in */
		for(k = 0; k < tntsampler->burnin; ++k)
		{
			tntsampler->tnt_case = 	(tntsampler->stats[0] > 0) + (tntsampler->stats[0] > 1) + (tntsampler->stats[0] == tntsampler->graph->ndyads);
			/*
				tnt_case description
				tnt_case = 0 -> E = 0
				tnt_case = 1 -> E = 1
				tnt_case = 2 -> 2 <= E <= N-1
				tnt_case = 3 -> E = N
			*/

			switch(tntsampler->tnt_case)
			{
				case 0:
					/* pick a random dyad */
					do {
	  				tntsampler->index = (((tntsampler->samprng->type->get) (tntsampler->samprng->state))) / (tntsampler->samprng_dyadscale);
	  			} while(tntsampler->index >= tntsampler->graph->ndyads);

					tntsampler->logproposal = tntsampler->logproposal_add[0];
					break;

				case 1:
					/* decide whether to pick an edge or pick a dyad */
					if(gsl_rng_uniform(tntsampler->samprng) < tntsampler->p)
					{
						/* pick the only edge and delete */
						tntsampler->edge_index = 0;
						tntsampler->logproposal = tntsampler->logproposal_del[1];
					}
					else
					{
						/* pick a random dyad */
						do {
							tntsampler->index = (((tntsampler->samprng->type->get) (tntsampler->samprng->state))) / (tntsampler->samprng_dyadscale);
						} while(tntsampler->index >= tntsampler->graph->ndyads);

						/* check if this dyad is an edge */
						if(tntsampler->graph->DYADlist[tntsampler->index]->edge)
						{
							/* pick the only edge and delete */
							tntsampler->edge_index = 0;
							tntsampler->logproposal = tntsampler->logproposal_del[1];
						}
						else
						{
							tntsampler->logproposal = tntsampler->logproposal_add[1];	/* adding */
						}
					}
					break;

				case 2:
					/* decide whether to pick an edge or pick a dyad */
					if(gsl_rng_uniform(tntsampler->samprng) < tntsampler->p)
					{
						/* pick an edge and delete */
						tntsampler->samprng_edgescale = tntsampler->samprng_range/tntsampler->stats[0];
						do {
							tntsampler->edge_index = (((tntsampler->samprng->type->get) (tntsampler->samprng->state))) / tntsampler->samprng_edgescale;
						} while(tntsampler->edge_index >= tntsampler->stats[0]);

						/* get the index of this edge */
						tntsampler->index = tntsampler->edges[tntsampler->edge_index];
						tntsampler->logproposal = tntsampler->logproposal_del[tntsampler->stats[0]];
					}
					else
					{
						/* pick a random dyad */
						do {
							tntsampler->index = (((tntsampler->samprng->type->get) (tntsampler->samprng->state))) / (tntsampler->samprng_dyadscale);
						} while(tntsampler->index >= tntsampler->graph->ndyads);

						/* check if this dyad is an edge */
						if(tntsampler->graph->DYADlist[tntsampler->index]->edge)
						{
							/* need to find the edge_index of this edge */
							for(tntsampler->edge_index = 0; tntsampler->edge_index < tntsampler->stats[0]; ++(tntsampler->edge_index))
							{
								if(tntsampler->edges[tntsampler->edge_index] == tntsampler->index) {break;}
							}
							tntsampler->logproposal = tntsampler->logproposal_del[tntsampler->stats[0]];
						}
						else
						{
							tntsampler->logproposal = tntsampler->logproposal_add[tntsampler->stats[0]];	/* adding */
						}
					}
					break;

				case 3:
					/* Pick a random edge - Since every dyad is an edge we can use rng_dyadscale rather than calculating rng_edgescale */
					do
	  			{
	  				tntsampler->edge_index = (((tntsampler->samprng->type->get) (tntsampler->samprng->state))) / (tntsampler->samprng_dyadscale);
	  			}
	  			while(tntsampler->edge_index >= tntsampler->graph->ndyads);

					tntsampler->logproposal = tntsampler->logproposal_add[tntsampler->graph->ndyads];
					break;

				default:
					/* should not happen but have a warning ready regardless */
					fprintf(stderr, "ERROR: tnt_case invalid, case %d\n", tntsampler->tnt_case);
			}

			/* set the changestats for the selected dyad */
			tntsampler->changestats[0] = get_edges_change(tntsampler->graph, tntsampler->index);
			tntsampler->changestats[1] = get_stars_change(tntsampler->graph, tntsampler->index);

	  	/* get the log acceptance probability logP(new_graph) - logP(old_graph) */
	  	tntsampler->logalpha = tntsampler->logproposal + theta[0]*tntsampler->changestats[0] + theta[1]*tntsampler->changestats[1];

	  	/* do the update with log probability logalpha */
			if((tntsampler->logalpha >= 0.0) || (log(gsl_rng_uniform_pos(tntsampler->samprng)) < tntsampler->logalpha))
			{
				tnt_accept_change(tntsampler, tntsampler->index, tntsampler->edge_index);
			}
		}

		while(draws_taken < tntsampler->ndraws)
		{
				/* run for gap + 1 */
				for(k = 0; k < tntsampler->gap+1; ++k)
				{
					tntsampler->tnt_case = 	(tntsampler->stats[0] > 0) + (tntsampler->stats[0] > 1) + (tntsampler->stats[0] == tntsampler->graph->ndyads);
					/*
						tnt_case description
						tnt_case = 0 -> E = 0
						tnt_case = 1 -> E = 1
						tnt_case = 2 -> 2 <= E <= N-1
						tnt_case = 3 -> E = N
					*/

					switch(tntsampler->tnt_case)
					{
						case 0:
							/* pick a random dyad */
							do {
								tntsampler->index = (((tntsampler->samprng->type->get) (tntsampler->samprng->state))) / (tntsampler->samprng_dyadscale);
							} while(tntsampler->index >= tntsampler->graph->ndyads);

							tntsampler->logproposal = tntsampler->logproposal_add[0];
							break;

						case 1:
							/* decide whether to pick an edge or pick a dyad */
							if(gsl_rng_uniform(tntsampler->samprng) < tntsampler->p)
							{
								/* pick the only edge and delete */
								tntsampler->edge_index = 0;
								tntsampler->logproposal = tntsampler->logproposal_del[1];
							}
							else
							{
								/* pick a random dyad */
								do {
									tntsampler->index = (((tntsampler->samprng->type->get) (tntsampler->samprng->state))) / (tntsampler->samprng_dyadscale);
								} while(tntsampler->index >= tntsampler->graph->ndyads);

								/* check if this dyad is an edge */
								if(tntsampler->graph->DYADlist[tntsampler->index]->edge)
								{
									/* pick the only edge and delete */
									tntsampler->edge_index = 0;
									tntsampler->logproposal = tntsampler->logproposal_del[1];
								}
								else
								{
									tntsampler->logproposal = tntsampler->logproposal_add[1];	/* adding */
								}
							}
							break;

						case 2:
							/* decide whether to pick an edge or pick a dyad */
							if(gsl_rng_uniform(tntsampler->samprng) < tntsampler->p)
							{
								/* pick an edge and delete */
								tntsampler->samprng_edgescale = tntsampler->samprng_range/tntsampler->stats[0];
								do {
									tntsampler->edge_index = (((tntsampler->samprng->type->get) (tntsampler->samprng->state))) / tntsampler->samprng_edgescale;
								} while(tntsampler->edge_index >= tntsampler->stats[0]);

								/* get the index of this edge */
								tntsampler->index = tntsampler->edges[tntsampler->edge_index];
								tntsampler->logproposal = tntsampler->logproposal_del[tntsampler->stats[0]];
							}
							else
							{
								/* pick a random dyad */
								do {
									tntsampler->index = (((tntsampler->samprng->type->get) (tntsampler->samprng->state))) / (tntsampler->samprng_dyadscale);
								} while(tntsampler->index >= tntsampler->graph->ndyads);

								/* check if this dyad is an edge */
								if(tntsampler->graph->DYADlist[tntsampler->index]->edge)
								{
									/* need to find the edge_index of this edge */
									for(tntsampler->edge_index = 0; tntsampler->edge_index < tntsampler->stats[0]; ++(tntsampler->edge_index))
									{
										if(tntsampler->edges[tntsampler->edge_index] == tntsampler->index) {break;}
									}
									tntsampler->logproposal = tntsampler->logproposal_del[tntsampler->stats[0]];
								}
								else
								{
									tntsampler->logproposal = tntsampler->logproposal_add[tntsampler->stats[0]];	/* adding */
								}
							}
							break;

						case 3:
							/* Pick a random edge - Since every dyad is an edge we can use rng_dyadscale rather than calculating rng_edgescale */
							do
							{
								tntsampler->edge_index = (((tntsampler->samprng->type->get) (tntsampler->samprng->state))) / (tntsampler->samprng_dyadscale);
							}
							while(tntsampler->edge_index >= tntsampler->graph->ndyads);

							tntsampler->logproposal = tntsampler->logproposal_del[tntsampler->graph->ndyads];
							break;

						default:
							/* should not happen but have a warning ready regardless */
							fprintf(stderr, "ERROR: tnt_case invalid, case %d\n", tntsampler->tnt_case);
					}

					/* set the changestats for the selected dyad */
					tntsampler->changestats[0] = get_edges_change(tntsampler->graph, tntsampler->index);
					tntsampler->changestats[1] = get_stars_change(tntsampler->graph, tntsampler->index);

					/* get the log acceptance probability logP(new_graph) - logP(old_graph) */
					tntsampler->logalpha = tntsampler->logproposal + theta[0]*tntsampler->changestats[0] + theta[1]*tntsampler->changestats[1];

					/* do the update with log probability logalpha */
					if((tntsampler->logalpha >= 0.0) || (log(gsl_rng_uniform_pos(tntsampler->samprng)) < tntsampler->logalpha))
					{
						tnt_accept_change(tntsampler, tntsampler->index, tntsampler->edge_index);
					}
				}

			/* take draw */
				tntsampler->samp_stats[draws_taken][0] = tntsampler->stats[0];
				tntsampler->samp_stats[draws_taken][1] = tntsampler->stats[1];
				draws_taken++;
		}
	}

	void tnt_run2(SAMPLER2 *tntsampler, double *theta)
	{
		size_t draws_taken = 0;
		size_t k;

		/* run the burn in */
		for(k = 0; k < tntsampler->burnin; ++k)
		{
			tntsampler->tnt_case = (tntsampler->stats_copy[0] > 0) + (tntsampler->stats_copy[0] > 1) + (tntsampler->stats_copy[0] == tntsampler->graph_copy->ndyads);
			/*
				tnt_case description
				tnt_case = 0 -> E = 0
				tnt_case = 1 -> E = 1
				tnt_case = 2 -> 2 <= E <= N-1
				tnt_case = 3 -> E = N
			*/
			switch(tntsampler->tnt_case)
			{
				case 0:
					/* pick a random dyad */
					do {
	  				tntsampler->index = (((tntsampler->samprng->type->get) (tntsampler->samprng->state))) / (tntsampler->samprng_dyadscale);
	  			} while(tntsampler->index >= tntsampler->graph_copy->ndyads);

					tntsampler->logproposal = tntsampler->logproposal_add[0];
					break;

				case 1:
					/* decide whether to pick an edge or pick a dyad */
					if(gsl_rng_uniform(tntsampler->samprng) < tntsampler->p)
					{
						/* pick the only edge and delete */
						tntsampler->edge_index = 0;
						tntsampler->logproposal = tntsampler->logproposal_del[1];
						tntsampler->index = tntsampler->edges_copy[0];
					}
					else
					{
						/* pick a random dyad */
						do {
							tntsampler->index = (((tntsampler->samprng->type->get) (tntsampler->samprng->state))) / (tntsampler->samprng_dyadscale);
						} while(tntsampler->index >= tntsampler->graph_copy->ndyads);

						/* check if this dyad is an edge */
						if(tntsampler->graph_copy->DYADlist[tntsampler->index]->edge)
						{
							/* pick the only edge and delete */
							tntsampler->edge_index = 0;
							tntsampler->logproposal = tntsampler->logproposal_del[1];
						}
						else
						{
							tntsampler->logproposal = tntsampler->logproposal_add[1];	/* adding */
						}
					}
					break;

				case 2:
					/* decide whether to pick an edge or pick a dyad */
					if(gsl_rng_uniform(tntsampler->samprng) < tntsampler->p)
					{
						/* pick an edge and delete */
						tntsampler->samprng_edgescale = tntsampler->samprng_range/tntsampler->stats_copy[0];
						do {
							tntsampler->edge_index = (((tntsampler->samprng->type->get) (tntsampler->samprng->state))) / tntsampler->samprng_edgescale;
						} while(tntsampler->edge_index >= tntsampler->stats_copy[0]);

						/* get the index of this edge */
						tntsampler->index = tntsampler->edges_copy[tntsampler->edge_index];
						tntsampler->logproposal = tntsampler->logproposal_del[tntsampler->stats_copy[0]];
					}
					else
					{
						/* pick a random dyad */
						do {
							tntsampler->index = (((tntsampler->samprng->type->get) (tntsampler->samprng->state))) / (tntsampler->samprng_dyadscale);
						} while(tntsampler->index >= tntsampler->graph_copy->ndyads);

						/* check if this dyad is an edge */
						if(tntsampler->graph_copy->DYADlist[tntsampler->index]->edge)
						{
							/* need to find the edge_index of this edge */
							for(tntsampler->edge_index = 0; tntsampler->edge_index < tntsampler->stats_copy[0]; ++(tntsampler->edge_index))
							{
								if(tntsampler->edges_copy[tntsampler->edge_index] == tntsampler->index) {break;}
							}
							tntsampler->logproposal = tntsampler->logproposal_del[tntsampler->stats_copy[0]];
						}
						else
						{
							tntsampler->logproposal = tntsampler->logproposal_add[tntsampler->stats_copy[0]];	/* adding */	
						}
					}
					break;

				case 3:
					/* Pick a random edge - Since every dyad is an edge we can use rng_dyadscale rather than calculating rng_edgescale */
					do
	  			{
	  				tntsampler->edge_index = (((tntsampler->samprng->type->get) (tntsampler->samprng->state))) / (tntsampler->samprng_dyadscale);
	  			}
	  			while(tntsampler->edge_index >= tntsampler->graph_copy->ndyads);

					tntsampler->logproposal = tntsampler->logproposal_del[tntsampler->graph_copy->ndyads];
					tntsampler->index = tntsampler->edges_copy[tntsampler->edge_index];
					break;

				default:
					/* should not happen but have a warning ready regardless */
					fprintf(stderr, "ERROR: tnt_case invalid, case %d\n", tntsampler->tnt_case);
			}

			/* set the changestats for the selected dyad */
			tntsampler->changestats[0] = get_edges_change(tntsampler->graph_copy, tntsampler->index);
			tntsampler->changestats[1] = get_stars_change(tntsampler->graph_copy, tntsampler->index);

	  	/* get the log acceptance probability logP(new_graph) - logP(old_graph) */
	  	tntsampler->logalpha = tntsampler->logproposal + theta[0]*tntsampler->changestats[0] + theta[1]*tntsampler->changestats[1];

	  	/* do the update with log probability logalpha */
			if((tntsampler->logalpha >= 0.0) || (log(gsl_rng_uniform_pos(tntsampler->samprng)) < tntsampler->logalpha))
			{
				tnt_accept_change2(tntsampler, tntsampler->index, tntsampler->edge_index);
			}
		}

		while(draws_taken < tntsampler->ndraws)
		{
			/* run for gap + 1 */
			for(k = 0; k < tntsampler->gap+1; ++k)
			{
				tntsampler->tnt_case = 	(tntsampler->stats_copy[0] > 0) + (tntsampler->stats_copy[0] > 1) + (tntsampler->stats_copy[0] == tntsampler->graph_copy->ndyads);
				/*
					tnt_case description
					tnt_case = 0 -> E = 0
					tnt_case = 1 -> E = 1
					tnt_case = 2 -> 2 <= E <= N-1
					tnt_case = 3 -> E = N
				*/

				switch(tntsampler->tnt_case)
				{
					case 0:
						/* pick a random dyad */
						do {
							tntsampler->index = (((tntsampler->samprng->type->get) (tntsampler->samprng->state))) / (tntsampler->samprng_dyadscale);
						} while(tntsampler->index >= tntsampler->graph_copy->ndyads);

						tntsampler->logproposal = tntsampler->logproposal_add[0];
						break;

					case 1:
						/* decide whether to pick an edge or pick a dyad */
						if(gsl_rng_uniform(tntsampler->samprng) < tntsampler->p)
						{
							/* pick the only edge and delete */
							tntsampler->edge_index = 0;
							tntsampler->logproposal = tntsampler->logproposal_del[1];
							tntsampler->index = tntsampler->edges_copy[0];
						}
						else
						{
							/* pick a random dyad */
							do {
								tntsampler->index = (((tntsampler->samprng->type->get) (tntsampler->samprng->state))) / (tntsampler->samprng_dyadscale);
							} while(tntsampler->index >= tntsampler->graph_copy->ndyads);

							/* check if this dyad is an edge */
							if(tntsampler->graph_copy->DYADlist[tntsampler->index]->edge)
							{
								/* pick the only edge and delete */
								tntsampler->edge_index = 0;
								tntsampler->logproposal = tntsampler->logproposal_del[1];
							}
							else
							{
								tntsampler->logproposal = tntsampler->logproposal_add[1];	/* adding */
							}
						}
						break;

					case 2:
						/* decide whether to pick an edge or pick a dyad */
						if(gsl_rng_uniform(tntsampler->samprng) < tntsampler->p)
						{
							/* pick an edge and delete */
							tntsampler->samprng_edgescale = tntsampler->samprng_range/tntsampler->stats_copy[0];
							do {
								tntsampler->edge_index = (((tntsampler->samprng->type->get) (tntsampler->samprng->state))) / tntsampler->samprng_edgescale;
							} while(tntsampler->edge_index >= tntsampler->stats_copy[0]);

							/* get the index of this edge */
							tntsampler->index = tntsampler->edges_copy[tntsampler->edge_index];
							tntsampler->logproposal = tntsampler->logproposal_del[tntsampler->stats_copy[0]];
						}
						else
						{
							/* pick a random dyad */
							do {
								tntsampler->index = (((tntsampler->samprng->type->get) (tntsampler->samprng->state))) / (tntsampler->samprng_dyadscale);
							} while(tntsampler->index >= tntsampler->graph_copy->ndyads);

							/* check if this dyad is an edge */
							if(tntsampler->graph_copy->DYADlist[tntsampler->index]->edge)
							{
								/* need to find the edge_index of this edge */
								for(tntsampler->edge_index = 0; tntsampler->edge_index < tntsampler->stats_copy[0]; ++(tntsampler->edge_index))
								{
									if(tntsampler->edges_copy[tntsampler->edge_index] == tntsampler->index) {break;}
								}
								tntsampler->logproposal = tntsampler->logproposal_del[tntsampler->stats_copy[0]];
							}
							else
							{
								tntsampler->logproposal = tntsampler->logproposal_add[tntsampler->stats_copy[0]];	/* adding */
							}
						}
						break;

					case 3:
						/* Pick a random edge - Since every dyad is an edge we can use rng_dyadscale rather than calculating rng_edgescale */
						do
						{
							tntsampler->edge_index = (((tntsampler->samprng->type->get) (tntsampler->samprng->state))) / (tntsampler->samprng_dyadscale);
						} while(tntsampler->edge_index >= tntsampler->graph_copy->ndyads);

						tntsampler->logproposal = tntsampler->logproposal_del[tntsampler->graph_copy->ndyads];
						tntsampler->index = tntsampler->edges_copy[tntsampler->edge_index];
						break;

					default:
						/* should not happen but have a warning ready regardless */
						fprintf(stderr, "ERROR: tnt_case invalid, case %d\n", tntsampler->tnt_case);
					}

					/* set the changestats for the selected dyad */
					tntsampler->changestats[0] = get_edges_change(tntsampler->graph_copy, tntsampler->index);
					tntsampler->changestats[1] = get_stars_change(tntsampler->graph_copy, tntsampler->index);

					/* get the log acceptance probability logP(new_graph) - logP(old_graph) */
					tntsampler->logalpha = tntsampler->logproposal + theta[0]*tntsampler->changestats[0] + theta[1]*tntsampler->changestats[1];

					/* do the update with log probability logalpha */
					if((tntsampler->logalpha >= 0.0) || (log(gsl_rng_uniform_pos(tntsampler->samprng)) < tntsampler->logalpha))
					{
						tnt_accept_change2(tntsampler, tntsampler->index, tntsampler->edge_index);
					}

					//if(tntsampler->stats_copy[1] != get_stars(tntsampler->graph_copy, 2)) printf("ERROR! %zu %zu\n", tntsampler->stats_copy[1], get_stars(tntsampler->graph_copy, 2));
				}

			/* take draw */
				tntsampler->samp_stats[draws_taken][0] = tntsampler->stats_copy[0];
				tntsampler->samp_stats[draws_taken][1] = tntsampler->stats_copy[1];
				draws_taken++;

		}/* end of while loop */

		return;
	}

/*
	SAMPLER OBJECT - DESTROY - This funciton frees a sampler
*/
	void tnt_destroy(SAMPLER *tntsampler)
	{
		free(tntsampler);
		return;
	}

	void tnt_destroy2(SAMPLER2 *tntsampler)
	{
		free(tntsampler);
		return;
	}

/* 
	get_dyad - gets the value of a dyad (edge=1,empty=0) 
*/
	int get_dyad(GRAPH *graph, size_t i, size_t j)
	{
		if(i > j) {
			/* swap */
			size_t temp = i;
			i = j;
			j = temp;
		}

		return graph->DYADlist[graph->offset[i] + j - i - 1]->edge; /* the -1 cannot be removed since offset is unsigned int */
	}

/*
	get_stars - gets the number of k stars in a graph
*/
	size_t get_stars(GRAPH *graph, size_t k)
	{
		if(graph->status!=0)
		{
			fprintf(stderr, "ERROR: graph status %d\n", graph->status);
			return 0;
		}
		size_t stars = 0;
		size_t i;
		for(i = 0; i < graph->nnodes; ++i)
		{
			if(graph->degree[i] > (k-1))
				stars += gsl_sf_choose(graph->degree[i],k);
		}

		return stars;
	}

/*
	get_edges_change - gets the change in the number of edges by toggling dyad at index
*/
	int get_edges_change(GRAPH *graph, size_t index)
	{
		return ((graph->DYADlist[index]->edge) ? -1 : 1);
	}

/*
	get_stars_change - gets the change in the number of stars by toggling dyad at index
*/
	int get_stars_change(GRAPH *graph, size_t index)
	{
		return ((graph->DYADlist[index]->edge) ? (2 - graph->degree[graph->DYADlist[index]->i] - graph->degree[graph->DYADlist[index]->j]) : (graph->degree[graph->DYADlist[index]->i] + graph->degree[graph->DYADlist[index]->j]));
	}

/*
	tnt_accept_change - makes all adjustments to accept the change by the TNT sampler
*/
	void tnt_accept_change(SAMPLER *tntsampler, size_t index, size_t edge_index)
	{
		size_t i = tntsampler->graph->DYADlist[index]->i;
		size_t j = tntsampler->graph->DYADlist[index]->j;

		tntsampler->graph->DYADlist[index]->edge = 1 - tntsampler->graph->DYADlist[index]->edge;				/* toggle the edge */

		if(tntsampler->graph->DYADlist[index]->edge) {
			/* EDGE JUST ADDED */
			/* Place edge in edges vector, at the end */
			tntsampler->edges[tntsampler->stats[0]] = index;
			tntsampler->graph->degree[i]++;
			tntsampler->graph->degree[j]++;
		}
		else {
			/* EDGE JUST REMOVED */
			/* remove the edge from the edges vector */
			for(size_t l = edge_index; l < (tntsampler->stats[0] - 1); ++l)
				tntsampler->edges[l] = tntsampler->edges[l+1];
			tntsampler->graph->degree[i]--;
			tntsampler->graph->degree[j]--;
		}

		tntsampler->stats[0] += tntsampler->changestats[0];		/* adjust number of edges */
		tntsampler->stats[1] += tntsampler->changestats[1];		/* adjust number of 2stars */

		return;
	}

	void tnt_accept_change2(SAMPLER2 *tntsampler, size_t index, size_t edge_index)
	{
		tntsampler->graph_copy->DYADlist[index]->edge = 1 - tntsampler->graph_copy->DYADlist[index]->edge;

		if(tntsampler->graph_copy->DYADlist[index]->edge) {
			/* EDGE JUST ADDED */
			/* Place edge in edges vector, at the end */
			tntsampler->edges_copy[tntsampler->stats_copy[0]] = index;
			(tntsampler->graph_copy->degree[tntsampler->graph_copy->DYADlist[index]->i])++;
			(tntsampler->graph_copy->degree[tntsampler->graph_copy->DYADlist[index]->j])++;
			(tntsampler->stats_copy[0])++;
		}
		else {
			/* EDGE JUST REMOVED */
			/* remove the edge from the edges vector */

			/* old method */
			//for(size_t l = edge_index; l < (tntsampler->stats_copy[0] - 1); ++l)
			//	tntsampler->edges_copy[l] = tntsampler->edges_copy[l+1];

			/* new method - monday 03/10/16 - order doesn't matter so just take last element and move it into index */
			tntsampler->edges_copy[edge_index] = tntsampler->edges_copy[tntsampler->stats_copy[0]-1];

			(tntsampler->graph_copy->degree[tntsampler->graph_copy->DYADlist[index]->i])--;
			(tntsampler->graph_copy->degree[tntsampler->graph_copy->DYADlist[index]->j])--;
			(tntsampler->stats_copy[0])--;
		}

		tntsampler->stats_copy[1] += tntsampler->changestats[1];		/* adjust number of 2stars */

		return;
	}
