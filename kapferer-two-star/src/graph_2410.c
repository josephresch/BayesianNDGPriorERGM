/*
	graph_2410.c
	============

	Two layers:

	(1) GRAPH — an undirected simple graph stored as a flat, index-addressable
	    array of DYADs (one per unordered pair i<j, total C(n,2) = ndyads).
	    Random access to dyad (i,j) is O(1) via an offset table:
	        DYADlist[offset[i] + j - i - 1]   (for i<j; swap if i>j)
	    This layout makes uniform dyad proposals, change-stat loops, and
	    incremental edge/degree updates fast.

	(2) SAMPLER2 — a Metropolis-Hastings sampler with the Tie-No-Tie (TNT)
	    proposal, fitting the fixed 3-parameter "esg" ERGM
	        stats = (edges, kstar2, gwesp(α=0.25)).
	    The sampler maintains a pristine copy (graph_orig / stats_orig /
	    edges_orig / nedges_orig) and a mutable copy (…_copy) that the MH
	    chain walks on. tnt_restore() rewinds copy→orig in O(n^2) so an outer
	    driver (HMC / exchange) can evaluate many θ proposals from the same
	    starting graph without re-duplicating.

	The "esg" model order is canonical throughout: stats[0]=edges,
	stats[1]=kstar2, stats[2]=gwesp. α is the fixed macro ALPHA_GWESP.
*/

#include "graph_2410.h"
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_sf_gamma.h>		/* for the n choose m function */

/*
	loadGRAPH — read an undirected simple graph from a plaintext file.

	File format (format tag == 1):
	    line 1:  "format <int>"        (must be 1)
	    line 2:  "nnodes <size_t>"     (number of vertices n)
	    lines 3…n+2: n rows of n whitespace-separated 0/1 entries,
	                  forming the full adjacency matrix. Only the strict
	                  upper triangle (j>i) is read; the lower triangle is
	                  ignored (graph is assumed symmetric).

	Builds: offset[], degree[], DYADlist[] (one DYAD per i<j pair), and
	maintains nedges / per-node degree as 1-entries are encountered.
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

		/* offset[i] = starting index in DYADlist of row-i dyads (i,i+1),(i,i+2),…
		   Row 0 spans (n-1) dyads, row 1 spans (n-2), …, so the recursion
		       offset[0] = 0,   offset[i] = offset[i-1] + (n-i)
		   places dyad (i,j) at DYADlist[offset[i] + j - i - 1]. Only n-1 rows
		   need an offset (the last row i=n-1 has no j>i). */
		graph->offset = malloc((nnodes-1)*sizeof(size_t));
		graph->offset[0] = 0;
		for(i = 1; i < (nnodes-1); ++i)
			graph->offset[i] = graph->offset[i-1] + (nnodes-i);

		/* per-node degree, incremented as edges are read */
		graph->degree = calloc(nnodes, sizeof(size_t));

		if(format != 1)
		{
			fprintf(stderr, "ERROR: graph format not supported!\n");
			fclose(graph_file);
			free(graph);
			return NULL;
		}

		/* read in graph — one DYAD struct per unordered pair i<j */
		graph->DYADlist = malloc(graph->ndyads*sizeof(DYAD *));
		index = 0;
		for(i = 0; i < graph->nnodes; ++i)
		{
			if(fgets(line, sizeof(line), graph_file)!=NULL)
			{
				for(j = i+1; j < graph->nnodes; ++j)
				{
					/* Row i is a whitespace-separated 0/1 vector of length n.
					   Each entry occupies 2 chars ("0 " or "1 "), so column j
					   starts at byte offset 2*j. Only upper triangle j>i is read. */
					sscanf(&line[2*j], "%d", &val);
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
	dupGRAPH2 — deep-copy a GRAPH.

	Strategy: start with a struct-assignment shallow copy (picks up scalar fields
	like nnodes/nedges/ndyads/status for free), then overwrite the three
	pointer-valued fields (offset, degree, DYADlist) with fresh allocations so
	the two graphs share no heap memory. Used once at tnt_attach2 time to make
	graph_copy, the mutable working graph the MH chain walks on.
*/
	GRAPH *dupGRAPH2(GRAPH *graph)
	{
		GRAPH *graphcopy = malloc(sizeof(GRAPH));
		size_t i, index;

		*graphcopy = *graph; /* shallow: copies all scalars AND the pointer
		                        values; we immediately overwrite the pointers
		                        below so no aliasing remains. */

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
	tnt_create2 — allocate an unattached SAMPLER2.

	Only scalar fields (RNG handle, schedule) are initialised here; anything
	that depends on the graph (stats vectors, logproposal tables, edges_copy,
	samp_stats matrix) is deferred to tnt_attach2 once a GRAPH is available.
	Split in two so a driver can reuse one SAMPLER2 across multiple graphs
	if it ever needs to.
*/
	SAMPLER2 *tnt_create2(size_t burnin, size_t ndraws, size_t gap, gsl_rng *samprng)
	{
		SAMPLER2 *tntsampler = malloc(sizeof(SAMPLER2));

		tntsampler->graph_attached = 0;
		tntsampler->type = 3; /* 3 = TNT proposal (legacy 1=GIBBS, 2=MGIBBS no longer built) */

		/* Cache the RNG's integer range once. tnt_run2 avoids gsl_rng_uniform_int
		   for hot-path dyad picks; instead it divides a raw uint draw by a
		   precomputed scale (samprng_dyadscale) and rejection-samples. */
		tntsampler->samprng = samprng;
		tntsampler->samprng_range = samprng->type->max - samprng->type->min;

		tntsampler->burnin	= burnin;
		tntsampler->ndraws	= ndraws;
		tntsampler->gap			= gap;

		return tntsampler;
	}

/*
	tnt_attach2 — bind a GRAPH to the sampler and finish setup.

	Does four things:
	  (a) stash `graph` as graph_orig (pristine, never mutated) and create
	      graph_copy = dupGRAPH2(graph) as the mutable working graph;
	  (b) parse the model string (must be some permutation of "esg") into
	      a fixed-position model[] flag vector — the canonical storage
	      order is always [edges, kstar2, gwesp] regardless of input order;
	  (c) allocate + compute stats_orig/stats_copy (incl. real-valued
	      kstar2/gwesp), allocate changestats, build edges_orig/edges_copy
	      (flat list of dyad indices that are currently edges);
	  (d) precompute logproposal_add[E] / logproposal_del[E] tables for
	      the TNT proposal (see block below).

	Postcondition: ready for tnt_run2. graph_orig is aliased, not copied,
	so callers must keep it alive until tnt_detach2.
*/
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

		/* parse the model — canonical order: e, s, g */
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
						fprintf(stderr, "ERROR: kstar2 selected more than once\n");
					}
					else
					{
						tntsampler->model[1] = 1;
						tntsampler->nstats++;
					}
					break;
				case 'g':
					if(tntsampler->model[2])
					{
						fprintf(stderr, "ERROR: gwesp selected more than once\n");
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
		if(tntsampler->nstats != 3 || !tntsampler->model[0] || !tntsampler->model[1] || !tntsampler->model[2])
		{
			fprintf(stderr, "ERROR: this build requires model \"esg\" (edges + kstar2 + gwesp)\n");
		}

		/* Precomputed scale for fast uniform-dyad draws: a raw RNG uint u in
		   [0, samprng_range] maps to dyad index u/samprng_dyadscale, rejecting
		   the occasional over-range result. Avoids the % bias of naive
		   `rand() % ndyads`. */
		tntsampler->samprng_dyadscale	= tntsampler->samprng_range/graph->ndyads;

		/* Three-stat model: each is a double. Integer edge count is also mirrored
		   as size_t in nedges_orig/copy below (needed to index logproposal_*[E]). */
		tntsampler->stats_orig = malloc(tntsampler->nstats*sizeof(double));
		tntsampler->stats_copy = malloc(tntsampler->nstats*sizeof(double));
		tntsampler->changestats = calloc(tntsampler->nstats, sizeof(double));

		/* Canonical order: [edges, kstar2, gwesp]. edges is exact; kstar2 is
		   O(n); gwesp is O(n * ndyads). All done ONCE here at attach.
		   Afterwards, tnt_run2 only ever adjusts stats_copy[] by changestats[]. */
		tntsampler->stats_orig[0] = tntsampler->stats_copy[0] = (double)graph->nedges;
		tntsampler->stats_orig[1] = tntsampler->stats_copy[1] = get_twostars(graph);
		tntsampler->stats_orig[2] = tntsampler->stats_copy[2] = get_gwesp(graph, ALPHA_GWESP);

		/* size_t mirror of stats_copy[0] kept in sync for array-indexing uses
		   (e.g. logproposal_del[ne]) where the double stats vector is unsuitable. */
		tntsampler->nedges_orig = tntsampler->nedges_copy = graph->nedges;

		/* edges_orig / edges_copy: flat list of dyad indices currently carrying
		   an edge. Used for O(1) uniform-random-edge draws in the TNT "tie"
		   case (see tnt_run2). Only the first nedges slots are live; the rest
		   of the buffer is scratch space that grows/shrinks via swap-with-last. */
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

		/* --- Log-proposal tables -------------------------------------------
		   TNT proposal: with prob p pick a uniformly random EXISTING edge
		   (always proposes a deletion); with prob q=1-p pick a uniformly
		   random DYAD (proposes an add if empty, delete if edge).

		   For each possible current edge count E, we precompute:
		     logproposal_add[E] = log q(E -> E+1) / q(E+1 -> E)   [add proposal]
		     logproposal_del[E] = log q(E -> E-1) / q(E-1 -> E)   [del proposal]
		   i.e. the log proposal ratio (reverse / forward) as it enters MH
		   acceptance. Boundary regimes E=0,1,N-1,N need separate formulas
		   because the "pick an edge" step collapses or is unavailable.
		   Interior formulas derived from mixing-two-kernels algebra:
		     q(add,  E -> E+1) = q / ndyads
		     q(del,  E -> E-1) = p/E + q/ndyads
		   Canceling the common q/ndyads gives the closed form below. */
		tntsampler->p = p;
		q = 1.0 - p;
		odds = p/q;
		Nodds = (graph->ndyads)*odds;

		tntsampler->logproposal_add = malloc((graph->ndyads+1)*sizeof(double));
		tntsampler->logproposal_del = malloc((graph->ndyads+1)*sizeof(double));

		/* E=0: only ADD is possible. Forward q(0->1)=q/N, reverse q(1->0)=p+q/N. */
		tntsampler->logproposal_add[0] = log(graph->ndyads*p + q);
		tntsampler->logproposal_del[0] = 0; /* unused (cannot delete from empty graph) */

		/* E=1: reverse is the E=0 case above, so del ratio is the negation. */
		tntsampler->logproposal_del[1] = -log(graph->ndyads*p + q);

		/* E=N-1: reverse move from full graph — see E=N formula below. */
		tntsampler->logproposal_add[graph->ndyads-1] = -log(q);

		/* E=N: only DEL is possible. Forward q(N->N-1)=p/N + q/N; reverse
		   q(N-1->N)=q/N. Ratio log = log(q) after cancellation. */
		tntsampler->logproposal_del[graph->ndyads] = log(q);
		tntsampler->logproposal_add[graph->ndyads] = 0; /* unused */

		/* Generic interior ADD: E -> E+1. Reverse proposes E+1 -> E via either
		   uniform-edge (prob p, picks among E+1 edges) or uniform-dyad (prob q).
		   Algebra collapses to log((N*odds + E + 1)/(E + 1)). */
		for(i = 1; i < graph->ndyads-1; ++i)
		{
			tntsampler->logproposal_add[i] = log(Nodds + i + 1.0) - log(i+1.0);
		}

		/* Generic interior DEL: E -> E-1, mirror of the add case. */
		for(i = 2; i < graph->ndyads; ++i)
		{
			tntsampler->logproposal_del[i] = log(i) - log(Nodds + i);
		}

		/* Pre-allocated draws buffer. tnt_run2 writes one row per post-burnin,
		   post-gap snapshot; rows are stats_copy at that moment. */
		tntsampler->samp_stats = malloc(tntsampler->ndraws*sizeof(double *));
		for(size_t i = 0; i < tntsampler->ndraws; ++i)
			tntsampler->samp_stats[i] = malloc(tntsampler->nstats*sizeof(double));

		return;
	}

/*
	tnt_detach2 — release everything tnt_attach2 allocated.

	graph_orig is only dropped (NULL'd) here — it was never owned by the
	sampler; the driver that called loadGRAPH is responsible for destroying
	it. graph_copy WAS allocated here (via dupGRAPH2), so it's freed.
*/
	void tnt_detach2(SAMPLER2 *tntsampler)
	{
		size_t i;
		tntsampler->graph_orig = NULL;
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
	tnt_restore — rewind all _copy state to _orig.

	Called by the outer driver (hmc.c / exchange_ergm.c) between proposals so
	each θ candidate is evaluated starting from the same observed-graph state.
	Cheaper than re-dup'ing the graph: we reuse the allocations in graph_copy
	and just overwrite fields.

	offset[] is not copied — it's a pure function of (nnodes, ndyads), which
	are immutable across a run, so graph_copy->offset already matches.
*/
	void tnt_restore(SAMPLER2 *tntsampler)
	{
		size_t k;

		/* edges_copy: only the first nedges slots hold live data; the rest is
		   stale scratch from previous runs and can be left as garbage. */
		for(k = 0; k < tntsampler->graph_orig->nedges; ++k)
		{
			tntsampler->edges_copy[k] = tntsampler->edges_orig[k];
		}
		for(k = 0; k < tntsampler->nstats; ++k)
		{
			tntsampler->stats_copy[k] = tntsampler->stats_orig[k];
		}
		tntsampler->nedges_copy = tntsampler->nedges_orig;

		/* graph_copy->nedges is read by tnt_run2 (to pick a TNT case) and by
		   tnt_accept_change2 when growing edges_copy, so it must track too. */
		tntsampler->graph_copy->nedges = tntsampler->graph_orig->nedges;

		for(k = 0; k < tntsampler->graph_orig->nnodes; ++k)
			tntsampler->graph_copy->degree[k] = tntsampler->graph_orig->degree[k];

		/* DYAD structs: {i,j,index} never change, but we copy them anyway
		   (cheap, keeps the code uniform). `edge` is the one that actually
		   gets toggled during MH and must be rewound. */
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
	tnt_run2 — the TNT Metropolis-Hastings inner loop.

	Runs for `burnin` iterations (no draw recorded), then repeatedly runs
	`gap+1` more iterations and snapshots stats_copy into samp_stats[]
	until `ndraws` rows are filled.

	Each iteration:
	    1. Classify by current edge count ne (tnt_case below).
	    2. Draw a proposal dyad index and set logproposal accordingly.
	    3. Compute incremental changestats[0..2] for that toggle.
	    4. logalpha = logproposal + theta . changestats.
	       (No log-ratio of the ERGM normaliser Z: this sampler generates
	       draws FROM the ERGM at θ; it does not compute a posterior ratio.)
	    5. Accept w.p. min(1, exp(logalpha)); on accept call tnt_accept_change2,
	       which toggles the dyad, updates degrees/nedges/edges_copy, and
	       folds changestats into stats_copy.

	Stats are maintained strictly incrementally after attach — they are
	never recomputed from scratch inside this loop.

	The four TNT cases (stored in tntsampler->tnt_case for debugging):
	    case 0: ne == 0        — empty graph, only ADD is possible
	    case 1: ne == 1        — one edge, edge-pool has a unique element
	    case 2: 1 < ne < N     — interior, full TNT mixture
	    case 3: ne == N        — complete graph, only DEL is possible
	The branchy expression
	    tnt_case = (ne>0) + (ne>1) + (ne==ndyads)
	turns the four disjoint regimes into a single switch discriminator.
*/
	void tnt_run2(SAMPLER2 *tntsampler, double *theta)
	{
		size_t draws_taken = 0;
		size_t k, s;
		size_t total_phase_iters;
		int phase_burnin = 1;   /* true for the first pass only */

		while(1)
		{
			/* First pass: burnin iterations. Subsequent passes: gap+1 iters
			   between draws (so gap=0 means every post-burnin iter is recorded). */
			total_phase_iters = phase_burnin ? tntsampler->burnin : (tntsampler->gap + 1);

			for(k = 0; k < total_phase_iters; ++k)
			{
				size_t ne = tntsampler->graph_copy->nedges;
				tntsampler->tnt_case = (ne > 0) + (ne > 1) + (ne == tntsampler->graph_copy->ndyads);

				switch(tntsampler->tnt_case)
				{
					case 0:
						/* Empty graph. Uniform dyad draw, always an ADD proposal.
						   The do/while is rejection sampling: a raw uint u is mapped
						   to u/samprng_dyadscale; if it overshoots ndyads we retry. */
						do {
							tntsampler->index = (((tntsampler->samprng->type->get) (tntsampler->samprng->state))) / (tntsampler->samprng_dyadscale);
						} while(tntsampler->index >= tntsampler->graph_copy->ndyads);

						tntsampler->logproposal = tntsampler->logproposal_add[0];
						break;

					case 1:
						/* Exactly one edge. With prob p take the "tie" branch: we
						   must propose deleting that one edge (edge_index = 0). */
						if(gsl_rng_uniform(tntsampler->samprng) < tntsampler->p)
						{
							tntsampler->edge_index = 0;
							tntsampler->logproposal = tntsampler->logproposal_del[1];
							tntsampler->index = tntsampler->edges_copy[0];
						}
						else
						{
							/* With prob q, uniform dyad. Could hit the sole edge
							   (proposes delete) or an empty dyad (proposes add). */
							do {
								tntsampler->index = (((tntsampler->samprng->type->get) (tntsampler->samprng->state))) / (tntsampler->samprng_dyadscale);
							} while(tntsampler->index >= tntsampler->graph_copy->ndyads);

							if(tntsampler->graph_copy->DYADlist[tntsampler->index]->edge)
							{
								tntsampler->edge_index = 0;
								tntsampler->logproposal = tntsampler->logproposal_del[1];
							}
							else
							{
								tntsampler->logproposal = tntsampler->logproposal_add[1];
							}
						}
						break;

					case 2:
						/* Interior regime — this is where the sampler spends almost
						   all of its time. Two branches mixing uniform-edge (prob p)
						   and uniform-dyad (prob q). */
						if(gsl_rng_uniform(tntsampler->samprng) < tntsampler->p)
						{
							/* "Tie" branch: pick a uniformly random edge from
							   edges_copy[0..ne). Same RNG-rescale-and-reject trick,
							   but with scale based on ne (not ndyads). */
							tntsampler->samprng_edgescale = tntsampler->samprng_range/ne;
							do {
								tntsampler->edge_index = (((tntsampler->samprng->type->get) (tntsampler->samprng->state))) / tntsampler->samprng_edgescale;
							} while(tntsampler->edge_index >= ne);

							tntsampler->index = tntsampler->edges_copy[tntsampler->edge_index];
							tntsampler->logproposal = tntsampler->logproposal_del[ne];
						}
						else
						{
							/* "No-tie" branch: uniform dyad. */
							do {
								tntsampler->index = (((tntsampler->samprng->type->get) (tntsampler->samprng->state))) / (tntsampler->samprng_dyadscale);
							} while(tntsampler->index >= tntsampler->graph_copy->ndyads);

							if(tntsampler->graph_copy->DYADlist[tntsampler->index]->edge)
							{
								/* We need edge_index so tnt_accept_change2 can
								   swap-remove it from edges_copy. Linear scan is
								   O(ne) but only runs in the ~q·(ne/N) fraction of
								   "no-tie happened to land on an edge" steps. */
								for(tntsampler->edge_index = 0; tntsampler->edge_index < ne; ++(tntsampler->edge_index))
								{
									if(tntsampler->edges_copy[tntsampler->edge_index] == tntsampler->index) {break;}
								}
								tntsampler->logproposal = tntsampler->logproposal_del[ne];
							}
							else
							{
								tntsampler->logproposal = tntsampler->logproposal_add[ne];
							}
						}
						break;

					case 3:
						/* Complete graph. Only DEL is possible. The rescale-draw here
						   uses dyadscale because ne == ndyads. edge_index doubles as
						   the dyad index (they're in bijection when ne==ndyads). */
						do {
							tntsampler->edge_index = (((tntsampler->samprng->type->get) (tntsampler->samprng->state))) / (tntsampler->samprng_dyadscale);
						} while(tntsampler->edge_index >= tntsampler->graph_copy->ndyads);

						tntsampler->logproposal = tntsampler->logproposal_del[tntsampler->graph_copy->ndyads];
						tntsampler->index = tntsampler->edges_copy[tntsampler->edge_index];
						break;

					default:
						fprintf(stderr, "ERROR: tnt_case invalid, case %d\n", tntsampler->tnt_case);
				}

				/* Incremental change-stats for the proposed toggle at dyad `index`.
				   All three functions read graph_copy — the PRE-toggle state —
				   which is exactly what the derivations assume. */
				tntsampler->changestats[0] = (double)get_edges_change(tntsampler->graph_copy, tntsampler->index);
				tntsampler->changestats[1] = get_twostars_change(tntsampler->graph_copy, tntsampler->index);
				tntsampler->changestats[2] = get_gwesp_change(tntsampler->graph_copy, tntsampler->index, ALPHA_GWESP);

				/* log MH acceptance:
				     log α = log [q(x'→x)/q(x→x')]  +  θ^T Δs(x,x')
				   The first summand is logproposal (precomputed table look-up
				   or a direct boundary value); the second is the log of the
				   ERGM ratio exp(θ·Δs), since Z(θ) cancels for a same-θ move. */
				tntsampler->logalpha = tntsampler->logproposal;
				for(s = 0; s < tntsampler->nstats; ++s)
					tntsampler->logalpha += theta[s] * tntsampler->changestats[s];

				/* Auto-accept when logalpha >= 0 saves one uniform draw + a log. */
				if((tntsampler->logalpha >= 0.0) || (log(gsl_rng_uniform_pos(tntsampler->samprng)) < tntsampler->logalpha))
				{
					tnt_accept_change2(tntsampler, tntsampler->index, tntsampler->edge_index);
				}
			}

			/* End of this phase: if it was burnin, flip the flag and continue to
			   the first inter-draw phase; otherwise record a draw. */
			if(phase_burnin)
			{
				phase_burnin = 0;
				continue;
			}

			for(s = 0; s < tntsampler->nstats; ++s)
				tntsampler->samp_stats[draws_taken][s] = tntsampler->stats_copy[s];
			draws_taken++;

			if(draws_taken >= tntsampler->ndraws) break;
		}

		return;
	}

/*
	SAMPLER OBJECT - DESTROY - This funciton frees a sampler
*/
	void tnt_destroy2(SAMPLER2 *tntsampler)
	{
		free(tntsampler);
		return;
	}

/*
	get_dyad — O(1) lookup of the edge indicator for dyad {i,j}.

	Canonicalises i<j by swapping, then uses offset[] to locate the DYAD:
	    index = offset[i] + j - i - 1
	The `-1` is because the first dyad of row i is (i, i+1), so the j-offset
	inside the row is j-i-1 (j=i+1 -> 0, j=i+2 -> 1, …). offset is size_t
	(unsigned), so rewriting the formula to remove the -1 risks underflow —
	keep it as-is.
*/
	int get_dyad(GRAPH *graph, size_t i, size_t j)
	{
		if(i > j) {
			size_t temp = i;
			i = j;
			j = temp;
		}

		return graph->DYADlist[graph->offset[i] + j - i - 1]->edge;
	}

/*
	get_edges_change — Δ(nedges) from toggling dyad `index`.
	If the dyad currently has an edge, toggling removes it (-1); else adds (+1).
*/
	int get_edges_change(GRAPH *graph, size_t index)
	{
		return ((graph->DYADlist[index]->edge) ? -1 : 1);
	}

/*
	f_gw — Hunter/Handcock weight function
	    f(L, alpha) = exp(alpha) * (1 - (1 - exp(-alpha))^L)
	f(0) = 0, f(1) = 1, f increasing in L for alpha > 0.
*/
	static double f_gw(double alpha, size_t L)
	{
		if(L == 0) return 0.0;
		return exp(alpha) * (1.0 - pow(1.0 - exp(-alpha), (double)L));
	}

/*
	get_SP — shared-partners count of dyad (i,j).
	    SP(i,j) = |{k : k != i,j, (i,k) edge AND (j,k) edge}|
	O(n) per call. No adjacency-list shortcut: we scan all k and call
	get_dyad twice per k. For the sizes this build targets (n ~ tens),
	the cache-friendly flat scan beats maintaining a per-node neighbour list.
*/
	size_t get_SP(GRAPH *graph, size_t i, size_t j)
	{
		size_t sp = 0, k;
		for(k = 0; k < graph->nnodes; ++k)
		{
			if(k == i || k == j) continue;
			if(get_dyad(graph, i, k) && get_dyad(graph, j, k)) sp++;
		}
		return sp;
	}

/*
	get_gwesp — Σ over EDGES (i,j) of f(SP(i,j), α).
	Full-graph O(n·ndyads). Only called at tnt_attach2 and in tests; the MH
	loop uses incremental change-stats instead.
*/
	double get_gwesp(GRAPH *graph, double alpha)
	{
		double total = 0.0;
		size_t idx;
		for(idx = 0; idx < graph->ndyads; ++idx)
		{
			DYAD *d = graph->DYADlist[idx];
			if(!d->edge) continue;
			total += f_gw(alpha, get_SP(graph, d->i, d->j));
		}
		return total;
	}

/*
	get_twostars — full-graph 2-star count: sum_i C(degree_i, 2).
	O(n). Called once at tnt_attach2.
*/
	double get_twostars(GRAPH *graph)
	{
		double total = 0.0;
		size_t i;
		for(i = 0; i < graph->nnodes; ++i)
		{
			double d = (double)graph->degree[i];
			total += d * (d - 1.0) / 2.0;
		}
		return total;
	}

/*
	get_twostars_change — change in 2-star count from toggling dyad at index.
	O(1). Uses pre-toggle degrees.

	Adding edge (a,b): each endpoint gains one degree, creating degree_a new
	2-stars through a and degree_b new 2-stars through b.
	    delta = degree_a + degree_b

	Removing edge (a,b): each endpoint loses one degree, destroying
	(degree_a - 1) 2-stars through a and (degree_b - 1) through b.
	    delta = -(degree_a - 1 + degree_b - 1)
*/
	double get_twostars_change(GRAPH *graph, size_t index)
	{
		DYAD *d = graph->DYADlist[index];
		size_t a = d->i, b = d->j;
		double da = (double)graph->degree[a];
		double db = (double)graph->degree[b];
		if(d->edge)
			return -(da - 1.0 + db - 1.0);
		else
			return da + db;
	}

/*
	get_gwesp_change — change in gwesp from toggling dyad at index.

	Derivation. Let (a,b) be the toggled dyad and delta in {+1,-1} the sign of
	the toggle (delta = +1 if we add the edge, -1 if we remove it). Let SP be
	the pre-toggle shared-partners map.

	    Delta gwesp =
	        delta * f(SP(a,b))
	      + sum_{k in CommonNbrs(a,b)} [f(SP(a,k) + delta) - f(SP(a,k))
	                                   + f(SP(b,k) + delta) - f(SP(b,k))]

	The first term captures the toggled dyad itself (contributes only as an edge).
	The sum: for each common neighbour k, both (a,k) and (b,k) are edges, so each
	contributes f of its pre-toggle SP to gwesp. Toggling (a,b) changes SP(a,k) by
	exactly delta iff b is currently (pre-toggle semantics) also a neighbour of k,
	which is equivalent to k being a common neighbour of a and b. Symmetric for (b,k).
*/
	double get_gwesp_change(GRAPH *graph, size_t index, double alpha)
	{
		DYAD *d = graph->DYADlist[index];
		size_t a = d->i, b = d->j, k;
		int delta = (d->edge) ? -1 : +1;
		double dg = 0.0;
		size_t sp_ab = get_SP(graph, a, b);

		/* Toggled dyad: if becoming an edge, gains f(sp_ab); if losing, drops f(sp_ab). */
		dg += (double)delta * f_gw(alpha, sp_ab);

		/* Common neighbours of a and b: each contributes to both (a,k) and (b,k) edge terms. */
		for(k = 0; k < graph->nnodes; ++k)
		{
			if(k == a || k == b) continue;
			if(!get_dyad(graph, a, k) || !get_dyad(graph, b, k)) continue;
			size_t sp_ak = get_SP(graph, a, k);
			size_t sp_bk = get_SP(graph, b, k);
			size_t sp_ak_new = (delta > 0) ? (sp_ak + 1) : (sp_ak - 1);
			size_t sp_bk_new = (delta > 0) ? (sp_bk + 1) : (sp_bk - 1);
			dg += f_gw(alpha, sp_ak_new) - f_gw(alpha, sp_ak);
			dg += f_gw(alpha, sp_bk_new) - f_gw(alpha, sp_bk);
		}

		return dg;
	}

/*
	tnt_accept_change2 — commit an accepted toggle to all of graph_copy's
	derived state. Four things are kept in sync:
	  (1) the DYAD's edge bit (flipped),
	  (2) edges_copy[] (append on add; swap-with-last on delete — order
	      is irrelevant since proposals only sample uniformly from it),
	  (3) degree[] for the two endpoints, and graph_copy->nedges,
	  (4) stats_copy[] by accumulating changestats[] computed before the
	      accept decision in tnt_run2.

	Called only on accepted proposals; rejected proposals leave everything
	untouched.
*/
	void tnt_accept_change2(SAMPLER2 *tntsampler, size_t index, size_t edge_index)
	{
		size_t k;

		/* Flip the edge bit: XOR with 1 would be equivalent. */
		tntsampler->graph_copy->DYADlist[index]->edge = 1 - tntsampler->graph_copy->DYADlist[index]->edge;

		if(tntsampler->graph_copy->DYADlist[index]->edge) {
			/* ADD: append to edges_copy at position nedges (the first free slot),
			   then bump endpoint degrees and the integer edge count. */
			tntsampler->edges_copy[tntsampler->graph_copy->nedges] = index;
			(tntsampler->graph_copy->degree[tntsampler->graph_copy->DYADlist[index]->i])++;
			(tntsampler->graph_copy->degree[tntsampler->graph_copy->DYADlist[index]->j])++;
			tntsampler->graph_copy->nedges++;
			tntsampler->nedges_copy = tntsampler->graph_copy->nedges;
		}
		else {
			/* DELETE: overwrite the edge-to-remove with the last live entry,
			   shrinking the live range by 1. O(1). edges_copy is an unordered
			   multiset-as-array, so any permutation is fine.
			   Note: if edge_index is already the last slot, we harmlessly copy
			   it onto itself. */
			tntsampler->edges_copy[edge_index] = tntsampler->edges_copy[tntsampler->graph_copy->nedges - 1];

			(tntsampler->graph_copy->degree[tntsampler->graph_copy->DYADlist[index]->i])--;
			(tntsampler->graph_copy->degree[tntsampler->graph_copy->DYADlist[index]->j])--;
			tntsampler->graph_copy->nedges--;
			tntsampler->nedges_copy = tntsampler->graph_copy->nedges;
		}

		/* Fold the incremental change-stats into the running totals. After this
		   line, stats_copy exactly equals what get_twostars/gwesp would return on
		   the new graph (up to float rounding — verified by test_changestats.c). */
		for(k = 0; k < tntsampler->nstats; ++k)
			tntsampler->stats_copy[k] += tntsampler->changestats[k];

		return;
	}
