/*
	hull.c
	======

	Empirical convex hull of observed integer sufficient-statistic points
	(s1, s2) in Z^2. See hull.h for the public interface.

	Why a hull at all?
	-----------------
	The NDG prior (see prior.c) penalises theta values that push the ERGM
	towards "degenerate" regions -- regions where the network's sufficient
	statistic g(Y) is extremely likely to sit at the boundary of its own
	lattice (e.g. the empty or complete graph). In the continuous
	parameter-space picture, "non-degenerate" means "g(Y) lies in the
	INTERIOR of the set of achievable (s1, s2) values".

	For a finite network with 2-dim statistics, the set of achievable
	(s1, s2) points is a finite subset of Z^2. Its CONVEX HULL in R^2
	is a polygon whose interior approximates the non-degenerate region.
	This module maintains that hull empirically: we don't know all the
	achievable points in advance (enumerating them would be cheating --
	it's the quantity we are trying to avoid), so instead we GROW a
	hull as new sufficient statistics are observed during HMC.

	Monotone growth
	---------------
	The hull only ever adds points: hull_add flips one cell of a "seen"
	bitmap; it never removes. The set of observed points is therefore
	monotonically increasing, and its convex hull is also monotonically
	increasing (never shrinks). This matters because the NDG estimator
	relies on the hull being a stable, consistent description of
	"non-degenerate" across leapfrog steps within the same HMC trajectory.

	Data structures at a glance
	---------------------------
	  seen[]      : a (s1_max+1) x (s2_max+1) bitmap, row-major. seen[i*stride+j] = 1
	                iff the integer point (i, j) has ever been passed to hull_add.
	                Serves BOTH as the set of observed points AND as the hash
	                that makes hull_add idempotent in O(1).

	  interior[]  : a parallel bitmap, same shape. interior[i*stride+j] = 1 iff
	                (i, j) is STRICTLY inside the current hull polygon (i.e.,
	                not on any edge and not equal to a vertex). Precomputed in
	                hull_rebuild so that hull_is_interior is O(1).

	  vert_s1[]/vert_s2[]
	              : the CCW-ordered vertex list of the current hull polygon,
	                length n_vertices. Computed by Andrew's monotone chain.

	  dirty       : 1 iff hull_add has been called since the last rebuild.
	                Both interior[] and vert_*[] are out of date when dirty.

	  n_observed  : |seen|, kept in sync with seen[] for cheap diagnostics.

	Asymptotic cost
	---------------
	Andrew's monotone chain is O(n log n) in general because it requires
	lex-sorted input. But we scan seen[] in row-major (s1, s2) order, which
	IS lex-sorted for free, so the sort cost vanishes and rebuild is O(n)
	in the number of observed points plus O((s1_max+1)*(s2_max+1)) for the
	interior-grid fill. For g7 (22 x 106 = 2332 cells, O(dozens) hull
	vertices, O(hundreds) observed points), a rebuild is sub-microsecond.

	Lazy rebuild
	------------
	Rebuilding after every hull_add would be wasteful when many adds
	happen between queries (typical: a whole pool of M samples is added
	back-to-back in prior_ndg_set_pool, then a single query spree at
	accept/reject). We flip a dirty bit on each add and defer rebuild
	until the NEXT query. This is always safe because nothing reads
	interior[] or the vertex list without going through a function that
	checks the dirty bit first.

	No ground-truth knowledge
	-------------------------
	This module never reads a precomputed "on-hull" indicator or an
	enumeration file. Its only knowledge is the stream of lattice points
	fed to hull_add. (Points can come from the HMC auxiliary sampler at
	runtime or from a warm-start file of pre-drawn samples -- either way,
	they are just integer (s1, s2) pairs.)
*/

#include "hull.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Constructors / destructors                                        */
/* ------------------------------------------------------------------ */

/*
	hull_create: allocate a HULL with the bounding box (0..s1_max) x (0..s2_max).

	The bitmaps are sized to the full lattice so seen[]/interior[] can be
	indexed directly by the integer statistic with no hashing. For the
	g7 example (n=7 nodes, edges + kstar2) the bounding box is 22 x 106
	= 2332 bytes per bitmap -- trivial. For larger networks the cost is
	still linear in the lattice area, not in the number of observed
	points, which is fine as long as s1_max*s2_max is not astronomical.

	Returns NULL on allocation failure. On partial failure (some allocs
	succeeded, others didn't) we free the partial object via hull_destroy
	and return NULL, so callers never get a half-initialised HULL.
*/
HULL *hull_create(int s1_max, int s2_max)
{
	if(s1_max < 0 || s2_max < 0) return NULL;

	HULL *h = calloc(1, sizeof(HULL));
	if(!h) return NULL;

	h->s1_max       = s1_max;
	h->s2_max       = s2_max;
	h->grid_stride  = s2_max + 1;   /* row-major stride: idx = s1*stride + s2 */

	size_t cells    = (size_t)(s1_max + 1) * (size_t)(s2_max + 1);
	h->seen         = calloc(cells, sizeof(unsigned char));   /* cleared -> 0 */
	h->interior     = calloc(cells, sizeof(unsigned char));   /* cleared -> 0 */

	/* Vertex list starts small and doubles as needed. 16 is enough for
	   almost every ERGM hull we ever see (typical counts: 4-20). */
	h->vert_capacity = 16;
	h->vert_s1      = malloc((size_t)h->vert_capacity * sizeof(int));
	h->vert_s2      = malloc((size_t)h->vert_capacity * sizeof(int));

	if(!h->seen || !h->interior || !h->vert_s1 || !h->vert_s2)
	{
		hull_destroy(h);
		return NULL;
	}

	h->n_vertices = 0;
	h->n_observed = 0;
	h->dirty      = 0;     /* nothing observed yet -> nothing to rebuild */
	return h;
}

void hull_destroy(HULL *h)
{
	if(!h) return;
	free(h->seen);
	free(h->interior);
	free(h->vert_s1);
	free(h->vert_s2);
	free(h);
}

/* ------------------------------------------------------------------ */
/*  Vertex buffer growth                                              */
/* ------------------------------------------------------------------ */

/*
	hull_grow_verts: ensure vert_s1[] / vert_s2[] can hold at least `need`
	entries. Doubles the capacity until it fits. Only updates
	vert_capacity when BOTH reallocs succeed so we never advertise more
	capacity than we actually own.

	Called from hull_rebuild; never called directly by public API.
*/
static void hull_grow_verts(HULL *h, int need)
{
	if(need <= h->vert_capacity) return;

	int newcap = h->vert_capacity;
	while(newcap < need) newcap *= 2;

	int *new_s1 = realloc(h->vert_s1, (size_t)newcap * sizeof(int));
	int *new_s2 = realloc(h->vert_s2, (size_t)newcap * sizeof(int));

	/* If either realloc failed, keep whatever succeeded and only update
	   vert_capacity when BOTH succeeded. This guarantees
	   vert_capacity never overstates real capacity, which is essential
	   for the caller's bounds-check loop below. */
	if(new_s1) h->vert_s1 = new_s1;
	if(new_s2) h->vert_s2 = new_s2;
	if(new_s1 && new_s2) h->vert_capacity = newcap;
}

/* ------------------------------------------------------------------ */
/*  Add a lattice point                                               */
/* ------------------------------------------------------------------ */

/*
	hull_add: record that the lattice point (s1, s2) has been observed.

	Semantics:
	  - Out-of-bounds points are silently ignored. This makes the caller
	    safe to feed raw sampler output even if the sampler briefly
	    deviates from the nominal lattice (which shouldn't happen, but
	    defensive is cheap).
	  - Already-seen points are a no-op: the bitmap membership test costs
	    one array read, and skipping the duplicate keeps n_observed a
	    true |unique set| count.
	  - A true first-time add flips the dirty bit. The next query of
	    interior or vertices will trigger a rebuild.

	Cost: O(1). No rebuild happens here; even if the caller adds a
	million points in a row, we do exactly one rebuild on the next query.
*/
void hull_add(HULL *h, int s1, int s2)
{
	if(!h) return;
	if(s1 < 0 || s1 > h->s1_max) return;
	if(s2 < 0 || s2 > h->s2_max) return;

	int idx = s1 * h->grid_stride + s2;
	if(h->seen[idx]) return;  /* already observed -- no-op, preserves n_observed */

	h->seen[idx] = 1;
	h->n_observed++;
	h->dirty = 1;             /* defer rebuild to the next query */
}

/* ------------------------------------------------------------------ */
/*  2D cross product (signed twice-area)                              */
/* ------------------------------------------------------------------ */
/*
	hull_cross: signed area of the parallelogram spanned by (a-o) and (b-o).

	Formula:
	    cross_oab = (a - o) x (b - o)
	              = (ax - ox)*(by - oy) - (ay - oy)*(bx - ox)

	Geometric interpretation for the turn o -> a -> b (three points in
	the plane, traversed in that order):

	     b          b
	      \ turn   turn /
	       \  CCW  CW  /
	        a          a           collinear:
	       /            \            o ---- a ---- b
	      o              o           cross == 0

	    cross > 0  : CCW ("left") turn at a.  Keep a -- it's a real corner.
	    cross == 0 : collinear. Drop a -- it's an interior point on the edge ob.
	                 We want only corners so the strict-interior test below
	                 doesn't accidentally put boundary lattice points inside.
	    cross < 0  : CW ("right") turn at a.  Drop a -- a bulge heading the
	                 wrong way, which means a would have been popped by a
	                 later point anyway.

	Why long arithmetic?
	    For g7 the largest coordinate is 105 and 2-stars can differ by 105,
	    so the product fits comfortably in a 32-bit int. But on larger
	    networks s1_max*s2_max can overflow int easily (e.g. a 100-node
	    graph has s2_max ~ 4.9e5, and the cross product can reach ~ 2.4e11).
	    `long` is at least 32 bits; we use it here so callers don't have
	    to worry about the network size.

	Equivalence:
	    (a - o) x (b - o) and (a - o) x (b - a) are the same number in
	    2D. The (b - o) form is marginally more symmetric; the choice is
	    otherwise immaterial.
*/
static long hull_cross(int ox, int oy, int ax, int ay, int bx, int by)
{
	return (long)(ax - ox) * (long)(by - oy)
	     - (long)(ay - oy) * (long)(bx - ox);
}

/* ------------------------------------------------------------------ */
/*  Rebuild: monotone chain + interior-grid fill                      */
/* ------------------------------------------------------------------ */
/*
	hull_rebuild: regenerate vert_s1[]/vert_s2[]/interior[] from seen[].

	This is the heart of the module. It runs in four phases:

	    Phase 1: clear interior[] and vert_*.
	    Phase 2: collect observed points into (xs, ys) in lex order.
	    Phase 3: Andrew's monotone chain to get CCW hull vertices.
	    Phase 4: for every lattice cell, test strict interiority and set interior[].

	Invariant on exit: dirty == 0 IFF the rebuild succeeded. Any alloc
	failure leaves dirty == 1 so the next query retries.

	Monotone chain sketch (phase 3)
	-------------------------------
	Given n points sorted lexicographically by (x, y), the convex hull
	polygon can be decomposed into two halves:

	    lower hull: the chain of vertices on the bottom of the polygon,
	                swept left-to-right;
	    upper hull: the chain of vertices on the top, swept right-to-left.

	For each pass we maintain a stack of candidate hull vertices and a
	simple rule: when considering a new point p, pop the top-of-stack
	vertex `a` if the turn `(prev, a, p)` is NOT a strict CCW turn
	(cross <= 0). The `<= 0` rather than `< 0` is critical: it also
	pops collinear vertices. Without it, three lattice points in a row
	on a hull edge would all become "vertices", and the interior test
	in phase 4 would wrongly classify a lattice point lying between
	them as interior of the degenerate triangle rather than on the edge.

	After both passes, the stack holds the polygon in CCW order, with
	the starting point appearing at both ends; we drop the duplicate.

	Picture:
	    inputs (lex order):
	      p0 (leftmost, ties broken by lower y)
	      p1 p2 ... pn-1 pn (rightmost, ties broken by higher y)

	    lower hull sweep (left -> right):
	      push p0; push p1; for i=2..n, while top two make a non-CCW
	      turn with p_i, pop; then push p_i. Result: a chain from p0
	      to the rightmost point that stays on the lower boundary.

	    upper hull sweep (right -> left, starting at pn-1 since pn is
	    already on the stack):
	      same rule, but we may not pop below the threshold t = k+1 or
	      we would eat into the lower hull's closing vertex.

	Interior fill (phase 4)
	-----------------------
	A point p is STRICTLY interior to a CCW polygon iff every directed
	edge sees p strictly on its left. In 2D cross-product form:

	    for every edge (v_i -> v_j) of the polygon,
	        cross( v_j - v_i, p - v_i ) > 0.

	If any cross is 0, p lies ON that edge (boundary, not interior).
	If any cross is < 0, p is outside.

	We iterate over the full bounding-box grid because the interior set
	is small relative to the grid (and both are small anyway). An
	alternative would be to scan-convert the polygon, but this direct
	test is simpler and only costs O(cells * n_vertices).

	Degenerate hulls (phase 4, fallthrough)
	---------------------------------------
	A polygon needs at least 3 non-collinear vertices to have any
	interior at all. For k < 3 (single point, or collinear points along
	one line) the interior is empty and interior[] stays all-zero.
*/
static void hull_rebuild(HULL *h)
{
	/* ---- Phase 1: clear previous state. ---- */
	size_t cells = (size_t)(h->s1_max + 1) * (size_t)(h->s2_max + 1);
	memset(h->interior, 0, cells);
	h->n_vertices = 0;

	if(h->n_observed == 0)
	{
		/* Nothing has been added yet -> hull is empty, interior is
		   empty. Mark clean and return. */
		h->dirty = 0;
		return;
	}

	/* ---- Phase 2: collect observed points in lex order. ---- */
	/* Row-major iteration over seen[] yields points sorted primarily
	   by s1, secondarily by s2 -- exactly what the monotone chain
	   expects. No explicit sort step is needed. */
	int n   = (int)h->n_observed;
	int *xs = malloc((size_t)n * sizeof(int));
	int *ys = malloc((size_t)n * sizeof(int));
	if(!xs || !ys)
	{
		free(xs); free(ys);
		/* Leave dirty=1 so the next query retries. Any stale
		   interior[] is already cleared, so queries that trigger this
		   path will just return 0 -- a safe degraded behaviour. */
		return;
	}

	int cnt = 0;
	int s1, s2, i, j;
	for(s1 = 0; s1 <= h->s1_max; ++s1)
	{
		for(s2 = 0; s2 <= h->s2_max; ++s2)
		{
			if(h->seen[s1 * h->grid_stride + s2])
			{
				xs[cnt] = s1;
				ys[cnt] = s2;
				cnt++;
			}
		}
	}
	/* Invariant: cnt == n (n_observed is kept in sync with seen[] by hull_add). */

	/* Degenerate: a single observed point. No hull polygon, just a
	   vertex. Record it and we're done. */
	if(n == 1)
	{
		hull_grow_verts(h, 1);
		h->vert_s1[0] = xs[0];
		h->vert_s2[0] = ys[0];
		h->n_vertices = 1;
		free(xs); free(ys);
		h->dirty = 0;
		return;
	}

	/* ---- Phase 3: Andrew's monotone chain. ---- */
	/* Scratch for the combined (lower + upper) chain. 2*n is a
	   comfortable upper bound; the true hull has at most n vertices. */
	int *hx = malloc((size_t)(2 * n) * sizeof(int));
	int *hy = malloc((size_t)(2 * n) * sizeof(int));
	if(!hx || !hy)
	{
		free(hx); free(hy); free(xs); free(ys);
		return;   /* leave dirty=1, retry on next query */
	}

	int k = 0;   /* stack pointer: number of vertices currently on the chain */

	/* Lower hull: sweep left to right. */
	for(i = 0; i < n; ++i)
	{
		/* Pop the top while the top two vertices plus the new point
		   form a non-CCW turn. This removes both right turns and
		   collinear points; the survivors are the true "corners". */
		while(k >= 2)
		{
			long cp = hull_cross(
				hx[k-2], hy[k-2],     /* previous vertex */
				hx[k-1], hy[k-1],     /* current top of stack */
				xs[i],   ys[i]);      /* candidate new vertex */
			if(cp <= 0) k--;   /* top is not a corner -> pop */
			else        break; /* strict left turn -> keep top */
		}
		hx[k] = xs[i];
		hy[k] = ys[i];
		k++;
	}

	/* Upper hull: sweep right to left, starting from the second-to-last
	   point (the rightmost point is already the closing vertex of the
	   lower hull).

	   The `threshold t` is a hard floor for the pop loop: we must NOT
	   pop below it or we'd eat into the lower hull's closing vertex.
	   Concretely, after the lower-hull sweep the chain is
	       [p0, ..., p_rightmost]   (k vertices)
	   and we want the upper hull to APPEND on top of that, not mutate
	   it. Setting t = k + 1 means the upper-hull pop loop stops as
	   soon as it would touch the lower hull's last vertex. */
	{
		int t = k + 1;
		for(i = n - 2; i >= 0; --i)
		{
			while(k >= t)
			{
				long cp = hull_cross(
					hx[k-2], hy[k-2],
					hx[k-1], hy[k-1],
					xs[i],   ys[i]);
				if(cp <= 0) k--;
				else        break;
			}
			hx[k] = xs[i];
			hy[k] = ys[i];
			k++;
		}
	}

	/* The upper-hull sweep adds p0 back at the end (it closes the
	   polygon). Drop the duplicate so the vertex list is a minimal
	   cycle. */
	k--;

	/* Commit the k CCW-ordered vertices into the persistent struct. */
	hull_grow_verts(h, k);
	for(i = 0; i < k; ++i)
	{
		h->vert_s1[i] = hx[i];
		h->vert_s2[i] = hy[i];
	}
	h->n_vertices = k;

	free(hx); free(hy); free(xs); free(ys);

	/* ---- Phase 4: fill the strict-interior grid. ---- */
	/* For a CCW polygon, a point p is strictly interior iff the cross
	   product of every directed edge (v_i -> v_j) with (p - v_i) is
	   strictly positive. The short-circuit break on any cp <= 0 means
	   the per-cell cost is typically far less than n_vertices
	   multiplications: failures are usually detected at the first or
	   second edge. */
	if(k >= 3)
	{
		for(s1 = 0; s1 <= h->s1_max; ++s1)
		{
			for(s2 = 0; s2 <= h->s2_max; ++s2)
			{
				int inside = 1;
				for(i = 0; i < k; ++i)
				{
					j = (i + 1 == k) ? 0 : (i + 1);   /* next vertex, wrapping */
					/* Inlined cross product:
					       cp = (v_j - v_i) x ( (s1,s2) - v_i ).
					   cp > 0  -> strict left of this edge, keep going.
					   cp == 0 -> on the edge, NOT interior (boundary).
					   cp < 0  -> right of this edge, point is outside. */
					long cp =
						  (long)(h->vert_s1[j] - h->vert_s1[i]) * (long)(s2 - h->vert_s2[i])
						- (long)(h->vert_s2[j] - h->vert_s2[i]) * (long)(s1 - h->vert_s1[i]);
					if(cp <= 0) { inside = 0; break; }
				}
				if(inside)
					h->interior[s1 * h->grid_stride + s2] = 1;
			}
		}
	}
	/* If k < 3 the polygon is degenerate (a point or a segment) and has
	   empty strict interior -- interior[] stays all-zero from the
	   phase-1 memset. */

	h->dirty = 0;
}

/* ------------------------------------------------------------------ */
/*  Queries                                                           */
/* ------------------------------------------------------------------ */

/*
	hull_is_interior: strict interior membership test.

	Why the seen[] gate?
	    An unseen lattice point returns 0 even if the polygon
	    geometrically contains it. This matters for the NDG prior,
	    where interior membership is applied ONLY to pool samples
	    that have been fed through hull_add (and therefore are in
	    seen[]). We never query a point we haven't added -- but the
	    gate is a cheap insurance policy against caller bugs.

	Cost: O(1) in the fast path (cached interior[] lookup). If the
	hull is dirty we pay one O(n + cells) rebuild, amortised across
	all subsequent queries until the next hull_add.
*/
int hull_is_interior(HULL *h, int s1, int s2)
{
	if(!h) return 0;
	if(s1 < 0 || s1 > h->s1_max) return 0;
	if(s2 < 0 || s2 > h->s2_max) return 0;

	int idx = s1 * h->grid_stride + s2;
	if(!h->seen[idx]) return 0;  /* never observed -- safe "unknown" answer */

	if(h->dirty) hull_rebuild(h);
	return h->interior[idx] ? 1 : 0;
}

size_t hull_n_observed(const HULL *h)
{
	return h ? h->n_observed : 0;
}

/* Forces a rebuild so the returned count reflects the current seen[]
   set, not a stale previous rebuild. */
int hull_n_vertices(HULL *h)
{
	if(!h) return 0;
	if(h->dirty) hull_rebuild(h);
	return h->n_vertices;
}

/*
	hull_dump: write a human-readable snapshot of the hull to `path`.
	Used by test_prior.c to produce /tmp/test_prior_dump.txt for the
	R-side cross-check. Format:

	    # hull dump: <n_observed> observed, <n_vertices> vertices, bbox=[0,s1_max]x[0,s2_max]
	    # section: observed
	    s1 s2
	    <integer pairs, one per line, in lex order>
	    # section: vertices (CCW)
	    s1 s2
	    <integer pairs, one per line, in CCW order>
*/
void hull_dump(HULL *h, const char *path)
{
	if(!h || !path) return;
	if(h->dirty) hull_rebuild(h);

	FILE *f = fopen(path, "w");
	if(!f) return;

	fprintf(f,
		"# hull dump: %zu observed, %d vertices, bbox=[0,%d]x[0,%d]\n",
		h->n_observed, h->n_vertices, h->s1_max, h->s2_max);

	fprintf(f, "# section: observed\n");
	fprintf(f, "s1 s2\n");
	int s1, s2, i;
	for(s1 = 0; s1 <= h->s1_max; ++s1)
	{
		for(s2 = 0; s2 <= h->s2_max; ++s2)
		{
			if(h->seen[s1 * h->grid_stride + s2])
				fprintf(f, "%d %d\n", s1, s2);
		}
	}

	fprintf(f, "# section: vertices (CCW)\n");
	fprintf(f, "s1 s2\n");
	for(i = 0; i < h->n_vertices; ++i)
		fprintf(f, "%d %d\n", h->vert_s1[i], h->vert_s2[i]);

	fclose(f);
}
