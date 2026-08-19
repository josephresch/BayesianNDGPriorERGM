#ifndef HULL_H
#define HULL_H

#include <stdlib.h>

/*
	hull.h

	Empirical convex hull of observed (s1, s2) integer sufficient-statistic
	points. The hull grows monotonically as points are added during HMC
	(and is bootstrapped from the warm-start auxiliary sample). Membership
	queries ("is this point strictly interior to the hull?") are O(1) via
	a precomputed grid.

	This module deliberately makes no reference to any ground-truth hull,
	enumeration, or precomputed indicator file. Its only knowledge comes
	from the integer points it has been told to add.

	The bounding box (s1_max, s2_max) must be provided at creation and is
	computed by the caller from n(y_obs) -- e.g., for an ERGM with edges
	and 2-stars, s1_max = n*(n-1)/2 and s2_max = n*(n-1)*(n-2)/2.

	A lattice point is "strictly interior" iff it lies inside the polygon
	formed by the current hull vertices, not on any edge. Hull vertices
	themselves and lattice points lying on a hull edge are classified as
	boundary (not interior).
*/

typedef struct hull {
	int s1_max;
	int s2_max;
	int grid_stride;        /* = s2_max + 1, for row-major indexing */
	unsigned char *seen;    /* (s1_max+1)*(s2_max+1): 1 if observed */
	unsigned char *interior;/* (s1_max+1)*(s2_max+1): 1 if strictly interior */
	int *vert_s1;           /* hull vertices in CCW order */
	int *vert_s2;
	int  n_vertices;
	int  vert_capacity;
	int  dirty;             /* 1 if interior[] needs rebuild before next query */
	size_t n_observed;      /* number of unique (s1, s2) seen */
} HULL;

HULL  *hull_create(int s1_max, int s2_max);
void   hull_destroy(HULL *h);

/* Add a single lattice point. No-op if already seen or out of bounds.
   Marks the hull dirty if this is the first time we've seen the point. */
void   hull_add(HULL *h, int s1, int s2);

/* Strict-interior test. Triggers a lazy rebuild if anything has been
   added since the last query. Out-of-bounds and unseen cells return 0. */
int    hull_is_interior(HULL *h, int s1, int s2);

/* Diagnostics. */
size_t hull_n_observed(const HULL *h);
int    hull_n_vertices(HULL *h);          /* triggers rebuild if dirty */
void   hull_dump(HULL *h, const char *path);   /* writes observed + vertices */

#endif
