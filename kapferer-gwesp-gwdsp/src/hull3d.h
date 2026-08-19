#ifndef HULL3D_H
#define HULL3D_H

#include <stdlib.h>

/*
	hull3d.h
	========

	Empirical convex hull of observed real-valued sufficient-statistic
	points (s1, s2, s3) in R^3. Used by the adaptive NDG prior (prior.c)
	for the Kapferer / egd model where the statistics
	(edges, gwesp, gwdsp) are a mix of integer and real-valued
	quantities and therefore cannot be tracked on an integer lattice
	bitmap.

	Design
	------
	Because the statistics are real-valued we cannot enumerate lattice
	cells, so:
	  * observed points are kept as a growing list of doubles;
	  * the hull is maintained as a triangulated polytope (arrays of face
	    vertex-triples plus a unit outward-normal plane per face);
	  * it is updated INCREMENTALLY on every hull3d_add (no "lazy rebuild
	    from scratch on next query" pattern -- full rebuild would be
	    quadratic in the number of observed points and unaffordable
	    for warm-start pools of O(10^5) samples).

	Incremental update algorithm per new point p
	--------------------------------------------
	  1. Mark a face F as "visible from p" iff its outward-facing signed
	     distance to p is positive (> eps). p is geometrically outside F.
	  2. If no face is visible, p is inside (or on the boundary of) the
	     current polytope -- skip (the hull doesn't need to change).
	  3. Otherwise, the visible faces form a connected cap. Every
	     undirected edge that is shared between two visible faces is
	     internal to the cap; every edge shared with a non-visible face
	     is a horizon edge. We identify horizons by counting how often
	     each undirected edge appears across the visible set:
	         count == 1 -> horizon
	         count == 2 -> internal
	  4. Delete all visible faces.
	  5. For each horizon edge (u, v), add a new face (u, v, p).
	     face_append_oriented() flips orientation on creation if needed
	     so the outward normal points AWAY from a fixed interior reference
	     point (the centroid of the seed tetrahedron).

	Seeding
	-------
	The polytope is not formed until 4 affinely-independent observed
	points have been seen. Points added before seeding are cached and
	inserted incrementally as soon as the seed tetrahedron is built.

	Interior test
	-------------
	A point q is STRICTLY interior iff for every face, the signed
	distance n . q - d is strictly less than -eps. This excludes points
	on the polytope boundary (where signed distance == 0) as well as
	points within numerical eps of a face.

	Monotone growth
	---------------
	hull3d_add is append-only with respect to the observation list and
	hull_extending with respect to the polytope: inserting a point that
	lies outside the current polytope can only add area (visible faces
	are replaced by the "tent" of new faces anchored at p). Inserting an
	interior point leaves the polytope unchanged.

	No ground-truth knowledge
	-------------------------
	This module never reads a precomputed enumeration. It maintains the
	empirical hull of the points it has been told to add.
*/

typedef struct hull3d {
	/* Observation list. Grows monotonically via hull3d_add. */
	double *obs_x;
	double *obs_y;
	double *obs_z;
	size_t  n_observed;
	size_t  obs_capacity;

	/* Faces of the current hull, as triangles (a, b, c) whose vertex
	   indices index into obs_*. Outward unit normal stored per face. */
	int    *face_a;
	int    *face_b;
	int    *face_c;
	double *nrm_x;       /* outward unit normal */
	double *nrm_y;
	double *nrm_z;
	double *nrm_d;       /* n . v  for any vertex v of the face (plane offset) */
	int     n_faces;
	int     face_capacity;

	/* Unique vertex indices (into obs_*) used by the current faces.
	   Lazily recomputed on query when vert_dirty == 1. */
	int    *vert_idx;
	int     n_vertices;
	int     vert_capacity;
	int     vert_dirty;

	/* Reference interior point: centroid of the seed tetrahedron, used
	   by face_append_oriented to determine outward orientation. Stays
	   inside the polytope for the lifetime of the hull because the
	   polytope only grows. */
	double  ref_x;
	double  ref_y;
	double  ref_z;
	int     have_tetra;   /* 1 once the seed tetrahedron has been built */

	/* Numerical tolerance for visibility / interiority / coplanarity. */
	double  eps;

	/* Scratch buffers for incremental insertion (persistent across
	   hull3d_add calls to avoid churn in the hot path). */
	int    *scratch_visible;
	int     scratch_visible_cap;
	int    *scratch_edge_u;
	int    *scratch_edge_v;
	int    *scratch_edge_ct;
	int     scratch_edge_cap;
	char   *scratch_del;
	int     scratch_del_cap;
} HULL3D;

/* Allocate an empty 3D hull. eps is the numerical tolerance used for
   visibility / coplanarity / strict-interior tests. Pass 0 for the
   default (1e-9). Returns NULL on allocation failure. */
HULL3D *hull3d_create(double eps);

void    hull3d_destroy(HULL3D *h);

/* Add one observed point. If the seed tetrahedron has not been built,
   the point is cached; once four affinely-independent points have been
   accumulated, the tetrahedron is seeded and any still-pending points
   are inserted incrementally. If the tetrahedron already exists, the
   new point is processed against the current polytope: extending it
   if the point is outside, else no-op. */
void    hull3d_add(HULL3D *h, double x, double y, double z);

/* Strict-interior test. Returns 1 iff (x, y, z) lies more than eps
   inside every face of the current polytope. Returns 0 if the polytope
   has not been seeded yet. Callers need NOT have added (x, y, z)
   beforehand; the query is a pure plane test. */
int     hull3d_is_interior(HULL3D *h, double x, double y, double z);

/* Diagnostics. */
size_t  hull3d_n_observed(const HULL3D *h);
int     hull3d_n_vertices(HULL3D *h);     /* triggers vertex refresh if dirty */
int     hull3d_n_faces(HULL3D *h);

/* Write a human-readable snapshot of the hull to `path`. Format:
     # hull3d dump: <n_observed> observed, <n_vertices> vertices,
     #             <n_faces> faces, eps=<eps>
     # section: observed
     s1 s2 s3
     <x y z, one per line, in insertion order>
     # section: vertex indices (into observed)
     idx
     <indices, one per line>
     # section: faces (vertex triples, indices into observed)
     a b c
     <triples, one per line>
   Used by test_prior.c for the R cross-check. */
void    hull3d_dump(HULL3D *h, const char *path);

#endif
