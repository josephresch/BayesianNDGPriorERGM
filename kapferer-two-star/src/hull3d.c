/*
	hull3d.c
	========

	Empirical convex hull of real-valued observed statistics (s1, s2, s3)
	in R^3. See hull3d.h for the public interface and a high-level
	description of the algorithm.

	This module is deliberately self-contained and references no
	ground-truth enumeration. Its only knowledge is the stream of points
	passed to hull3d_add.

	Algorithmic summary
	-------------------
	The polytope is represented as a triangulation: an array of faces,
	each a vertex triple (a, b, c) with an outward unit-normal plane
	(n_x, n_y, n_z, d) satisfying n . v = d for any vertex v of the face.
	Signed distance of an arbitrary point p from the face plane, oriented
	outward, is n . p - d.

	On each hull3d_add:
	  * If have_tetra == 0: cache the point in obs_*. If n_observed
	    reaches 4, try to seed a tetrahedron (see try_seed_tetrahedron).
	    On success, all remaining cached points are inserted incrementally.
	  * If have_tetra == 1: process the new point against the current
	    polytope (see process_point_incremental).

	process_point_incremental is O(F) in signed-distance tests plus
	O(E^2) in the horizon-edge search where F is the number of hull
	faces and E = 3 * F_visible. For random 3D point clouds, E[F] ~
	log^2 n, so per-insertion cost is effectively O(polylog n).

	Numerical tolerance eps
	-----------------------
	eps plays three roles, and the three thresholds together carve the
	signed-distance axis into three regimes:

	        <-- strict interior --|-- boundary band --|-- strict exterior -->
	                           -eps                  +eps

	  * visibility (> +eps): a face is visible iff signed distance is
	    strictly greater than +eps. Using strict > eps (not >= 0) avoids
	    creating zero-area new faces from a point that's numerically
	    coplanar with an existing face.
	  * seeding: we require four points whose affine independence is
	    at least eps (distance, perpendicular distance, and height of
	    the 4th from the triangle plane). Falling short at any stage
	    means "still too colinear / too coplanar, wait for more data".
	  * interior test (< -eps): a point is STRICTLY interior iff every
	    signed distance is < -eps. The boundary band (|signed distance|
	    <= eps) is classified as NON-interior on purpose: for the NDG
	    prior, "non-degenerate" explicitly excludes points sitting on a
	    face, and we'd rather reject a borderline sample than falsely
	    admit one. The visibility and interior tests deliberately use
	    the same band: a point in the band neither extends the hull nor
	    counts as interior, keeping both behaviours conservative.

	No attempt is made to detect or recover from catastrophic
	floating-point cancellation. For the Kapferer / egd model the
	sufficient statistics are O(10^2 .. 10^3) in magnitude and the
	default eps (1e-9) leaves ~9 digits of relative tolerance, which is
	comfortable for double-precision arithmetic.
*/

#include "hull3d.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================== */
/*  Buffer growth helpers                                             */
/* ================================================================== */

static void obs_grow(HULL3D *h, size_t need)
{
	if(need <= h->obs_capacity) return;
	size_t newcap = h->obs_capacity ? h->obs_capacity : 64;
	while(newcap < need) newcap *= 2;
	double *nx = realloc(h->obs_x, newcap * sizeof(double));
	double *ny = realloc(h->obs_y, newcap * sizeof(double));
	double *nz = realloc(h->obs_z, newcap * sizeof(double));
	if(nx) h->obs_x = nx;
	if(ny) h->obs_y = ny;
	if(nz) h->obs_z = nz;
	/* Only advertise the new capacity if ALL three reallocs succeeded;
	   otherwise callers that size off obs_capacity could overrun the
	   shorter allocation. */
	if(nx && ny && nz) h->obs_capacity = newcap;
}

static void face_grow(HULL3D *h, int need)
{
	if(need <= h->face_capacity) return;
	int newcap = h->face_capacity ? h->face_capacity : 32;
	while(newcap < need) newcap *= 2;
	int    *na  = realloc(h->face_a, (size_t)newcap * sizeof(int));
	int    *nb  = realloc(h->face_b, (size_t)newcap * sizeof(int));
	int    *nc  = realloc(h->face_c, (size_t)newcap * sizeof(int));
	double *nnx = realloc(h->nrm_x,  (size_t)newcap * sizeof(double));
	double *nny = realloc(h->nrm_y,  (size_t)newcap * sizeof(double));
	double *nnz = realloc(h->nrm_z,  (size_t)newcap * sizeof(double));
	double *nnd = realloc(h->nrm_d,  (size_t)newcap * sizeof(double));
	if(na)  h->face_a = na;
	if(nb)  h->face_b = nb;
	if(nc)  h->face_c = nc;
	if(nnx) h->nrm_x  = nnx;
	if(nny) h->nrm_y  = nny;
	if(nnz) h->nrm_z  = nnz;
	if(nnd) h->nrm_d  = nnd;
	if(na && nb && nc && nnx && nny && nnz && nnd)
		h->face_capacity = newcap;
}

static void vert_grow(HULL3D *h, int need)
{
	if(need <= h->vert_capacity) return;
	int newcap = h->vert_capacity ? h->vert_capacity : 16;
	while(newcap < need) newcap *= 2;
	int *nv = realloc(h->vert_idx, (size_t)newcap * sizeof(int));
	if(nv) { h->vert_idx = nv; h->vert_capacity = newcap; }
}

/* ================================================================== */
/*  Face plane helpers                                                */
/* ================================================================== */

/*
	face_set_plane: compute and store the outward unit normal and plane
	offset for face `face_idx`, given that its vertex triple (a, b, c)
	is already stored.

	The raw normal is the cross product (b - a) x (c - a). We then
	normalize to unit length so that signed distances are expressed in
	the same linear units as the input coordinates, which makes the
	`eps` tolerance meaningful without further scaling.

	The sign of (a, b, c) encodes the outward direction: this helper
	does NOT attempt to flip it. Callers that want outward orientation
	(see face_append_oriented) check the sign against a known interior
	reference point and permute (b, c) if needed BEFORE calling this
	function.
*/
static void face_set_plane(HULL3D *h, int face_idx)
{
	int a = h->face_a[face_idx];
	int b = h->face_b[face_idx];
	int c = h->face_c[face_idx];
	double ax = h->obs_x[a], ay = h->obs_y[a], az = h->obs_z[a];
	double bx = h->obs_x[b], by = h->obs_y[b], bz = h->obs_z[b];
	double cx = h->obs_x[c], cy = h->obs_y[c], cz = h->obs_z[c];
	double ux = bx - ax, uy = by - ay, uz = bz - az;
	double vx = cx - ax, vy = cy - ay, vz = cz - az;
	double nx = uy * vz - uz * vy;
	double ny = uz * vx - ux * vz;
	double nz = ux * vy - uy * vx;
	double mag = sqrt(nx*nx + ny*ny + nz*nz);
	if(mag > 0.0) { nx /= mag; ny /= mag; nz /= mag; }
	h->nrm_x[face_idx] = nx;
	h->nrm_y[face_idx] = ny;
	h->nrm_z[face_idx] = nz;
	h->nrm_d[face_idx] = nx * ax + ny * ay + nz * az;
}

/*
	face_signed_distance: signed distance of (x, y, z) from the plane of
	face_idx, positive on the outward side.
*/
static double face_signed_distance(HULL3D *h, int face_idx,
                                   double x, double y, double z)
{
	return h->nrm_x[face_idx] * x
	     + h->nrm_y[face_idx] * y
	     + h->nrm_z[face_idx] * z
	     - h->nrm_d[face_idx];
}

/*
	face_append_oriented: append a new triangular face (a, b, c) with its
	outward normal oriented AWAY from the fixed interior reference point
	(ref_x, ref_y, ref_z). If the initial orientation has ref on the
	outward side, swap b <-> c to flip the normal.

	Using a fixed interior reference avoids the need to reason about
	consistent CCW orientation across the whole polytope: every face is
	oriented independently based on one global invariant (ref is inside
	the polytope).

	The reference is the centroid of the seed tetrahedron, which stays
	strictly interior to the polytope for the lifetime of the hull
	because the polytope only grows.
*/
static void face_append_oriented(HULL3D *h, int a, int b, int c,
                                 double ref_x, double ref_y, double ref_z)
{
	face_grow(h, h->n_faces + 1);
	if(h->n_faces + 1 > h->face_capacity) return;  /* alloc failure -- skip */

	int f = h->n_faces;
	h->face_a[f] = a;
	h->face_b[f] = b;
	h->face_c[f] = c;
	face_set_plane(h, f);

	if(face_signed_distance(h, f, ref_x, ref_y, ref_z) > 0.0)
	{
		/* Reference on the outward side -> orientation is wrong.
		   Swap b and c to flip the normal. */
		h->face_b[f] = c;
		h->face_c[f] = b;
		face_set_plane(h, f);
	}

	h->n_faces++;
}

/* ================================================================== */
/*  Seeding the tetrahedron                                           */
/* ================================================================== */

/*
	try_seed_tetrahedron: find 4 affinely-independent points in the
	current obs_* list and build the seed tetrahedron.

	Robust seed selection (a staircase of escalating affine dimension):
	  i0 = leftmost (min x; ties broken arbitrarily)             -- anchor
	  i1 = point farthest from i0                                -- line
	  i2 = point farthest from line through i0 and i1            -- plane
	  i3 = point with largest perpendicular height from plane (i0, i1, i2)  -- volume

	Each step extends the affine span by one dimension and picks the
	candidate MAXIMALLY separated from the current span, so the final
	tetrahedron is as well-conditioned as the observation set allows.
	Picking naively (e.g. the first four points) can produce a nearly
	degenerate tetrahedron whose face normals are unstable under
	floating-point arithmetic.

	If at any stage the best candidate is within eps of the already-
	selected affine span, the observations are too degenerate and we
	give up (have_tetra stays 0). The caller will retry on the next
	hull3d_add.

	On success:
	  * the reference interior point is set to the centroid of the seed
	    tetrahedron;
	  * four faces are appended with correct outward orientations;
	  * ALL other observed points are inserted incrementally so the
	    polytope reflects the entire observation history, not just the
	    four seeds.
*/
static void process_point_incremental(HULL3D *h, size_t p_idx);

static void try_seed_tetrahedron(HULL3D *h)
{
	size_t n = h->n_observed;
	if(n < 4) return;

	/* Seed 1: leftmost (lexicographic on x). */
	size_t i0 = 0;
	for(size_t i = 1; i < n; ++i)
		if(h->obs_x[i] < h->obs_x[i0]) i0 = i;

	/* Seed 2: farthest from i0. Reject if max distance <= eps. */
	size_t i1 = (size_t)-1;
	double max_d2 = h->eps * h->eps;
	for(size_t i = 0; i < n; ++i)
	{
		if(i == i0) continue;
		double dx = h->obs_x[i] - h->obs_x[i0];
		double dy = h->obs_y[i] - h->obs_y[i0];
		double dz = h->obs_z[i] - h->obs_z[i0];
		double d2 = dx*dx + dy*dy + dz*dz;
		if(d2 > max_d2) { max_d2 = d2; i1 = i; }
	}
	if(i1 == (size_t)-1) return;

	/* Seed 3: farthest from line i0 -- i1. Measured as |u x (p_i - p0)|
	   / |u|, squared for speed. Reject if best perpendicular <= eps. */
	double ux = h->obs_x[i1] - h->obs_x[i0];
	double uy = h->obs_y[i1] - h->obs_y[i0];
	double uz = h->obs_z[i1] - h->obs_z[i0];
	double u_len_sq = ux*ux + uy*uy + uz*uz;

	size_t i2 = (size_t)-1;
	double max_perp2 = h->eps * h->eps;
	for(size_t i = 0; i < n; ++i)
	{
		if(i == i0 || i == i1) continue;
		double dx = h->obs_x[i] - h->obs_x[i0];
		double dy = h->obs_y[i] - h->obs_y[i0];
		double dz = h->obs_z[i] - h->obs_z[i0];
		double cx = uy*dz - uz*dy;
		double cy = uz*dx - ux*dz;
		double cz = ux*dy - uy*dx;
		double perp2 = (cx*cx + cy*cy + cz*cz) / u_len_sq;
		if(perp2 > max_perp2) { max_perp2 = perp2; i2 = i; }
	}
	if(i2 == (size_t)-1) return;

	/* Seed 4: largest perpendicular height from plane (i0, i1, i2).
	   Measured as |(p_i - p0) . n| / |n|, where n = (p1 - p0) x (p2 - p0).
	   Reject if best height <= eps. */
	double vx = h->obs_x[i2] - h->obs_x[i0];
	double vy = h->obs_y[i2] - h->obs_y[i0];
	double vz = h->obs_z[i2] - h->obs_z[i0];
	double nx = uy*vz - uz*vy;
	double ny = uz*vx - ux*vz;
	double nz = ux*vy - uy*vx;
	double nmag = sqrt(nx*nx + ny*ny + nz*nz);

	size_t i3 = (size_t)-1;
	double max_h = h->eps;
	for(size_t i = 0; i < n; ++i)
	{
		if(i == i0 || i == i1 || i == i2) continue;
		double dx = h->obs_x[i] - h->obs_x[i0];
		double dy = h->obs_y[i] - h->obs_y[i0];
		double dz = h->obs_z[i] - h->obs_z[i0];
		double dot = dx*nx + dy*ny + dz*nz;
		double ha  = fabs(dot) / nmag;
		if(ha > max_h) { max_h = ha; i3 = i; }
	}
	if(i3 == (size_t)-1) return;

	/* Record centroid as the reference interior point. */
	h->ref_x = (h->obs_x[i0] + h->obs_x[i1] + h->obs_x[i2] + h->obs_x[i3]) * 0.25;
	h->ref_y = (h->obs_y[i0] + h->obs_y[i1] + h->obs_y[i2] + h->obs_y[i3]) * 0.25;
	h->ref_z = (h->obs_z[i0] + h->obs_z[i1] + h->obs_z[i2] + h->obs_z[i3]) * 0.25;

	/* Build the four faces. Vertex ordering passed here is arbitrary;
	   face_append_oriented will permute b <-> c if needed so every
	   outward normal points AWAY from the centroid. */
	h->n_faces = 0;
	face_append_oriented(h, (int)i1, (int)i2, (int)i3, h->ref_x, h->ref_y, h->ref_z);
	face_append_oriented(h, (int)i0, (int)i2, (int)i3, h->ref_x, h->ref_y, h->ref_z);
	face_append_oriented(h, (int)i0, (int)i1, (int)i3, h->ref_x, h->ref_y, h->ref_z);
	face_append_oriented(h, (int)i0, (int)i1, (int)i2, h->ref_x, h->ref_y, h->ref_z);

	h->have_tetra = 1;
	h->vert_dirty = 1;

	/* Replay every other observation as an incremental insertion so the
	   polytope reflects the full history, not just the four seeds.
	   Points that fall inside the seed tetrahedron are no-ops; points
	   outside extend it. */
	for(size_t i = 0; i < n; ++i)
	{
		if(i == i0 || i == i1 || i == i2 || i == i3) continue;
		process_point_incremental(h, i);
	}
}

/* ================================================================== */
/*  Incremental insertion of one point                                */
/* ================================================================== */

/*
	process_point_incremental: extend the current polytope by one point.

	Assumes have_tetra == 1 and the point's coordinates are already
	stored in obs_*[p_idx].

	Steps:
	  1. Find all faces visible from the point (signed distance > eps).
	     If none, point is inside -- return.
	  2. Count each undirected edge among visible faces' edges.
	     count == 1 -> horizon (edge on the cap boundary);
	     count == 2 -> internal (edge inside the visible cap).
	  3. Delete visible faces via in-place compaction (O(F)).
	  4. Add one new face (u, v, p) per horizon edge.
	  5. Mark vertex list dirty.

	Why the edge-count trick works (the key "kink")
	-----------------------------------------------
	Every triangular face contributes three undirected edges. In the
	set of VISIBLE faces (the "cap" of the polytope that the new point
	can see), an edge is either:
	  - shared by two visible triangles: it is interior to the cap and
	    must disappear when we replace the cap with a fan of new faces
	    joined at p; or
	  - shared by one visible triangle and one HIDDEN triangle: it is
	    on the boundary of the cap, where the hidden polytope meets the
	    soon-to-be-replaced visible region. These are the horizon
	    edges; each becomes the base of exactly one new triangle
	    (horizon_edge, p).
	A well-formed polytope has every edge in exactly two triangles
	globally, so within the visible subset an edge appearing in ONE
	triangle is automatically the one also shared with a hidden
	triangle -- no separate scan of the hidden set is needed. This
	reduces horizon discovery to an O(E) edge-count pass over the cap,
	avoiding an O(F_total) scan of the whole polytope.

	The O(E^2) inner loop (linear search to deduplicate edges) is
	deliberate: for the hull sizes we see (|faces| ~ 50, |visible| << |faces|)
	a hash table would add allocator traffic and cache misses that
	swamp its asymptotic advantage.
*/
static void process_point_incremental(HULL3D *h, size_t p_idx)
{
	double px = h->obs_x[p_idx];
	double py = h->obs_y[p_idx];
	double pz = h->obs_z[p_idx];

	/* ---- 1. Visible faces. ---- */
	if(h->scratch_visible_cap < h->n_faces)
	{
		int newcap = h->scratch_visible_cap ? h->scratch_visible_cap : 32;
		while(newcap < h->n_faces) newcap *= 2;
		int *nv = realloc(h->scratch_visible, (size_t)newcap * sizeof(int));
		if(!nv) return;
		h->scratch_visible     = nv;
		h->scratch_visible_cap = newcap;
	}
	int n_vis = 0;
	for(int f = 0; f < h->n_faces; ++f)
	{
		if(face_signed_distance(h, f, px, py, pz) > h->eps)
			h->scratch_visible[n_vis++] = f;
	}
	if(n_vis == 0) return;  /* point inside current hull */

	/* ---- 2. Count undirected edges across visible faces. ----
	   Each visible triangle contributes 3 edges; we canonicalise each
	   as an ordered pair (min, max) so edges traversed in opposite
	   directions by adjacent faces collapse to the same key. After the
	   pass, ct[k] == 1 identifies horizon edges; ct[k] == 2 identifies
	   internal edges that disappear with the cap. */
	int needed_edges = 3 * n_vis;
	if(h->scratch_edge_cap < needed_edges)
	{
		int newcap = h->scratch_edge_cap ? h->scratch_edge_cap : 96;
		while(newcap < needed_edges) newcap *= 2;
		int *nu = realloc(h->scratch_edge_u,  (size_t)newcap * sizeof(int));
		int *nv = realloc(h->scratch_edge_v,  (size_t)newcap * sizeof(int));
		int *nc = realloc(h->scratch_edge_ct, (size_t)newcap * sizeof(int));
		if(nu) h->scratch_edge_u  = nu;
		if(nv) h->scratch_edge_v  = nv;
		if(nc) h->scratch_edge_ct = nc;
		if(!(nu && nv && nc)) return;
		h->scratch_edge_cap = newcap;
	}
	int n_edges = 0;
	for(int vi = 0; vi < n_vis; ++vi)
	{
		int f = h->scratch_visible[vi];
		int fv[3] = { h->face_a[f], h->face_b[f], h->face_c[f] };
		for(int e = 0; e < 3; ++e)
		{
			int a = fv[e];
			int b = fv[(e+1) % 3];
			int u = (a < b) ? a : b;
			int v = (a < b) ? b : a;
			int found = -1;
			for(int k = 0; k < n_edges; ++k)
			{
				if(h->scratch_edge_u[k] == u && h->scratch_edge_v[k] == v)
				{
					found = k;
					break;
				}
			}
			if(found >= 0)
			{
				h->scratch_edge_ct[found]++;
			}
			else
			{
				h->scratch_edge_u[n_edges]  = u;
				h->scratch_edge_v[n_edges]  = v;
				h->scratch_edge_ct[n_edges] = 1;
				n_edges++;
			}
		}
	}

	/* ---- 3. Delete visible faces by in-place compaction. ---- */
	if(h->scratch_del_cap < h->n_faces)
	{
		int newcap = h->scratch_del_cap ? h->scratch_del_cap : 64;
		while(newcap < h->n_faces) newcap *= 2;
		char *nd = realloc(h->scratch_del, (size_t)newcap);
		if(!nd) return;
		h->scratch_del     = nd;
		h->scratch_del_cap = newcap;
	}
	memset(h->scratch_del, 0, (size_t)h->n_faces);
	for(int vi = 0; vi < n_vis; ++vi) h->scratch_del[h->scratch_visible[vi]] = 1;

	int write = 0;
	for(int f = 0; f < h->n_faces; ++f)
	{
		if(!h->scratch_del[f])
		{
			if(write != f)
			{
				h->face_a[write] = h->face_a[f];
				h->face_b[write] = h->face_b[f];
				h->face_c[write] = h->face_c[f];
				h->nrm_x[write]  = h->nrm_x[f];
				h->nrm_y[write]  = h->nrm_y[f];
				h->nrm_z[write]  = h->nrm_z[f];
				h->nrm_d[write]  = h->nrm_d[f];
			}
			write++;
		}
	}
	h->n_faces = write;

	/* ---- 4. Add one new face per horizon edge. ---- */
	for(int k = 0; k < n_edges; ++k)
	{
		if(h->scratch_edge_ct[k] == 1)
		{
			face_append_oriented(h,
				h->scratch_edge_u[k],
				h->scratch_edge_v[k],
				(int)p_idx,
				h->ref_x, h->ref_y, h->ref_z);
		}
	}

	h->vert_dirty = 1;
}

/* ================================================================== */
/*  Vertex list refresh                                               */
/* ================================================================== */

static void refresh_vert_list(HULL3D *h)
{
	h->n_vertices = 0;
	if(!h->have_tetra || h->n_observed == 0 || h->n_faces == 0)
	{
		h->vert_dirty = 0;
		return;
	}

	char *used = calloc(h->n_observed, sizeof(char));
	if(!used) return;  /* leave vert_dirty=1 so next query retries */

	for(int f = 0; f < h->n_faces; ++f)
	{
		used[h->face_a[f]] = 1;
		used[h->face_b[f]] = 1;
		used[h->face_c[f]] = 1;
	}

	for(size_t i = 0; i < h->n_observed; ++i)
	{
		if(used[i])
		{
			vert_grow(h, h->n_vertices + 1);
			if(h->n_vertices < h->vert_capacity)
				h->vert_idx[h->n_vertices++] = (int)i;
		}
	}
	free(used);
	h->vert_dirty = 0;
}

/* ================================================================== */
/*  Public API                                                        */
/* ================================================================== */

HULL3D *hull3d_create(double eps)
{
	HULL3D *h = calloc(1, sizeof(HULL3D));
	if(!h) return NULL;

	h->eps = (eps > 0.0) ? eps : 1e-9;

	h->obs_capacity = 64;
	h->obs_x = malloc(h->obs_capacity * sizeof(double));
	h->obs_y = malloc(h->obs_capacity * sizeof(double));
	h->obs_z = malloc(h->obs_capacity * sizeof(double));

	h->face_capacity = 32;
	h->face_a = malloc((size_t)h->face_capacity * sizeof(int));
	h->face_b = malloc((size_t)h->face_capacity * sizeof(int));
	h->face_c = malloc((size_t)h->face_capacity * sizeof(int));
	h->nrm_x  = malloc((size_t)h->face_capacity * sizeof(double));
	h->nrm_y  = malloc((size_t)h->face_capacity * sizeof(double));
	h->nrm_z  = malloc((size_t)h->face_capacity * sizeof(double));
	h->nrm_d  = malloc((size_t)h->face_capacity * sizeof(double));

	h->vert_capacity = 16;
	h->vert_idx = malloc((size_t)h->vert_capacity * sizeof(int));

	if(!h->obs_x || !h->obs_y || !h->obs_z ||
	   !h->face_a || !h->face_b || !h->face_c ||
	   !h->nrm_x || !h->nrm_y || !h->nrm_z || !h->nrm_d ||
	   !h->vert_idx)
	{
		hull3d_destroy(h);
		return NULL;
	}

	return h;
}

void hull3d_destroy(HULL3D *h)
{
	if(!h) return;
	free(h->obs_x);
	free(h->obs_y);
	free(h->obs_z);
	free(h->face_a);
	free(h->face_b);
	free(h->face_c);
	free(h->nrm_x);
	free(h->nrm_y);
	free(h->nrm_z);
	free(h->nrm_d);
	free(h->vert_idx);
	free(h->scratch_visible);
	free(h->scratch_edge_u);
	free(h->scratch_edge_v);
	free(h->scratch_edge_ct);
	free(h->scratch_del);
	free(h);
}

void hull3d_add(HULL3D *h, double x, double y, double z)
{
	if(!h) return;

	/* Append to obs_*. */
	if(h->n_observed + 1 > h->obs_capacity)
	{
		obs_grow(h, h->n_observed + 1);
		if(h->n_observed + 1 > h->obs_capacity) return;  /* alloc failed */
	}
	h->obs_x[h->n_observed] = x;
	h->obs_y[h->n_observed] = y;
	h->obs_z[h->n_observed] = z;
	h->n_observed++;

	if(h->have_tetra)
	{
		process_point_incremental(h, h->n_observed - 1);
	}
	else if(h->n_observed >= 4)
	{
		try_seed_tetrahedron(h);
	}
}

int hull3d_is_interior(HULL3D *h, double x, double y, double z)
{
	if(!h || !h->have_tetra) return 0;
	if(h->n_faces <= 0) return 0;
	for(int f = 0; f < h->n_faces; ++f)
	{
		/* Strictly interior means strictly negative signed distance for
		   every face. The -eps threshold excludes points numerically on
		   the polytope boundary so samples sitting on a face are
		   classified as boundary rather than interior. */
		if(face_signed_distance(h, f, x, y, z) > -h->eps) return 0;
	}
	return 1;
}

size_t hull3d_n_observed(const HULL3D *h)
{
	return h ? h->n_observed : 0;
}

int hull3d_n_vertices(HULL3D *h)
{
	if(!h) return 0;
	if(h->vert_dirty) refresh_vert_list(h);
	return h->n_vertices;
}

int hull3d_n_faces(HULL3D *h)
{
	return h ? h->n_faces : 0;
}

void hull3d_dump(HULL3D *h, const char *path)
{
	if(!h || !path) return;
	if(h->vert_dirty) refresh_vert_list(h);

	FILE *f = fopen(path, "w");
	if(!f) return;

	fprintf(f,
		"# hull3d dump: %zu observed, %d vertices, %d faces, eps=%.3e\n",
		h->n_observed, h->n_vertices, h->n_faces, h->eps);

	fprintf(f, "# section: observed\n");
	fprintf(f, "s1 s2 s3\n");
	for(size_t i = 0; i < h->n_observed; ++i)
		fprintf(f, "%.17g %.17g %.17g\n",
			h->obs_x[i], h->obs_y[i], h->obs_z[i]);

	fprintf(f, "# section: vertex indices (into observed)\n");
	fprintf(f, "idx\n");
	for(int i = 0; i < h->n_vertices; ++i)
		fprintf(f, "%d\n", h->vert_idx[i]);

	fprintf(f, "# section: faces (vertex triples, indices into observed)\n");
	fprintf(f, "a b c\n");
	for(int i = 0; i < h->n_faces; ++i)
		fprintf(f, "%d %d %d\n",
			h->face_a[i], h->face_b[i], h->face_c[i]);

	fclose(f);
}
