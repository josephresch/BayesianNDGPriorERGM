/*
    test_changestats.c

    Self-consistency check for GWESP / GWDSP change-statistics on the Kapferer
    network. For a sample of dyads, verify

        get_gwesp_change(G, idx, alpha) == gwesp(G_toggled) - gwesp(G)
        get_gwdsp_change(G, idx, alpha) == gwdsp(G_toggled) - gwdsp(G)

    where G_toggled is G with dyad `idx` flipped. The full stats on both sides
    are recomputed from scratch via get_gwesp / get_gwdsp, so the change-stats
    implementation is tested independently of its own incremental bookkeeping.

    The toggle also exercises adding AND removing edges: when idx is currently
    an edge we remove it, otherwise we add it; then we flip back and re-check
    the reverse delta to verify both sign paths.

    Exit code: 0 on all-pass, 1 on any failure.
*/

#include "graph_2410.h"

#include <gsl/gsl_rng.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Hand-apply the same bookkeeping as tnt_accept_change, but without a sampler.
   Toggles DYADlist[idx]->edge and fixes up nedges / degree[i] / degree[j]. */
static void toggle_dyad(GRAPH *g, size_t idx)
{
    DYAD *d = g->DYADlist[idx];
    if(d->edge)
    {
        d->edge = 0;
        g->nedges--;
        g->degree[d->i]--;
        g->degree[d->j]--;
    }
    else
    {
        d->edge = 1;
        g->nedges++;
        g->degree[d->i]++;
        g->degree[d->j]++;
    }
}

static int check_one(GRAPH *g, size_t idx, double tol, int verbose)
{
    /* Pre-toggle stats and predicted deltas. */
    double esp_before = get_gwesp(g, ALPHA_GWESP);
    double dsp_before = get_gwdsp(g, ALPHA_GWDSP);
    double d_esp_pred = get_gwesp_change(g, idx, ALPHA_GWESP);
    double d_dsp_pred = get_gwdsp_change(g, idx, ALPHA_GWDSP);
    int was_edge      = g->DYADlist[idx]->edge;

    /* Forward toggle. */
    toggle_dyad(g, idx);
    double esp_after = get_gwesp(g, ALPHA_GWESP);
    double dsp_after = get_gwdsp(g, ALPHA_GWDSP);
    double d_esp_act = esp_after - esp_before;
    double d_dsp_act = dsp_after - dsp_before;

    int pass_fwd =
        (fabs(d_esp_pred - d_esp_act) < tol) &&
        (fabs(d_dsp_pred - d_dsp_act) < tol);

    /* Reverse-direction check: the change-stat on the toggled graph should
       be the negative of the forward delta (pre-toggle semantics again). */
    double d_esp_pred_rev = get_gwesp_change(g, idx, ALPHA_GWESP);
    double d_dsp_pred_rev = get_gwdsp_change(g, idx, ALPHA_GWDSP);

    int pass_rev =
        (fabs(d_esp_pred_rev + d_esp_pred) < tol) &&
        (fabs(d_dsp_pred_rev + d_dsp_pred) < tol);

    /* Restore the graph and verify stats come back exactly. */
    toggle_dyad(g, idx);
    double esp_restore = get_gwesp(g, ALPHA_GWESP);
    double dsp_restore = get_gwdsp(g, ALPHA_GWDSP);

    int pass_restore =
        (fabs(esp_restore - esp_before) < tol) &&
        (fabs(dsp_restore - dsp_before) < tol);

    int pass = pass_fwd && pass_rev && pass_restore;

    if(verbose || !pass)
    {
        size_t ii = g->DYADlist[idx]->i;
        size_t jj = g->DYADlist[idx]->j;
        printf("  dyad idx=%4zu (i=%2zu,j=%2zu, was_edge=%d): "
               "d_esp pred=%+.12e act=%+.12e | "
               "d_dsp pred=%+.12e act=%+.12e  [%s%s%s]\n",
               idx, ii, jj, was_edge,
               d_esp_pred, d_esp_act,
               d_dsp_pred, d_dsp_act,
               pass_fwd ? "fwd " : "FWD!",
               pass_rev ? "rev " : "REV!",
               pass_restore ? "restore" : "RESTORE!");
    }

    return pass;
}

int main(int nargs, char **args)
{
    const char *graph_path = (nargs >= 2) ? args[1] : "../kapferer.txt";
    size_t n_samples       = (nargs >= 3) ? (size_t)atol(args[2]) : 60;
    unsigned long seed     = (nargs >= 4) ? strtoul(args[3], NULL, 10) : 20260417UL;
    double tol             = 1e-9;
    int verbose            = (nargs >= 5) ? atoi(args[4]) : 0;

    printf("test_changestats: graph=%s samples=%zu seed=%lu tol=%.1e\n",
           graph_path, n_samples, seed, tol);

    GRAPH *g = loadGRAPH((char *)graph_path);
    if(!g)
    {
        fprintf(stderr, "ERROR: failed to load graph %s\n", graph_path);
        return EXIT_FAILURE;
    }
    printf("  loaded: nnodes=%zu ndyads=%zu nedges=%zu\n",
           g->nnodes, g->ndyads, g->nedges);

    double esp0 = get_gwesp(g, ALPHA_GWESP);
    double dsp0 = get_gwdsp(g, ALPHA_GWDSP);
    printf("  full stats: edges=%zu gwesp(%.3f)=%.10f gwdsp(%.3f)=%.10f\n",
           g->nedges, ALPHA_GWESP, esp0, ALPHA_GWDSP, dsp0);

    gsl_rng *rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, seed);

    size_t fails = 0, edges_tested = 0, nonedges_tested = 0;

    /* Deterministic coverage: a few fixed probes (first/last dyad, plus the
       first current edge and first current non-edge), then random fill-in. */
    size_t fixed_idx[4] = {0, g->ndyads - 1, 0, 0};
    size_t fi;
    /* locate first edge / first non-edge */
    for(fi = 0; fi < g->ndyads; ++fi) if(g->DYADlist[fi]->edge)  { fixed_idx[2] = fi; break; }
    for(fi = 0; fi < g->ndyads; ++fi) if(!g->DYADlist[fi]->edge) { fixed_idx[3] = fi; break; }

    printf("\n--- fixed probes ---\n");
    size_t p;
    for(p = 0; p < 4; ++p)
    {
        size_t idx = fixed_idx[p];
        if(g->DYADlist[idx]->edge) edges_tested++; else nonedges_tested++;
        if(!check_one(g, idx, tol, 1)) fails++;
    }

    printf("\n--- %zu random probes ---\n", n_samples);
    size_t s;
    for(s = 0; s < n_samples; ++s)
    {
        size_t idx = gsl_rng_uniform_int(rng, g->ndyads);
        if(g->DYADlist[idx]->edge) edges_tested++; else nonedges_tested++;
        if(!check_one(g, idx, tol, verbose)) fails++;
    }

    printf("\nsummary: tested=%zu (edges=%zu non-edges=%zu) failures=%zu\n",
           n_samples + 4, edges_tested, nonedges_tested, fails);

    gsl_rng_free(rng);
    destroyGRAPH(g);

    if(fails == 0) { printf("PASS\n"); return EXIT_SUCCESS; }
    printf("FAIL\n");
    return EXIT_FAILURE;
}
