# Kapferer, geometrically weighted model

Paper Section 7.1. Kapferer's tailor shop network, 39 nodes and 158 undirected
edges, model `edges + gwesp(0.25, fixed) + gwdsp(0.25, fixed)` (model code
`egd` in the C driver).

The geometrically weighted statistics already control degeneracy through their
diminishing-return structure, so the non-degeneracy prior has little left to do.
This folder is the control case. If the prior interfered with a well-specified
model it would show up here, and it does not. All five prior settings give
predictive distributions that overlap.

The companion folder `kapferer-two-star/` fits the same network with the
degeneracy-prone two-star count in place of GWDSP, and there the prior changes the answer.

## Contents

    01_find_map.R              Robbins-Monro search for the posterior mode
    02_find_mass_matrix.R      HMC mass matrix from the auxiliary covariance at the MAP
    03_build_ndg_prior.R       simulate 100,000 networks at the MPLE -> data/ndg_prior.txt
    04_run_hmc_single.R        one chain, plus goodness-of-fit reporting
    05_run_hmc_replicates.R    20 chains, the protocol the paper reports
    ergm_helpers.R             shared simulation and prior-gradient helpers
    verify_changestats.R       ergm::summary vs the C change statistics

    data/      network and generated warm-start pool
    src/       C sampler
    tuning/    MAP, mass matrix, and the logs from steps 01 and 02
    chains/    raw text output from the C binary, overwritten on each run
    results/   posteriors and figures

Run the R scripts from this folder. Steps 01 and 02 only matter if the network
or model changes; their answers are already compiled into `src/hmc.c` and saved
under `tuning/`.

## Build and run

Needs GSL 2.2+ and R with `ergm`, `network`, `numDeriv`, `geometry`, `coda`.

    cd src && make all      # builds ./hmc
    make test               # expect 41 C assertions and 6 R cross-checks passing
    cd .. && Rscript 03_build_ndg_prior.R
    Rscript 05_run_hmc_replicates.R

Set `prior_type` and `ndg_p` at the top of script 04 or 05 to choose the prior.

## Data

    kapferer.txt     observed network, "graphformat 1" as the C code reads it
    ndg_prior.txt    warm-start pool written by 03_build_ndg_prior.R

Observed statistics: edges 158, gwesp(0.25, fixed) 185.7892,
gwdsp(0.25, fixed) 671.1218.

Unlike the n = 7 study there is no enumeration here. 2^741 graphs cannot be
listed, so the convex hull is estimated adaptively from the warm-start pool and
grown during sampling, and there is no exact posterior to compare against.

## Results

    posterior-20rep-<prior>.RDS   20 chains, pooled. Table 3.
    gof-20rep-<prior>.png         four-panel goodness-of-fit. Figure 2.
    gof-20rep-<prior>-<panel>.png the same panels saved individually
    trace-20rep-<prior>-*.png     marginal densities and pairwise hex plots
    prior-sensitivity-20rep.png   marginal densities overlaid across priors
    posterior-1chain-*.RDS        single chains and their diagnostics, from script 04

`<prior>` is `normal`, `uniform`, or `ndg-p<p>` with p in {1, 5, 10}. Figure 2
in the paper uses the normal, uniform, `ndg-p1` and `ndg-p10` panels.

## Sampler

Adapted from Stoehr, Benson and Friel (2019). The prior module and the hull
code were added for this paper.

    hmc.c            noisy leapfrog gradient and the leapfrog estimator of Z ratios
    prior.c          normal, uniform, and non-degeneracy priors behind one interface
    hull3d.c         3D convex hull, grown as auxiliary statistics arrive
    graph_2410.c     ERGM change statistics and the tie-no-tie sampler
    test_changestats.c  full-graph statistics, cross-checked by verify_changestats.R

Values from 01 and 02, already in `hmc.c` and under `tuning/`:

    MAP          (-1.6108722642630, 0.6626369583688, -0.1886951948627)
    mass matrix  (row-major, 3x3)
        1216.006849699398572  1865.081165436773972   4845.486883567362383
        1865.081165436773972  2886.529283705796843   7414.617605358649598
        4845.486883567362383  7414.617605358649598  20077.955452080885152

Priors: `normal` is the Caimo and Friel default N(0, diag(100)) per dimension;
`uniform` has bounds from MCMLE +/- 10 SE with GWDSP widened, edges in [-8, 2],
gwesp in [-2, 5], gwdsp in [-0.2, 0.05]; `ndg` takes strength p through `ndg_p`
and the warm-start pool through `ndg_file`.

`data/ndg_prior.txt` has a row count on line 1, then one row per network:
`s1 s2 s3` for edges, gwesp and gwdsp. No hull or reference parameter is
written. The prior builds its own 3D hull as sampling proceeds.

GWESP and GWDSP change statistics use a hardcoded alpha of 0.25 in
`graph_2410.c`. The warm-start pool is simulated at the MPLE rather than the
MAP, so the prior does not depend on the posterior workflow it feeds.
