# Two-star model, n = 7

Paper Sections 5 and 6. Network of 7 nodes with observed statistics
`g(y_obs) = (7 edges, 14 two-stars)`, model `edges + kstar(2)`.

At this size the sample space has 2^21 graphs and only 144 distinct `(e, t)`
pairs, so the posterior can be computed exactly by enumeration. That exact
posterior is what the noisy HMC output is measured against, and it is the only
place in the paper where the approximation has a ground truth.

## Contents

    01_find_map.R              Robbins-Monro search for the posterior mode
    02_find_mass_matrix.R      HMC mass matrix from the auxiliary covariance at the MAP
    03_build_ndg_prior.R       simulate 100,000 networks at the MPLE -> data/ndg_prior.txt
    04_run_hmc_single.R        one chain, plus mu-space reporting
    05_run_hmc_replicates.R    20 chains, the protocol the paper reports
    ergm_helpers.R             shared simulation and prior-gradient helpers

    data/      network, enumeration products, generated warm-start pool
    src/       C sampler (see below)
    chains/    raw text output from the C binary, overwritten on each run
    results/   posteriors used by the paper

Run the R scripts from this folder. Steps 01 and 02 only matter if the network
or model changes; their answers are already compiled into `src/hmc.c` and
repeated below.

## Build and run

Needs GSL 2.2+ and R with `ergm`, `network`, `numDeriv`, `MASS`, `ggplot2`,
`spatstat.geom` and `spatstat.explore`.

Scripts 04 and 05 also need **RcppERGM**, which is not on CRAN. They call
`RcppERGM::EtaToMu` for the natural-parameter to mean-value map and
`RcppERGM::PND` for the exact non-degeneracy probability, which together supply
every mu-space number in Table 1 and Figure 1. Its source ships at the
repository root; install it with `R CMD INSTALL RcppERGM` from there. Scripts
01, 02 and 03 and the `make test` battery do not need it.

    cd src && make all      # builds ./hmc
    make test               # C assertions plus the R cross-check
    cd .. && Rscript 03_build_ndg_prior.R
    Rscript 05_run_hmc_replicates.R

Set `prior_type` and `ndg_p` at the top of script 04 or 05 to choose the prior.

## Data

    g7.txt            observed network, "graphformat 1" as the C code reads it
    g7.RData          all 2^21 graphs on 7 nodes
    g7kstars.RData    their star counts
    inhull.RDS        interior indicator over the 144 distinct (e, t) pairs
    onhull.RDS        boundary indicator over the same
    onhull_eta.RDS    natural parameters at the boundary points
    ndg_prior.txt     warm-start pool written by 03_build_ndg_prior.R

The five enumeration files come from the exact computation in the companion
paper. Scripts 04 and 05 read them for the mu-space reporting and the KL
comparison. They are inputs here, never regenerated.

## Results

    posterior-20rep-<prior>.RDS      20 chains, pooled. Table 1.
    posterior-exact-<prior>.RDS      exact posterior on a grid, the reference
                                     for D_KL(p || p_nHMC) in Table 1
    posterior-1chain-<prior>.RDS     single chains from 04_run_hmc_single.R

`<prior>` is `normal`, `uniform`, or `ndg-p<p>`. Table 1 covers
p in {1, 1.5, 2, 3, 5, 10, 20}. Exact posteriors also exist at p = 50 and
p = 100; the paper does not report them.

Figure 1 is written to `results/<name>-mu-overlay.png` by the plotting block at
the end of script 05. The copies in the manuscript were produced before the
files were renamed.

Table 2, the sensitivity of D_KL to the dual-averaging acceptance target, comes
from rerunning script 05 with `target_accept` set to 0.574, 0.651 and 0.75.
Only the 0.651 outputs are kept here.

## Verification

`make test` in `src/` builds `test_prior` and runs it with `src/test_prior.R`.
The battery is self-contained: it generates its own pool and does not read
`data/ndg_prior.txt`.

## Sampler

Adapted from Stoehr, Benson and Friel (2019), whose implementation is written
in C against GSL; this code stays in C. `hmc.c`, `hmc_driver.c` and
`graph_2410.c` are theirs with modifications. The prior module and the hull
code were added for this paper.

    hmc.c          noisy leapfrog gradient and the leapfrog estimator of Z ratios
    prior.c        normal, uniform, and non-degeneracy priors behind one interface
    hull.c         2D convex hull, grown as auxiliary statistics arrive
    graph_2410.c   ERGM change statistics and the tie-no-tie sampler
    makefile       `make all` builds hmc; `make test` runs the verification battery

Values from 01 and 02, already in `hmc.c`:

    MAP          (-2.4783349784476716, 0.4694443750091412)
    mass matrix  { 26.020460921843735, 108.351743486973660,
                  108.351743486973660, 492.502589178357425 }
    MPLE         (-3.0385268905675438, 0.6771034983042240)

Priors: `normal` is N(0, diag(10)) per dimension, `uniform` is
[-4, 2] x [-0.05, 1], `ndg` takes strength p through `ndg_p` and the warm-start
pool through `ndg_file`.

`data/ndg_prior.txt` has a row count on line 1, then one row per network:
`s1 s2`, the edge count and the two-star count, exactly as
`03_build_ndg_prior.R` writes them. The C prior derives the hull membership
itself and tracks its own reference parameter, so neither is stored.
