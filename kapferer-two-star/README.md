# Kapferer, two-star model

Paper Section 7.2. Kapferer's tailor shop network, 39 nodes and 158 undirected
edges, model `edges + kstar(2) + gwesp(0.25, fixed)` (model code `esg` in the
C driver).

Same network and same pipeline as `kapferer-gwesp-gwdsp/`, with the
unweighted two-star count in place of GWDSP. That one substitution reintroduces
a degeneracy-prone statistic, and it is the setting where the non-degeneracy
prior makes a measurable difference.

Under the normal prior the posterior predictive interquartile range for edges
runs from 17 to 688 out of a possible 741, so a quarter of the simulated
networks sit near the complete graph. At p = 1 that range narrows to [14, 236].
The theta-space posterior barely moves, an edges coefficient of -4.16 against
-4.17, which is why point estimates would show nothing.

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
    tuning/    MAP and mass matrix from steps 01 and 02
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

Observed statistics: edges 158, kstar(2) 1566, gwesp(0.25, fixed) 185.7892.

The warm-start pool `data/ndg_prior.txt` is not shipped with this folder. Run
`03_build_ndg_prior.R` to generate it before running 04 or 05 with
`prior_type = "ndg"`; it writes 100,000 rows of (edges, kstar2, gwesp).

The convex hull is estimated adaptively in three dimensions, seeded from the
warm-start pool and grown during sampling. This is the constraint behind the
paper's recommendation of p = 1 here against p in [5, 10] for the n = 7 study,
where the hull is known exactly. At p = 5 the predictive spread widens back
past the uniform prior's, and by p = 10 it is indistinguishable from the
normal prior's.

## Results

    posterior-20rep-<prior>.RDS   20 chains, pooled. Table 4.
    gof-20rep-<prior>.png         four-panel goodness-of-fit. Figure 3.
    gof-20rep-<prior>-<panel>.png the same panels saved individually
    trace-20rep-<prior>-*.png     marginal densities and pairwise hex plots
    prior-sensitivity-20rep.png   marginal densities overlaid across priors
    cross-prior-gof.png           goodness-of-fit compared across priors
    degeneracy-diagnostics.png    near-empty and near-complete summaries
    scalar-summaries.png          scalar posterior summaries by prior

`<prior>` is `normal`, `uniform`, or `ndg-p<p>` with p in {1, 5, 10}. Figure 3
in the paper uses the normal, uniform, `ndg-p1` and `ndg-p10` panels. The last
three figures above are diagnostics that did not make it into the paper.

## Sampler

Adapted from Stoehr, Benson and Friel (2019). The prior module and the hull
code were added for this paper. Identical to the sampler in
`kapferer-gwesp-gwdsp/` apart from the change statistics and the compiled
mass matrix.

    hmc.c            noisy leapfrog gradient and the leapfrog estimator of Z ratios
    prior.c          normal, uniform, and non-degeneracy priors behind one interface
    hull3d.c         3D convex hull, grown as auxiliary statistics arrive
    graph_2410.c     ERGM change statistics and the tie-no-tie sampler
    test_changestats.c  full-graph statistics, cross-checked by verify_changestats.R

Values from 01 and 02, already in `hmc.c` and under `tuning/`:

    MAP          (-1.93315089722051, 0.0279204041483374, 0.150863778607253)
    mass matrix  (row-major, 3x3)
         177.308012024048   3064.39038476954    309.130641439083
        3064.390384769540  54491.37983567130   5411.603185022940
         309.130641439083   5411.60318502294    574.033817648792

Priors: `normal` is the Caimo and Friel default N(0, diag(100)) per dimension;
`uniform` has bounds from MCMLE +/- 10 SE; `ndg` takes strength p through
`ndg_p` and the warm-start pool through `ndg_file`.

`data/ndg_prior.txt` has a row count on line 1, then one row per network:
`s1 s2 s3` for edges, kstar2 and gwesp. No hull or reference parameter is
written. The prior builds its own 3D hull as sampling proceeds.

GWESP change statistics use a hardcoded alpha of 0.25 in `graph_2410.c`. The
warm-start pool is simulated at the MPLE rather than the MAP, so the prior does
not depend on the posterior workflow it feeds.
