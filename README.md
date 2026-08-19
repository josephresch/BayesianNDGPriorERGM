# Bayesian Inference for ERGMs with Non-degeneracy Priors

Code and data for Resch and Handcock, *Bayesian Inference for
Exponential-Family Random Graph Models with Non-degeneracy Priors*.

The paper develops a noisy Hamiltonian Monte Carlo sampler for the ERGM
posterior under the non-degeneracy prior, approximates that prior by importance
reweighting the auxiliary networks the sampler already draws, and evaluates the
result on two networks. The propriety and moment theory for the prior is in the
companion paper, Resch and Handcock (2026).

Every directory corresponds to one study in the paper.

| Directory | Network and model |
|---|---|
| `two-star-n7/` | 7 nodes, `edges + kstar(2)` |
| `kapferer-gwesp-gwdsp/` | Kapferer, `edges + gwesp(0.25) + gwdsp(0.25)` |
| `kapferer-two-star/` | Kapferer, `edges + kstar(2) + gwesp(0.25)` |

The n = 7 study is the only one where the posterior can be computed exactly, so
it is the only place the approximation is checked against a ground truth. The
two Kapferer studies are a matched pair on the same network. The first uses a
specification that already controls degeneracy and is the control. The second
substitutes the two-star count, and there the prior changes the answer.

## NoisyHMC Sampler

The sampler follows Stoehr, Benson and Friel (2019), whose implementation is
written in C against GSL. Each `src/` starts from that code and stays in C.
`hmc.c`, `hmc_driver.c` and `graph_2410.c` are theirs with modifications;
`prior.c` and the hull code were written for this paper in the same language.

| Paper | Code |
|---|---|
| Noisy leapfrog gradient (Sec. 3) | `src/hmc.c`, `get_loggrad` |
| Leapfrog estimator of the normalizing-constant ratio (Sec. 3) | `src/hmc.c`, `get_logalphahmc_massmatrix` |
| Non-degeneracy probability (Sec. 4.1) | `src/prior.c` |
| Support-balance family, strength p (Sec. 4.2) | `src/prior.c`, `ndg_p` argument to `hmc` |
| Self-normalized importance-reweighting estimator (Sec. 6.1) | `src/prior.c` |
| Adaptive convex hull (Sec. 6.2) | `src/hull.c` in 2D, `src/hull3d.c` in 3D |
| Warm-start pool at the MPLE (Sec. 6.2) | `03_build_ndg_prior.R` |
| Robbins-Monro mode search (Sec. 6.2) | `01_find_map.R` |
| Mass matrix from the auxiliary covariance (Sec. 6.2) | `02_find_mass_matrix.R` |
| Dual averaging on eps and L (Sec. 6.2) | `src/hmc.c`, warmup block |
| 20 replicate chains of 100,000 iterations | `05_run_hmc_replicates.R` |

The reference parameter is reset to the current leapfrog point at every step, which is what
makes the normalizing constants cancel in the self-normalized ratio. Auxiliary
draws therefore cannot be pooled across steps, so the number of draws N alone
controls the quality of the prior approximation. `prior.c` enforces this rather
than accumulating a growing pool.

## Layout

All three directories share a structure.

    01_find_map.R              posterior mode, Robbins-Monro
    02_find_mass_matrix.R      HMC mass matrix at the mode
    03_build_ndg_prior.R       warm-start pool for the non-degeneracy prior
    04_run_hmc_single.R        one chain, with reporting
    05_run_hmc_replicates.R    20 chains, the protocol the paper reports
    ergm_helpers.R             shared simulation and prior-gradient helpers
    verify_changestats.R       C change statistics vs ergm::summary (Kapferer only)

    data/      network file and the generated warm-start pool
    src/       C sampler and makefile
    tuning/    mode and mass matrix from steps 01 and 02 (Kapferer only)
    chains/    raw text from the C binary, overwritten on each run
    results/   the posteriors and figures the paper reports

Run the R scripts from inside their own directory. Scripts are numbered in
execution order. Steps 01 and 02 only need rerunning if the network or the
model changes; their answers are compiled into `src/hmc.c` and recorded in each
directory's README.

## Reproducing

Requires GSL 2.2+ and R with `ergm`, `network`, `numDeriv`, `MASS`, `ggplot2`
and `gridExtra`; `geometry` and `coda` for the Kapferer directories;
`spatstat.geom` and `spatstat.explore` for the n = 7 overlay figure.

`two-star-n7/` additionally needs **RcppERGM**, which is not on CRAN. Scripts 04
and 05 call `RcppERGM::EtaToMu` and `RcppERGM::PND` to map the sampled natural
parameters into mean-value space and to evaluate the exact non-degeneracy
probability. Every mu-space quantity in Table 1 and Figure 1 goes through it.
Its source ships at the repository root, so install it first:

    R CMD INSTALL RcppERGM

It needs `Rcpp` and `RcppEigen`, both on CRAN. `RcppERGM_0.6-0.tar.gz` is the
same package prebuilt if you would rather use `install.packages(..., repos = NULL)`.

    cd two-star-n7/src && make all
    make test
    cd .. && Rscript 03_build_ndg_prior.R
    Rscript 05_run_hmc_replicates.R

`prior_type` and `ndg_p` sit at the top of scripts 04 and 05, and are set a
second time in the reporting section further down, which reads a stored run so
tables and figures can be rebuilt without re-sampling. Seeds are fixed.
The C driver takes an explicit seed per replicate, derived from `master_seed`
in script 05, so a rerun on the same machine reproduces the stored chains.
Per-replicate wall-clock times are recorded in `chains/*_time.txt` from the runs
behind the paper.

The binaries checked in under `src/` were built on macOS ARM and will need
rebuilding elsewhere. `make all` is the only build step.

## Verification

Each directory carries checks that do not touch the main workflow.

    */verify_changestats.R              ergm::summary vs the C change statistics
    src/test_prior.c and test_prior.R   run together by `make test`

For the Kapferer directories `make test` should report 41 C assertions and 6 R
cross-checks passing.

## References

Resch, J. and Handcock, M. S. (2026). Non-degeneracy priors for
exponential-family random graph models. Submitted.
Replication material: https://github.com/josephresch/NondegeneracyPriorERGM

Stoehr, J., Benson, A. and Friel, N. (2019). Noisy Hamiltonian Monte Carlo for
doubly intractable distributions. *Journal of Computational and Graphical
Statistics* 28(1), 220-232.

Kapferer, B. (1972). *Strategy and Transaction in an African Factory*.
Manchester University Press.
