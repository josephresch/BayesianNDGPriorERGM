#
#   test_prior.R
#
#   Independent R cross-check of the adaptive NDG prior implementation in
#   hull.c + prior.c. Reads /tmp/test_prior_dump.txt (produced by
#   test_prior.c), recomputes the convex hull, the self-normalised IS
#   estimator of Phat, the analytical gradient of log Phat, and
#   numDeriv::grad applied to the pure-R log-density, then compares
#   everything to the C values at tight tolerance.
#
#   This script has NO access to any ground-truth ERGM enumeration --
#   it consumes only the dumped pool and (theta, tref, p) metadata.
#

suppressPackageStartupMessages({
    library(numDeriv)
})

TOL_TIGHT <- 1e-10
TOL_FD    <- 1e-6

# ---------------------------------------------------------------------
#   Pass/fail bookkeeping
# ---------------------------------------------------------------------
n_pass <- 0L
n_fail <- 0L
rec <- function(tag, ok) {
    if (isTRUE(ok)) {
        n_pass <<- n_pass + 1L
        cat(sprintf("  [PASS] %s\n", tag))
    } else {
        n_fail <<- n_fail + 1L
        cat(sprintf("  [FAIL] %s\n", tag))
    }
}
close_abs <- function(got, want, tol = TOL_TIGHT) {
    all(is.finite(c(got, want))) && max(abs(got - want)) <= tol
}

# ---------------------------------------------------------------------
#   Parse the C dump
# ---------------------------------------------------------------------
dump_path <- "/tmp/test_prior_dump.txt"
stopifnot(file.exists(dump_path))
lines <- readLines(dump_path)

stopifnot(lines[[1]] == "POOL")
meta <- as.numeric(strsplit(lines[[2]], "\\s+")[[1]])
m        <- as.integer(meta[1])
p_str    <- meta[2]
tref     <- c(meta[3], meta[4])
pool     <- matrix(NA_integer_, nrow = m, ncol = 2L)
for (i in seq_len(m)) {
    r <- as.integer(strsplit(lines[[2 + i]], "\\s+")[[1]])
    pool[i, ] <- r
}

row_hull <- 2L + m + 1L
stopifnot(lines[[row_hull]] == "HULL")
nv <- as.integer(lines[[row_hull + 1L]])
c_hull <- matrix(NA_integer_, nrow = nv, ncol = 2L)
for (i in seq_len(nv)) {
    r <- as.integer(strsplit(lines[[row_hull + 1L + i]], "\\s+")[[1]])
    c_hull[i, ] <- r
}

row_thetas <- row_hull + 1L + nv + 1L
stopifnot(lines[[row_thetas]] == "THETAS")
n_theta <- as.integer(lines[[row_thetas + 1L]])
theta_mat <- matrix(NA_real_, nrow = n_theta, ncol = 2L)
c_lp      <- numeric(n_theta)
c_grad    <- matrix(NA_real_, nrow = n_theta, ncol = 2L)
for (k in seq_len(n_theta)) {
    r <- as.numeric(strsplit(lines[[row_thetas + 1L + k]], "\\s+")[[1]])
    theta_mat[k, ] <- r[1:2]
    c_lp[k]        <- r[3]
    c_grad[k, ]    <- r[4:5]
}

cat(sprintf(
    "R cross-check: m=%d, nv=%d, n_theta=%d, p_strength=%.3f, tref=(%.3f,%.3f)\n",
    m, nv, n_theta, p_str, tref[1], tref[2]))

# ---------------------------------------------------------------------
#   TEST A: hull cross-check via R's chull()
# ---------------------------------------------------------------------
#
#   Important: the C hull was built from the union of the seed square
#     {(0,0),(4,0),(4,4),(0,4)}
#   AND the pool. We have to replicate that union in R or the chull()
#   result will be a subset of the C hull.
#
cat("\n=== TEST A: R chull() vs C monotone-chain hull ===\n")

seed_square <- rbind(c(0,0), c(4,0), c(4,4), c(0,4))
all_points  <- unique(rbind(seed_square, pool))

r_idx <- chull(all_points[, 1], all_points[, 2])  # CCW order from R
r_hull <- all_points[r_idx, , drop = FALSE]

# chull() may return a non-strict hull that contains collinear edge
# midpoints. Filter them: a point is a "corner" iff it is not collinear
# with its two neighbours.
cross2d <- function(o, a, b) {
    (a[1] - o[1]) * (b[2] - o[2]) -
    (a[2] - o[2]) * (b[1] - o[1])
}
nrh <- nrow(r_hull)
keep <- rep(TRUE, nrh)
for (i in seq_len(nrh)) {
    prev <- r_hull[ifelse(i == 1, nrh, i - 1), ]
    cur  <- r_hull[i, ]
    nxt  <- r_hull[ifelse(i == nrh, 1, i + 1), ]
    if (abs(cross2d(prev, cur, nxt)) < .Machine$double.eps) {
        keep[i] <- FALSE
    }
}
r_hull <- r_hull[keep, , drop = FALSE]

# Canonicalise both hulls: cast to numeric, sort lex by (s1, s2).
# Order around the polygon differs between implementations (C is CCW,
# R's chull() is clockwise), but the unordered vertex set must agree.
canon <- function(M) {
    M2 <- matrix(as.numeric(M), ncol = 2L)
    o <- order(M2[, 1], M2[, 2])
    M2[o, , drop = FALSE]
}

rec("C hull vertex count == R chull() strict-corner count",
    nrow(c_hull) == nrow(r_hull))
rec("C hull vertex set == R chull() vertex set",
    isTRUE(all.equal(canon(c_hull), canon(r_hull))))

# ---------------------------------------------------------------------
#   TEST B: pure-R Eq. 14 density vs C
# ---------------------------------------------------------------------
#
#   We have to re-derive in_hull[] in R. A lattice point g is in the
#   STRICT interior of the CCW polygon iff every directed edge sees g
#   strictly on its left. Build this as a helper that takes a 2-col
#   hull matrix (CCW order).
#
cat("\n=== TEST B: pure-R Eq. 14 Phat vs C log density ===\n")

# Reorder c_hull into CCW order (the C implementation already emits
# CCW from the monotone chain; we assume it matches). Use the C hull
# as the authoritative polygon.
is_strict_interior <- function(poly, px, py) {
    k <- nrow(poly)
    if (k < 3) return(FALSE)
    for (i in seq_len(k)) {
        j <- if (i == k) 1L else i + 1L
        cp <-
            (poly[j, 1] - poly[i, 1]) * (py - poly[i, 2]) -
            (poly[j, 2] - poly[i, 2]) * (px - poly[i, 1])
        if (cp <= 0) return(FALSE)
    }
    TRUE
}

in_hull_R <- vapply(seq_len(m), function(i) {
    is_strict_interior(c_hull, pool[i, 1], pool[i, 2])
}, logical(1))

# Pure-R self-normalised IS estimator of log Phat.
log_phat_R <- function(theta) {
    d <- theta - tref
    z <- as.numeric(pool %*% d)
    zmax <- max(z)
    w <- exp(z - zmax)
    N <- sum(w[in_hull_R])
    D <- sum(w)
    if (N <= 0 || D <= 0) return(-Inf)
    log(N) - log(D)
}
log_p_R <- function(theta) p_str * log_phat_R(theta)

lp_R <- vapply(seq_len(n_theta), function(k) log_p_R(theta_mat[k, ]),
               numeric(1))

rec("log density matches C for all thetas",
    close_abs(lp_R, c_lp))

# ---------------------------------------------------------------------
#   TEST C: analytical gradient matches both C and numDeriv::grad
# ---------------------------------------------------------------------
cat("\n=== TEST C: analytical gradient vs C and numDeriv::grad ===\n")

grad_log_p_R <- function(theta) {
    d <- theta - tref
    z <- as.numeric(pool %*% d)
    zmax <- max(z)
    w <- exp(z - zmax)
    D    <- sum(w)
    N    <- sum(w[in_hull_R])
    if (N <= 0 || D <= 0) return(c(0, 0))
    Dg0  <- sum(w * pool[, 1])
    Dg1  <- sum(w * pool[, 2])
    Ng0  <- sum(w[in_hull_R] * pool[in_hull_R, 1])
    Ng1  <- sum(w[in_hull_R] * pool[in_hull_R, 2])
    p_str * c(Ng0 / N - Dg0 / D, Ng1 / N - Dg1 / D)
}

g_R <- t(vapply(seq_len(n_theta),
                function(k) grad_log_p_R(theta_mat[k, ]),
                numeric(2)))

rec("analytical R gradient matches C gradient",
    close_abs(g_R, c_grad))

# numDeriv::grad applied to the PURE-R log_p_R. If this matches the
# analytical gradient we've triple-verified: C analytical = R analytical
# = R finite difference.
g_nd <- t(vapply(seq_len(n_theta),
                 function(k) grad(log_p_R, theta_mat[k, ]),
                 numeric(2)))

rec("numDeriv::grad on pure-R log_p matches C gradient",
    close_abs(g_nd, c_grad, TOL_FD))
rec("numDeriv::grad on pure-R log_p matches R analytical gradient",
    close_abs(g_nd, g_R, TOL_FD))

# ---------------------------------------------------------------------
cat(sprintf("\n=== R cross-check summary: %d passed, %d failed ===\n",
            n_pass, n_fail))
if (n_fail > 0L) {
    quit(status = 1L)
} else {
    quit(status = 0L)
}
