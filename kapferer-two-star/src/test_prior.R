#
#   test_prior.R
#
#   Independent R cross-check of the adaptive NDG prior implementation in
#   hull3d.c + prior.c (3D). Reads /tmp/test_prior_dump.txt (produced by
#   test_prior.c), cross-checks the 3D convex hull against
#   geometry::convhulln, recomputes the self-normalised IS estimator of
#   Phat and its analytical gradient in pure R, and verifies
#   numDeriv::grad on the pure-R log-density as a finite-difference
#   sanity check.
#
#   This script has NO access to any ground-truth ERGM enumeration --
#   it consumes only the dumped pool, hull, and (theta, tref, p) metadata.
#

suppressPackageStartupMessages({
    library(numDeriv)
    library(geometry)
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
cur <- 1L

read_tokens <- function(line) {
    as.numeric(strsplit(trimws(line), "\\s+")[[1]])
}

stopifnot(lines[[cur]] == "POOL"); cur <- cur + 1L
meta <- read_tokens(lines[[cur]]); cur <- cur + 1L
m        <- as.integer(meta[1])
p_str    <- meta[2]
tref     <- c(meta[3], meta[4], meta[5])
pool     <- matrix(NA_real_, nrow = m, ncol = 3L)
for (i in seq_len(m)) {
    pool[i, ] <- read_tokens(lines[[cur]]); cur <- cur + 1L
}

stopifnot(lines[[cur]] == "HULL_OBS"); cur <- cur + 1L
n_obs <- as.integer(lines[[cur]]); cur <- cur + 1L
obs <- matrix(NA_real_, nrow = n_obs, ncol = 3L)
for (i in seq_len(n_obs)) {
    obs[i, ] <- read_tokens(lines[[cur]]); cur <- cur + 1L
}

stopifnot(lines[[cur]] == "HULL_VERT"); cur <- cur + 1L
n_vert <- as.integer(lines[[cur]]); cur <- cur + 1L
# C-side indices are 0-based.
c_vert_idx0 <- integer(n_vert)
for (i in seq_len(n_vert)) {
    c_vert_idx0[i] <- as.integer(lines[[cur]]); cur <- cur + 1L
}

stopifnot(lines[[cur]] == "HULL_FACES"); cur <- cur + 1L
n_face <- as.integer(lines[[cur]]); cur <- cur + 1L
c_faces0 <- matrix(NA_integer_, nrow = n_face, ncol = 3L)
for (i in seq_len(n_face)) {
    c_faces0[i, ] <- as.integer(strsplit(trimws(lines[[cur]]), "\\s+")[[1]])
    cur <- cur + 1L
}

stopifnot(lines[[cur]] == "THETAS"); cur <- cur + 1L
n_theta <- as.integer(lines[[cur]]); cur <- cur + 1L
theta_mat <- matrix(NA_real_, nrow = n_theta, ncol = 3L)
c_lp      <- numeric(n_theta)
c_grad    <- matrix(NA_real_, nrow = n_theta, ncol = 3L)
for (k in seq_len(n_theta)) {
    r <- read_tokens(lines[[cur]]); cur <- cur + 1L
    theta_mat[k, ] <- r[1:3]
    c_lp[k]        <- r[4]
    c_grad[k, ]    <- r[5:7]
}

cat(sprintf(
    "R cross-check: m=%d, n_obs=%d, n_vert=%d, n_face=%d, n_theta=%d, p=%.3f\n",
    m, n_obs, n_vert, n_face, n_theta, p_str))

# ---------------------------------------------------------------------
#   TEST A: hull vertex set vs geometry::convhulln
# ---------------------------------------------------------------------
cat("\n=== TEST A: R convhulln vs C hull3d ===\n")

# geometry::convhulln returns a matrix of triangular face indices (1-based).
# The unique vertices of that face matrix are the hull vertices.
r_face_idx1 <- geometry::convhulln(obs)
r_vert_idx1 <- sort(unique(as.integer(r_face_idx1)))

c_vert_idx1 <- sort(c_vert_idx0 + 1L)
rec("C hull vertex count == convhulln vertex count",
    length(c_vert_idx1) == length(r_vert_idx1))
rec("C hull vertex set == convhulln vertex set",
    identical(c_vert_idx1, r_vert_idx1))

# ---------------------------------------------------------------------
#   TEST B: pure-R Eq.14 Phat vs C log density
# ---------------------------------------------------------------------
cat("\n=== TEST B: pure-R Eq.14 log density vs C ===\n")

# Build per-face outward-facing plane equations from the C-side face list
# so our strict-interior test mirrors the C implementation semantically.
# The polytope is convex, so the centroid of its vertex coordinates is
# strictly interior and can be used as an orientation reference.
vert_xyz <- obs[c_vert_idx0 + 1L, , drop = FALSE]
ref_interior <- colMeans(vert_xyz)

# For each face (a,b,c), compute unit normal and signed offset d = n.v
# oriented outward (dot(n, ref_interior) - d < 0).
face_planes <- matrix(NA_real_, nrow = n_face, ncol = 4L)  # nx ny nz d
for (j in seq_len(n_face)) {
    idx <- c_faces0[j, ] + 1L
    A <- obs[idx[1], ]
    B <- obs[idx[2], ]
    C <- obs[idx[3], ]
    n <- c(
        (B[2] - A[2]) * (C[3] - A[3]) - (B[3] - A[3]) * (C[2] - A[2]),
        (B[3] - A[3]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[3] - A[3]),
        (B[1] - A[1]) * (C[2] - A[2]) - (B[2] - A[2]) * (C[1] - A[1])
    )
    nlen <- sqrt(sum(n * n))
    stopifnot(nlen > 0)
    n <- n / nlen
    d <- sum(n * A)
    # Flip if interior reference is on the outward side.
    if (sum(n * ref_interior) - d > 0) {
        n <- -n
        d <- -d
    }
    face_planes[j, ] <- c(n, d)
}

# Strict interior iff signed_dist = n.p - d < -eps for EVERY face.
# Use the same default eps the C hull uses (1e-9) so classifications
# align bit-for-bit.
eps_strict <- 1e-9
is_strict_interior_R <- function(px, py, pz) {
    all((face_planes[, 1] * px
       + face_planes[, 2] * py
       + face_planes[, 3] * pz
       - face_planes[, 4]) < -eps_strict)
}

in_hull_R <- vapply(seq_len(m), function(i)
    is_strict_interior_R(pool[i, 1], pool[i, 2], pool[i, 3]),
    logical(1))

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
#   TEST C: analytical gradient vs C and numDeriv::grad
# ---------------------------------------------------------------------
cat("\n=== TEST C: analytical gradient vs C and numDeriv::grad ===\n")

grad_log_p_R <- function(theta) {
    d <- theta - tref
    z <- as.numeric(pool %*% d)
    zmax <- max(z)
    w <- exp(z - zmax)
    D <- sum(w)
    N <- sum(w[in_hull_R])
    if (N <= 0 || D <= 0) return(c(0, 0, 0))
    c(
        p_str * (sum(w[in_hull_R] * pool[in_hull_R, 1]) / N -
                 sum(w              * pool[, 1])         / D),
        p_str * (sum(w[in_hull_R] * pool[in_hull_R, 2]) / N -
                 sum(w              * pool[, 2])         / D),
        p_str * (sum(w[in_hull_R] * pool[in_hull_R, 3]) / N -
                 sum(w              * pool[, 3])         / D)
    )
}

g_R <- t(vapply(seq_len(n_theta),
                function(k) grad_log_p_R(theta_mat[k, ]),
                numeric(3)))
rec("analytical R gradient matches C gradient",
    close_abs(g_R, c_grad))

g_nd <- t(vapply(seq_len(n_theta),
                 function(k) grad(log_p_R, theta_mat[k, ]),
                 numeric(3)))
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
