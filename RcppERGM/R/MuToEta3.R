##' Fast Estimation of natural parameter from mean-value parameter Using L-BFGS Algorithm
##'
##' \code{MuToEta()} uses the L-BFGS algorithm to efficiently map a mean-value to
##' the matching natural parameter
##' It is in fact an application of the C++ function
##' \code{optim_lbfgs()} provided by \pkg{RcppNumerical} to perform L-BFGS
##' optimization.
##'
##' @param mv The vector of target mean-values
##' @param gy The matrix of statistics in the model.
##' @param freq The vector of frequencies of the rows of gy in the graph space, thatis, the number of graphs with those statistics.
##' @param start The initial guess of the coefficient vector.
##' @param eps_f Iteration stops if \eqn{|f-f'|/|f|<\epsilon_f}{|f-f'|/|f|<eps_f},
##'              where \eqn{f} and \eqn{f'} are the current and previous value
##'              of the objective function (negative log likelihood) respectively.
##' @param eps_g Iteration stops if
##'              \eqn{||g|| < \epsilon_g * \max(1, ||\beta||)}{||g|| < eps_g * max(1, ||beta||)},
##'              where \eqn{\beta}{beta} is the current coefficient vector and
##'              \eqn{g} is the gradient.
##' @param maxit Maximum number of iterations.
##'
##' @return \code{MuToEta()} returns a list with the following components:
##' \item{coefficients}{Coefficient vector}
##' \item{sumofsquares}{The minimized Euclidean distance between the mean-value based on eta and the target. Thisshould be zero.}
##' \item{converged}{Whether the optimization algorithm has converged}
##'
##' @author Mark S. Handcock 
##'
##' @seealso \code{\link[stats]{glm.fit}()}
##'
##' @export
##'
##' @keywords models
##' @keywords regression
##'
##' @examples
##' set.seed(123)
##' n = 1000
##' p = 100
##' x = matrix(rnorm(n * p), n)
##' beta = runif(p)
##' xb = c(x %*% beta)
##' p = 1 / (1 + exp(-xb))
##' y = rbinom(n, 1, p)
##'
##' system.time(res1 <- glm.fit(x, y, family = binomial()))
##' system.time(res2 <- fastLR(x, y))
##' max(abs(res1$coefficients - res2$coefficients))
MuToEta3 <- function(mv, gy, freq, start = rep(0, ncol(gy)),
                    eps_f = 1e-8, eps_g = 1e-5, maxit = 300)
{
    MuToEta3_(gy, freq, mv, start, eps_f, eps_g, maxit)
}
#' @export
PEta3 <- function(eta, gy, freq)
{
    PEta3_(gy, freq, eta)
}
#' @export
PND3 <- function(eta, gy, freq)
{
    PND3_(gy, freq, eta)
}
#' @export
EtaToMu3 <- function(eta, gy, freq)
{
    EtaToMu3_(gy, freq, eta)
}
