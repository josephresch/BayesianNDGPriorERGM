#include <RcppERGM.h>

typedef Eigen::Map<Eigen::MatrixXd> MapMat;
typedef Eigen::Map<Eigen::VectorXd> MapVec;

class fastMuToEta: public Numer::MFuncGrad
{
private:
    const MapMat GY;
    const MapVec FREQ;
    const MapVec TARGET;
    const int n;
    Eigen::VectorXd expgyeta;  // contains
    Eigen::VectorXd egy;   // contains 
public:
    fastMuToEta(const MapMat gy_, const MapVec freq_, const MapVec target_) :
        GY(gy_),
        FREQ(freq_),
        TARGET(target_),
        n(GY.rows()),
        expgyeta(n),
        egy(2)
    {}

    double f_grad(Numer::Constvec& eta, Numer::Refvec grad)
    {
        // Euclidean distance
        expgyeta.noalias() =(GY * eta);
        double maxexp = -10000000.0;
        for(int i = 0; i < n; i++) {
          if(expgyeta[i] > maxexp) maxexp = expgyeta[i];
        }
        double ceta = 0.0;
        for(int i = 0; i < n; i++) {
          expgyeta[i] = exp(expgyeta[i]-maxexp);
          ceta += expgyeta[i]*FREQ(i);
        }
        egy[0] = 0.0;
        egy[1] = 0.0;
        double co = 0.0;
        double cov00 = 0.0;
        double cov01 = 0.0;
        double cov11 = 0.0;
        for(int i = 0; i < n; i++) {
          co = expgyeta[i] * FREQ(i);
          egy[0] += GY(i,0) * co;
          egy[1] += GY(i,1) * co;
          cov00  += GY(i,0)*GY(i,0) * co;
          cov01  += GY(i,0)*GY(i,1) * co;
          cov11  += GY(i,1)*GY(i,1) * co;
        }
        egy[0] /= ceta;
        egy[1] /= ceta;
        cov00 = cov00 / ceta - egy(0)*egy(0);
        cov01 = cov01 / ceta - egy(0)*egy(1);
        cov11 = cov11 / ceta - egy(1)*egy(1);

        const double f = (egy[0] - TARGET(0))*(egy[0] - TARGET(0)) + (egy[1] - TARGET(1))*(egy[1] - TARGET(1));

        // Gradient
        grad(0) = 2.0 * (egy[0] - TARGET(0))*cov00 + (egy[1] - TARGET(1))*cov01;
        grad(1) = 2.0 * (egy[0] - TARGET(0))*cov01 + (egy[1] - TARGET(1))*cov11;

        return f;
    }
};

// [[Rcpp::export]]
Rcpp::List MuToEta_(Rcpp::NumericMatrix gy, Rcpp::NumericVector freq, Rcpp::NumericVector target,
                    Rcpp::NumericVector start,
                    double eps_f, double eps_g, int maxit
                   )
{
    const MapMat gyx = Rcpp::as<MapMat>(gy);
    const MapVec freqx = Rcpp::as<MapVec>(freq);
    const MapVec targetx = Rcpp::as<MapVec>(target);
    // Euclidean distance from the target
    fastMuToEta nll(gyx, freqx, targetx);
    // Initial guess
    Rcpp::NumericVector b = Rcpp::clone(start);
    MapVec eta(b.begin(), b.length());

    double fopt;
    int status = optim_lbfgs(nll, eta, fopt, maxit, eps_f, eps_g);
    if(status < 0)
        Rcpp::warning("algorithm did not converge");

    return Rcpp::List::create(
        Rcpp::Named("eta")      = b,
        Rcpp::Named("Euclidean distance")= fopt,
        Rcpp::Named("converged")         = (status >= 0)
    );
}
// [[Rcpp::export]]
Rcpp::List PEta_(Rcpp::NumericMatrix gy, Rcpp::NumericVector freq, Rcpp::NumericVector eta)
{
    unsigned int n=static_cast<unsigned int>(gy.rows());
    Rcpp::NumericVector expgyeta(n);
    double tmp = 0.0;
    double maxexp = -10000000.0;
    for(int i = 0; i < n; i++) {
      tmp = gy(i,0)*eta[0] + gy(i,1)*eta[1];
      if(tmp > maxexp) maxexp = tmp;
      expgyeta[i] = tmp;
    }
    double ceta = 0.0;
    for(int i = 0; i < n; i++) {
      expgyeta[i] = exp(expgyeta[i]-maxexp);
      ceta += expgyeta[i]*freq[i];
    }
    for(int i = 0; i < n; i++) {
      expgyeta[i] /= ceta;
    }
    return Rcpp::List::create(
        Rcpp::Named("probability")      = expgyeta
    );
}
// [[Rcpp::export]]
Rcpp::List PND_(Rcpp::NumericMatrix gy, Rcpp::NumericVector freq, Rcpp::NumericVector eta)
{
    unsigned int n=static_cast<unsigned int>(gy.rows());
    Rcpp::NumericVector expgyeta(n);
    double tmp = 0.0;
    double maxexp = -10000000.0;
    for(int i = 0; i < n; i++) {
      tmp = gy(i,0)*eta[0] + gy(i,1)*eta[1];
      if(tmp > maxexp) maxexp = tmp;
      expgyeta[i] = tmp;
    }
    double ceta = 0.0;
    for(int i = 0; i < n; i++) {
      expgyeta[i] = exp(expgyeta[i]-maxexp);
      ceta += expgyeta[i]*freq[i];
    }
    tmp = expgyeta[0]*freq[0]+expgyeta[1]*freq[1]+expgyeta[2]*freq[2]+expgyeta[3]*freq[3]
         +expgyeta[6]*freq[6]+expgyeta[11]*freq[11]+expgyeta[16]*freq[16]
         +expgyeta[22]*freq[22]+expgyeta[33]*freq[33]+expgyeta[44]*freq[44]
         +expgyeta[54]*freq[54]+expgyeta[66]*freq[66]+expgyeta[79]*freq[79]
         +expgyeta[91]*freq[91]+expgyeta[101]*freq[101]+expgyeta[113]*freq[113]
         +expgyeta[122]*freq[122]+expgyeta[130]*freq[130]+expgyeta[136]*freq[136]
         +expgyeta[140]*freq[140]+expgyeta[142]*freq[142]+expgyeta[143]*freq[143];
    tmp /= ceta;
    return Rcpp::List::create(
        Rcpp::Named("probability")      = 1.0 - tmp
    );
}
// [[Rcpp::export]]
Rcpp::List EtaToMu_(Rcpp::NumericMatrix gy, Rcpp::NumericVector freq, Rcpp::NumericVector eta)
{
    unsigned int n=static_cast<unsigned int>(gy.rows());
    Rcpp::NumericVector mv(2);
    Rcpp::NumericVector expgyeta(n);
    double tmp = 0.0;
    double maxexp = -10000000.0;
    for(int i = 0; i < n; i++) {
      tmp = gy(i,0)*eta[0] + gy(i,1)*eta[1];
      if(tmp > maxexp) maxexp = tmp;
      expgyeta[i] = tmp;
    }
    double ceta = 0.0;
    mv[0]=0.0;
    mv[1]=0.0;
    for(int i = 0; i < n; i++) {
      expgyeta[i] = exp(expgyeta[i]-maxexp);
      tmp = expgyeta[i]*freq[i];
      mv[0] += gy(i,0)*tmp;
      mv[1] += gy(i,1)*tmp;
      ceta += tmp;
    }
    mv[0] /= ceta;
    mv[1] /= ceta;
// Rprintf("%f %f \n", mv[0], mv[1]);
    return Rcpp::List::create(
        Rcpp::Named("mean-value") = mv
    );
}
