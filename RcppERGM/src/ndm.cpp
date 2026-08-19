#include <RcppERGM.h>

using namespace Numer;

typedef Eigen::Map<Eigen::MatrixXd> MapMat;
typedef Eigen::Map<Eigen::VectorXd> MapVec;

class fastMuToEtaNDM: public MFuncGrad
{
private:
    const MapMat GY;
    const MapVec FREQ;
    const MapVec TARGET;
    const int n;
    Eigen::VectorXd expgyeta;  // contains
    Eigen::VectorXd egy;   // contains 
public:
    fastMuToEtaNDM(const MapMat gy_, const MapVec freq_, const MapVec target_) :
        GY(gy_),
        FREQ(freq_),
        TARGET(target_),
        n(GY.rows()),
        expgyeta(n),
        egy(3)
    {}

    double f_grad(Constvec& eta, Refvec grad)
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
        egy[2] = 0.0;
        double co = 0.0;
        double cov00 = 0.0;
        double cov01 = 0.0;
        double cov11 = 0.0;
        double cov02 = 0.0;
        double cov12 = 0.0;
        double cov22 = 0.0;
        for(int i = 0; i < n; i++) {
          co = expgyeta[i] * FREQ(i);
          egy[0] += GY(i,0) * co;
          egy[1] += GY(i,1) * co;
          egy[2] += GY(i,2) * co;
          cov00  += GY(i,0)*GY(i,0) * co;
          cov01  += GY(i,0)*GY(i,1) * co;
          cov11  += GY(i,1)*GY(i,1) * co;
          cov02  += GY(i,0)*GY(i,2) * co;
          cov12  += GY(i,1)*GY(i,2) * co;
          cov22  += GY(i,2)*GY(i,2) * co;
        }
        egy[0] /= ceta;
        egy[1] /= ceta;
        egy[2] /= ceta;
        cov00 = cov00 / ceta - egy(0)*egy(0);
        cov01 = cov01 / ceta - egy(0)*egy(1);
        cov11 = cov11 / ceta - egy(1)*egy(1);
        cov02 = cov02 / ceta - egy(0)*egy(2);
        cov12 = cov12 / ceta - egy(1)*egy(2);
        cov22 = cov22 / ceta - egy(2)*egy(2);

        const double f = (egy[0] - TARGET(0))*(egy[0] - TARGET(0)) + (egy[1] - TARGET(1))*(egy[1] - TARGET(1)) +
                         (egy[2] - TARGET(2))*(egy[2] - TARGET(2));

        // Gradient
        grad(0) = 2.0 * (egy[0] - TARGET(0))*cov00 + (egy[1] - TARGET(1))*cov01 + (egy[2] - TARGET(2))*cov02;
        grad(1) = 2.0 * (egy[0] - TARGET(0))*cov01 + (egy[1] - TARGET(1))*cov11 + (egy[2] - TARGET(2))*cov12;
        grad(2) = 2.0 * (egy[0] - TARGET(0))*cov02 + (egy[1] - TARGET(1))*cov12 + (egy[2] - TARGET(2))*cov22;

        return f;
    }
};

// [[Rcpp::export]]
Rcpp::List MuToEtaNDM_(Rcpp::NumericMatrix gy, Rcpp::NumericVector freq, Rcpp::NumericVector target,
                    Rcpp::NumericVector start,
                    double eps_f, double eps_g, int maxit
                   )
{
    const MapMat gyx = Rcpp::as<MapMat>(gy);
    const MapVec freqx = Rcpp::as<MapVec>(freq);
    const MapVec targetx = Rcpp::as<MapVec>(target);
    unsigned int n=static_cast<unsigned int>(gy.rows());
    // Euclidean distance from the target
    fastMuToEtaNDM nll(gyx, freqx, targetx);
    // Initial guess
    Rcpp::NumericVector b = Rcpp::clone(start);
    Rcpp::NumericVector expgyeta(n);
    MapVec eta(b.begin(), b.length());

    double fopt;
    int status = optim_lbfgs(nll, eta, fopt, maxit, eps_f, eps_g);
    if(status < 0)
        Rcpp::warning("algorithm did not converge");

    double maxexp = -10000000.0;
    for(int i = 0; i < n; i++) {
  //   expgyeta[i] = gy[i,0]*eta[0] + gy[i,1]*eta[1] + gy[i,2]*eta[2];
       expgyeta[i] = gy(i,0)*eta(0) + gy(i,1)*eta[1] + gy(i,2)*eta[2];
       if(expgyeta[i] > maxexp) maxexp = expgyeta[i];
    }
    double ceta = 0.0;
    for(int i = 0; i < n; i++) {
       expgyeta[i] = exp(expgyeta[i]-maxexp);
       ceta += expgyeta[i]*freq[i];
    }

    double llik = b(0)*target[0] + b(1)*target[1] + b(2)*target[2] - log(ceta);

    return Rcpp::List::create(
        Rcpp::Named("eta")      = b,
        Rcpp::Named("Euclidean distance")= fopt,
        Rcpp::Named("llik")      = llik,
        Rcpp::Named("converged")         = (status >= 0)
    );
}
// [[Rcpp::export]]
Rcpp::List PEtaNDM_(Rcpp::NumericMatrix gy, Rcpp::NumericVector freq, Rcpp::NumericVector eta)
{
    unsigned int n=static_cast<unsigned int>(gy.rows());
    Rcpp::NumericVector expgyeta(n);
    double tmp = 0.0;
    double maxexp = -10000000.0;
    for(int i = 0; i < n; i++) {
      tmp = gy(i,0)*eta[0] + gy(i,1)*eta[1] + gy(i,2)*eta[2] ;
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
Rcpp::List PNDNDM_(Rcpp::NumericMatrix gy, Rcpp::NumericVector freq, Rcpp::NumericVector eta)
{
    unsigned int n=static_cast<unsigned int>(gy.rows());
    Rcpp::NumericVector expgyeta(n);
    double tmp = 0.0;
    double maxexp = -10000000.0;
    for(int i = 0; i < n; i++) {
      tmp = gy(i,0)*eta[0] + gy(i,1)*eta[1] + gy(i,2)*eta[2] ;
      if(tmp > maxexp) maxexp = tmp;
      expgyeta[i] = tmp;
    }
    double ceta = 0.0;
    for(int i = 0; i < n; i++) {
      expgyeta[i] = exp(expgyeta[i]-maxexp);
      ceta += expgyeta[i]*freq[i];
    }
    // First sum over the boundary points for (edges, 2-stars)
    tmp =  expgyeta[0]*freq[0]+expgyeta[1]*freq[1]+expgyeta[2]*freq[2]+expgyeta[3]*freq[3]
          +expgyeta[6]*freq[6]+expgyeta[11]*freq[11]+expgyeta[16]*freq[16]
         +expgyeta[22]*freq[22]+expgyeta[33]*freq[33]+expgyeta[44]*freq[44]
         +expgyeta[54]*freq[54]+expgyeta[66]*freq[66]+expgyeta[79]*freq[79]
         +expgyeta[91]*freq[91]+expgyeta[101]*freq[101]+expgyeta[113]*freq[113]
        +expgyeta[122]*freq[122]+expgyeta[130]*freq[130]+expgyeta[136]*freq[136]
        +expgyeta[140]*freq[140]+expgyeta[142]*freq[142]+expgyeta[143]*freq[143];
    // These are the peeling of the hull - that is the boundary after the 
    // actual boundary (above) has been removed
    // tmp += expgyeta[4]*freq[4]+expgyeta[5]*freq[5]+expgyeta[8]*freq[8]+expgyeta[13]*freq[13]
    //      +expgyeta[19]*freq[19]+expgyeta[25]*freq[25]+expgyeta[36]*freq[36]
    //      +expgyeta[46]*freq[46]+expgyeta[57]*freq[57]+expgyeta[68]*freq[68]
    //      +expgyeta[81]*freq[81]+expgyeta[93]*freq[93]+expgyeta[103]*freq[103]
    //     +expgyeta[114]*freq[114]+expgyeta[123]*freq[123]+expgyeta[131]*freq[131]
    //     +expgyeta[137]*freq[137];
    tmp /= ceta;
    return Rcpp::List::create(
        Rcpp::Named("probability")      = 1.0 - tmp
    );
}
// [[Rcpp::export]]
Rcpp::List EtaToMuNDM_(Rcpp::NumericMatrix gy, Rcpp::NumericVector freq, Rcpp::NumericVector eta)
{
    unsigned int n=static_cast<unsigned int>(gy.rows());
    Rcpp::NumericVector mv(2);
    Rcpp::NumericVector expgyeta(n);
    double tmp = 0.0;
    double maxexp = -10000000.0;
    for(int i = 0; i < n; i++) {
      tmp = gy(i,0)*eta[0] + gy(i,1)*eta[1] + gy(i,2)*eta[2] ;
      if(tmp > maxexp) maxexp = tmp;
      expgyeta[i] = tmp;
    }
    double ceta = 0.0;
    mv[0]=0.0;
    mv[1]=0.0;
    mv[2]=0.0;
    for(int i = 0; i < n; i++) {
      expgyeta[i] = exp(expgyeta[i]-maxexp);
      tmp = expgyeta[i]*freq[i];
      mv[0] += gy(i,0)*tmp;
      mv[1] += gy(i,1)*tmp;
      mv[2] += gy(i,2)*tmp;
      ceta += tmp;
    }
    mv[0] /= ceta;
    mv[1] /= ceta;
    mv[2] /= ceta;
// Rprintf("%f %f \n", mv[0], mv[1]);
    return Rcpp::List::create(
        Rcpp::Named("mean-value") = mv
    );
}
