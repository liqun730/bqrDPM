// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <string>
#include <iostream>
#include <fstream>
//#include <Rcpp.h>

using namespace Rcpp;
using namespace arma;
using namespace std;

double LLDE (double x) {
  if (x<=0) {
    return std::log(0.5) + ::Rf_dexp(-x, 1, 1);
  }
  else {
    return std::log(0.5) + ::Rf_dexp(x, 1, 1);
  }
}

double qDE (double p) {
  if (p >= 0.5) {
    return ::Rf_qexp(2*p-1, 1, 1, 0);
  }
  else {
    return -::Rf_qexp(1-2*p, 1, 1, 0);
  }
}
//log(error density)
double LLerror (double x, double p, double qc) {
  return LLDE(x+qc);
}

//log-likelihood of beta
double LLbetaDE (double beta, double sd, double p, NumericVector resid, double qc) {
  double result = 0.0;
  for (int i=0; i<resid.size(); i++) {
    result += LLerror(resid(i), p, qc);
  }
  return result + ::Rf_dnorm4(beta, 0, sd, 1);
}

//' BQR Based on DE
//'
//'Bayesian quantile regression using double exponential distribution (DE) as Likelihood Function.
//' @param Xr Matrix of covariate.
//' @param Yr Vector of response.
//' @param p Quantile of interest.
//' @param nsim Number of simulation.
//' @param initCoef Initial value (vector) of the model coefficients.
//' @param burn Number of burn-in.
//' @return List of model coefficients and acceptance rate.
//' @export
//'
//[[Rcpp::export]]
List bqrDE (NumericMatrix Xr, NumericVector Yr, double p, int nsim, NumericVector initCoef, int burn = 5000) {
  const double qc = qDE(p);
  // nsim is the number of simulation
  // Y is the response vector
  if (p<0 || p>1){
    throw Rcpp::exception("the percentile should be between 0 and 1!");
  }

  int size = Yr.size(); //sample size
  int ncov = Xr.ncol(); //number of covariate + 1
  if (size != Xr.nrow()){
    throw Rcpp::exception("the dimension of the design matrix and the response does not match!");
  }
  arma::mat X(Xr.begin(), size, ncov, false);
  arma::colvec Y(Yr.begin(), size, false);

  RNGScope scope;         // Initialize Random number generator
  int i;
  int sim;


  //fixed hyperparameters

  arma::colvec sdDiag = 10000 * arma::ones(ncov); //prior sd of the regression coefficients

  //initialize parameters to be updated
  arma::colvec beta = initCoef;  //regression coefficient
  NumericVector resid(size);   //residual
  resid = Y - X * beta;   //calculate the residual


  //outputs
  NumericMatrix betaRec(nsim, ncov);
  NumericVector nAccept(ncov, 0.0);

  //tuning parameter in the MH algorithm
  NumericVector att(ncov, 0.0);
  NumericVector acc(ncov, 0.0);
  NumericVector can(ncov, 0.25);
  double mhRate;
  NumericVector canResid(size);

  for (sim=0; sim<nsim; sim++){
    //Rcout << "*************************************************\n";
    //Rcout<<sim<<"\n";
    if (sim % 1000 == 0) {
       Rcout<<sim<<"\n";
    }

    //MH update for the regression coefficients
    for (int k=0; k<ncov; k++){
      //Rcout<<"regression coefficients # " <<(k+1)<<"\n";


      att(k)++;
      double canbeta = ::Rf_rnorm(beta(k), 2*can(k));
      for (i=0; i<size; i++) {
        canResid(i) = resid(i) + X(i,k) * (beta(k) - canbeta);
      }
      mhRate = LLbetaDE(canbeta, sdDiag(k), p, canResid, qc) - LLbetaDE(beta(k), sdDiag(k), p, resid, qc);
      if (!(mhRate!=mhRate)) {
        //Rcout<<cangamma<<"\t"<<mhRate<<"\n";
        if (::Rf_runif(0,1) < exp(mhRate)) {
          beta(k) = canbeta;
	  resid = Y - X * beta;   //update the residual
          acc(k)++;
	  if (sim >= burn) {
  	    nAccept(k)++;
    	  }
        }
      }

    }

    for (i=0; i<ncov; i++) {
      if (att(i)>200 && sim<burn) {
        can(i) = can(i) * (acc(i)/att(i)<0.2?0.5:1) * (acc(i)/att(i)>0.7?1.5:1);
        acc(i) = 1.0;
        att(i) = 1.0;
      }
    }

    betaRec.row(sim) = as<NumericVector>(wrap(beta));
    //Rcout << "*************************************************\n";
    //Rcout << "\n";
  }
  //betaRecFile.close();
  //test

  return List::create(Rcpp::Named("beta") = betaRec, Rcpp::Named("acceptRate") = nAccept/(nsim - burn));           // Return to R
}



