// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <string>
#include <iostream>
#include <fstream>
//#include <Rcpp.h>

using namespace Rcpp;
using namespace arma;
using namespace std;


//log-likelihood of beta
double LLbeta (double beta, double sd, double p, NumericVector resid, NumericVector sign, double qc) {
  double result = 0.0;
  for (int i=0; i<resid.size(); i++) {
    result += ::Rf_dnorm4(resid(i), -qc*sign(i), sign(i), 1);
  }
  return result + ::Rf_dnorm4(beta, 0, sd, 1);
}

//log-likelihood of gamma
double LLgamma (double gamma, double sd, double p, NumericVector resid, NumericVector sign, double qc) {
  double result = 0.0;
  for (int i=0; i<resid.size(); i++) {
    result += ::Rf_dnorm4(resid(i), -qc*sign(i), sign(i), 1);
  }
  return result + ::Rf_dnorm4(gamma, 0, sd, 1);
}

//' BQR with Heteroscedasticity Based on Normal Distribution
//'
//'Bayesian quantile regression with heteroscedasticity using normal distribution as Likelihood Function.
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
List bqrNH (NumericMatrix Xr, NumericVector Yr, double p, int nsim, NumericVector initCoef, int burn = 5000) {
  const double qc = ::Rf_qnorm5(p, 0, 1, 1, 0);
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
  arma::colvec gamma(ncov); //model heteroscedasticity
  gamma(0) = 1.0;
  for (i=1; i<ncov; i++) {
    gamma(i) = 0.0;
  }
  NumericVector resid(size);   //residual
  NumericVector sign(size, 1.0);  //x gamma
  resid = Y - X * beta;   //calculate the residual




  //outputs
  NumericMatrix betaRec(nsim, ncov);
  NumericMatrix gammaRec(nsim, ncov);
  NumericVector nAccept(ncov*2, 0.0);

  //tuning parameter in the MH algorithm
  NumericVector att(ncov*2, 0.0);
  NumericVector acc(ncov*2, 0.0);
  NumericVector can(ncov*2, 0.25);
  double mhRate;
  NumericVector canResid(size);
  NumericVector canSign(size);

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
      mhRate = LLbeta(canbeta, sdDiag(k), p, canResid, sign, qc) - LLbeta(beta(k), sdDiag(k), p, resid, sign, qc);
      if (!(mhRate!=mhRate)) {
        //Rcout<<cangamma<<"\t"<<mhRate<<"\n";
        if (::Rf_runif(0,1) < exp(mhRate)) {
          beta(k) = canbeta;
  	  resid = clone(canResid);
          acc(k)++;
	  if (sim >= burn) {
  	    nAccept(k)++;
    	  }
        }
      }
    }

    //MH update for gamma
    for (int k=0; k<ncov; k++) {
      att(ncov+k)++;
      double cangamma = ::Rf_rnorm(gamma(k), 2*can(ncov+k));
      bool hasNeg = false;
      for (i=0; i<size; i++) {
        canSign(i) = sign(i) + X(i,k) * (cangamma - gamma(k));
        if (canSign(i) <= 0) {
	  hasNeg = true;
          break;
        }
      }
      if (hasNeg) {
        continue;
      }
      else {
	mhRate = LLgamma(cangamma, sdDiag(k), p, resid, canSign, qc) - LLgamma(gamma(k), sdDiag(k), p, resid, sign, qc);
	if (!(mhRate!=mhRate)) {
	  //Rcout<<cangamma<<"\t"<<mhRate<<"\n";
	  if (::Rf_runif(0,1) < exp(mhRate)) {
	    gamma(k) = cangamma;
            sign = clone(canSign);
	    acc(ncov+k)++;
	    if (sim >= burn) {
	      nAccept(ncov+k)++;
	    }
	  }
	}
      }

    }

    for (i=0; i<ncov*2; i++) {
      if (att(i)>200 && sim<burn) {
        can(i) = can(i) * (acc(i)/att(i)<0.2?0.5:1) * (acc(i)/att(i)>0.7?1.5:1);
        acc(i) = 1.0;
        att(i) = 1.0;
      }
    }

    betaRec.row(sim) = as<NumericVector>(wrap(beta));
    gammaRec.row(sim) = as<NumericVector>(wrap(gamma));
    //Rcout << "*************************************************\n";
    //Rcout << "\n";
  }
  //betaRecFile.close();
  //test

  return List::create(Rcpp::Named("beta") = betaRec, Rcpp::Named("gamma") = gammaRec, Rcpp::Named("acceptRate") = nAccept/(nsim - burn));           // Return to R
}



