// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <string>
#include <iostream>
#include <fstream>
//#include <Rcpp.h>

using namespace Rcpp;
using namespace arma;
using namespace std;

double rho (double u, double p) {
  return u<0? u*(p-1) : u*p;
}

double LLASL (double x, double sigma, double p) {
  return std::log(p*(1-p)/sigma)-rho(x, p)/sigma;
}


//log-likelihood of beta
double LLbetaALD (double beta, double sd, double p, NumericVector resid, double sigma) {
  double result = 0.0;
  for (int i=0; i<resid.size(); i++) {
    result += LLASL(resid(i), sigma, p);
  }
  return result + ::Rf_dnorm4(beta, 0, sd, 1);
}

//' BQR Based on ALD
//'
//' Bayesian quantile regression using the Asymetric Laplacian Distribution (ALD) as Likelihood Function.
//'
//' @param Xr Matrix of covariate.
//' @param Yr Vector of response.
//' @param p Quantile of interest.
//' @param nsim Number of simulation.
//' @param initCoef Initial value (vector) of the model coefficients.
//' @param sigma The prespecified scale parameter of the asymmetric Laplace distribution.
//' @param burn Number of burn-in.
//' @return List of model coefficients and acceptance rate.
//' @export
//'
//[[Rcpp::export]]
List bqrALD (NumericMatrix Xr, NumericVector Yr, double p, int nsim, NumericVector initCoef, double sigma, int burn = 5000) {
  // nsim is the number of simulation
  // sigma is the prespecified scale parameter of the asymmetric laplace distribution
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
    //if (sim % 1000 == 0) {
    //   Rcout<<sim<<"\n";
    //}

    //MH update for the regression coefficients
    for (int k=0; k<ncov; k++){
      //Rcout<<"regression coefficients # " <<(k+1)<<"\n";


      att(k)++;
      double canbeta = ::Rf_rnorm(beta(k), 2*can(k));
      for (i=0; i<size; i++) {
        canResid(i) = resid(i) + X(i,k) * (beta(k) - canbeta);
      }
      mhRate = LLbetaALD(canbeta, sdDiag(k), p, canResid, sigma) - LLbetaALD(beta(k), sdDiag(k), p, resid, sigma);
      if (!(mhRate!=mhRate)) {
        //Rcout<<cangamma<<"\t"<<mhRate<<"\n";
        if (::Rf_runif(0,1) < exp(mhRate)) {
          beta(k) = canbeta;
          acc(k)++;
	  if (sim >= burn) {
  	    nAccept(k)++;
    	  }
        }
      }


      //Rcout << beta(k) <<"\n";
      resid = Y - X * beta;   //update the residual
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



