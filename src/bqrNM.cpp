// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <string>
#include <iostream>
#include <fstream>
//#include <Rcpp.h>

using namespace Rcpp;
using namespace arma;
using namespace std;

//calculate the quantile of normal mixtures
double calcQNormalMix(NumericVector weight, NumericVector mu, NumericVector sigma, double p) {
  int n = weight.size();
  double qleft = ::Rf_qnorm5(p, mu(0), sigma(0), 1, 0);
  if (n==1) {
    return qleft;
  }
  double qright = qleft;
  double qmid = qleft;
  double temp;
  int i;
  for (i=1; i<n; i++) {
    temp = ::Rf_qnorm5(p, mu(i), sigma(i), 1, 0);
    if (qleft > temp) {
      qleft = temp;
    }
    else if (qright < temp) {
      qright = temp;
    }
  }
  while (qright-qleft > 0.0000000001) {
    qmid = (qright + qleft) / 2.0;
    temp = 0.0;
    for (i=0; i<n; i++) {
      temp += weight(i) * ::Rf_pnorm5(qmid, mu(i), sigma(i), 1, 0);
    }
    if (temp < p) {
      qleft = qmid;
    }
    else if (temp > p) {
      qright = qmid;
    }
    else {
      return qmid;
    }
  }
  return qmid;
}

//log(error density)
double LLerror (double x, double p, NumericVector weight, NumericVector mu, NumericVector sigma, double qc) {
  double result = 1.0;
  for (int i=0; i<weight.size(); i++) {
    result *= ::Rf_dnorm4(x, mu(i)-qc, sigma(i), 0);
  }
  return std::log(result);
}

//log-likelihood of beta
double LLbeta (double beta, double sd, double p, NumericVector resid, NumericVector weight, NumericVector mu, NumericVector sigma, double qc) {
  double result = 0.0;
  for (int i=0; i<resid.size(); i++) {
    result += LLerror(resid(i), p, weight, mu, sigma, qc);
  }
  return result + ::Rf_dnorm4(beta, 0, sd, 1);
}

//' BQR Based on Mixture of Normal Distribution
//'
//'Bayesian quantile regression using a mixture of normal distributions as Likelihood Function.
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
List bqrNM (NumericMatrix Xr, NumericVector Yr, double p, int nsim, NumericVector initCoef, NumericVector weight, NumericVector mu, NumericVector sigma, int burn = 5000) {
  // nsim is the number of simulation
  // Y is the response vector
  const double qc = calcQNormalMix(weight, mu, sigma, p);
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
      mhRate = LLbeta(canbeta, sdDiag(k), p, canResid, weight, mu, sigma, qc) - LLbeta(beta(k), sdDiag(k), p, resid, weight, mu, sigma, qc);
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


      //Rcout << beta(k) <<"\n";

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



