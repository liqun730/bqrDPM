// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <string>
#include <iostream>
#include <fstream>
//#include <Rcpp.h>

using namespace Rcpp;
using namespace arma;
using namespace std;

//calculate the sample quantile
double adrejSamplerBetaHA(NumericVector array, double p) {
  NumericVector data = clone(array);
  int n = (int)(data.size()*p);
  std::nth_element(data.begin(), data.begin()+n, data.end());
  double result = data(n);
  std::nth_element(data.begin(), data.begin()+n-1, data.end());
  result = (data(n) + data(n-1))/2;
  return result;
}


// accurately calculate the ratio of two doubles
double ar(double num, double denom) {
  if (num == 0 && denom == 0) {
    throw Rcpp::exception("Both the numerator and the denominator in a ratio are equal to 0!");
  }
  else if (num == 0 && denom != 0) {
    return 0.0;
  }
  else if (num > 0 && denom == 0) {
    return 1.0/0.0;
  }
  else if (num < 0 && denom == 0) {
    return -1.0/0.0;
  }
  else {
    if (num > 0 && denom > 0) {
      return exp(log(num) - log(denom));
    }
    else if (num > 0 && denom < 0) {
      return -exp(log(num) - log(-denom));
    }
    else if (num < 0 && denom > 0) {
      return -exp(log(-num) - log(denom));
    }
    else {
      return exp(log(-num) - log(-denom));
    }
  }
}


//the equation used to solve for the quantile
double fLambdaHA(double x, double p){
  return x * log(x) + 1.0 - x - p * pow(x - 1.0, 2.0);
}

double dLogisticHA(double x, double p, double tau, double sigma) {
  double temp = -(x-tau)/sigma;
  if (temp > 0) {
    return exp(log((1.0-p)/p) - temp - log(sigma) - 2.0 * log((1.0-p)/p + exp(-temp)));
  }
  else {
    return exp(log((1.0-p)/p) + temp - log(sigma) - 2.0 * log(1.0 + exp(temp)*(1.0-p)/p));
  }
}

double lLogisticHA(double x, double p, double tau, double sigma) {
  double temp = -(x-tau)/sigma;
  if (temp > 0) {
    return log((1.0-p)/p) - temp - log(sigma) - 2.0 * log((1.0-p)/p + exp(-temp));
  }
  else {
    return log((1.0-p)/p) + temp - log(sigma) - 2.0 * log(1.0 + exp(temp)*(1.0-p)/p);
  }
}

//the derivative of the function used to solve for the quantile
double fLambdaHADer(double x, double p){
  return log(x) - 2 * p * (x - 1.0);
}

//the equation used to solve for the quantile of mixture of logistic distribution
double fqHA(double x, NumericVector gamma, NumericVector tau, NumericVector sigma, double p) {
  int n = gamma.size();
  if ( n != tau.size()) {
    throw Rcpp::exception("The number of component is inconsistent!!!");
  }

  double value = 0;
  double temp;
  for (int i=0; i<n; i++) {
    //value += gamma(i) / (p + (1.0-p) * exp(-(x-tau(i))/sigma(i)));
    temp = -(x+tau(i)) / sigma(i);
    if (temp < 0) {
      value += gamma(i) / (p + (1.0-p) * exp(temp));
    }
    else {
      value += gamma(i) * exp(-temp) / (p * exp(-temp) + (1.0-p));
    }
  }
  value -= 1.0;
  return value;
}



//the derivative of the equation used to solve for the quantile of mixture of logistic distribution
double fqHADer(double x, NumericVector gamma, NumericVector tau, NumericVector sigma, double p) {
  int n = gamma.size();
  if ( n != tau.size()) {
    throw Rcpp::exception("The number of component is inconsistent!!!");
  }
  double sum = 0.0;
  for (int i=0; i<n; i++) {
    sum += gamma(i);
  }
  if (sum != 1.0) {
    Rcout<<"The mixing probabilities sum to "<<sum<<"\n";
    throw Rcpp::exception("The mixing probabilities should sum to 1!!!");
  }

  double value = 0;
  double temp;
  double temp1;
  for (int i=0; i<n; i++) {
    temp = (tau(i)-x)/sigma(i) - 2 * log(p+(1.0-p)*exp(-(x-tau(i))/sigma(i)));
    temp1 = -(x-tau(i)) / sigma(i);
    if (temp1 > 0) {
      temp -= (2 * temp1 + 2 * log(p * exp(-temp1) + 1.0 - p));
    }
    else {
      temp -= 2 * log(p+(1.0-p)*exp(temp1));
    }

    if (temp != temp) {
      throw Rcpp::exception("Something is wrong with fqHADer!!!");
    }
    else {
      value += (1.0-p)/sigma(i) * exp(temp);
    }

    //value += gamma(i) * (1.0-p) * exp(-(x-tau(i))/sigma(i)) / sigma(i) / pow(p + (1.0-p) * exp(-(x-tau(i))/sigma(i)), 2.0);
  }
  return value;
}

double findRoughRootQHA(NumericVector gamma, NumericVector tau, NumericVector sigma, double p, double threshold, double maxTau, double minTau){
  double left = minTau;
  double right = maxTau;
  if (fqHA(left, gamma, tau, sigma, p) > 0 || fqHA(right, gamma, tau, sigma, p) < 0) {
    throw Rcpp::exception("Something is wrong with findRoughRootQHA!!!");
  }
  double mid = (left + right) / 2.0;
  while (right - left > threshold){
    if (fqHA(mid, gamma, tau, sigma, p) > 0){
      right = mid;
    }
    else if (fqHA(mid, gamma, tau, sigma, p) < 0){
      left = mid;
    }
    else {
      return mid;
    }
    mid = (left + right) / 2.0;
  }
  return mid;
}


double findRootNewtonQHA(NumericVector gamma, NumericVector tau, NumericVector sigma, double p, double threshold){
  //use Newton method to solve the quantile of mixture of logistic distribution
  int n = gamma.size();
  if (n==1) {
    return -tau(0);
  }
  if ( n != tau.size()) {
    throw Rcpp::exception("The number of component is inconsistent!!!");
  }
  double sum = 0.0;
  for (int i=0; i<n; i++) {
    sum += gamma(i);
  }
  if (sum != 1.0) {
    Rcout<<"The mixing probabilities sum to "<<sum<<"\n";
    throw Rcpp::exception("The mixing probabilities should sum to 1!!!");
  }

  double maxTau;
  double minTau;
  if (-tau(0) > -tau(1)) {
    maxTau = -tau(0);
    minTau = -tau(1);
  }
  else {
    maxTau = -tau(1);
    minTau = -tau(0);
  }
  for (int i=2; i< n; i++) {
    if (-tau(i)>maxTau) {
      maxTau = -tau(i);
    }
    if (-tau(i)<minTau) {
      minTau = -tau(i);
    }
  }
  return findRoughRootQHA(gamma, tau, sigma, p, threshold, maxTau, minTau);

  /*//use the output of midpoint method as the initial value
  double xold = findRoughRootQHA(gamma, tau, sigma, p, 0.001, maxTau, minTau);
  double xnew = xold - fqHA(xold, gamma, tau, sigma, p) / fqHADer(xold, gamma, tau, sigma, p);
  //Rcout << xold << "\t" << xnew << "\t" << fqHA(xold, gamma, tau, sigma, p) << "\t" << fqHADer(xold, gamma, tau, sigma, p) << "\n";
  while (std::abs(xold - xnew) > threshold){
    xold = xnew;
    xnew = xold - fqHA(xold, gamma, tau, sigma, p) / fqHADer(xold, gamma, tau, sigma, p);
    //Rcout << xold << "\t" << xnew << "\t" << fqHA(xold, gamma, tau, sigma, p) << "\t" << fqHADer(xold, gamma, tau, sigma, p) << "\n";
  }
  return xnew;*/
}


//normalize a vector (which has been taken logrithm) such that they sum to one
NumericVector normalizeHA(NumericVector x, int n){
  //x is the vector of logrithm of the values to be nornmalized
  if (n == 0){
    throw Rcpp::exception("The vector of valued to be normalized has length 0!!!");
  }
  double max = x(0);
  for (int i=1; i<n; i++){
    if (x(i) > max){
      max = x(i);
    }
  }
  double denom = 0.0;
  for (int i=0; i<n; i++){
    denom += exp(x(i) - max);
  }
  NumericVector prob(n);
  for (int i=0; i<n; i++){
    prob(i) = exp(x(i) - max - log(denom));
    ////Rcout<<x(i) <<"-->"<<prob(i) << "\t";
  }
 // //Rcout<<"PROB\n";
  return prob;
}



// find the an rough approximation of the root of an equation by binary search
double findRoughRootHA(double p, double threshold){
  // p is not equal to 0.5
  // threshold controls when to stop the iteration
  if (p < 0.5){
    double left = 1.1;
    double right = 5.0;
    while (fLambdaHA(left, p) < 0){
      left = sqrt(left);
    }
    while (fLambdaHA(right, p) > 0){
      right = pow(right, 2.0);
    }
    double mid = (left + right) / 2.0;
    while (right - left > threshold){
      if (fLambdaHA(mid, p) > 0){
        left = mid;
      }
      else if (fLambdaHA(mid, p) < 0){
        right = mid;
      }
      else {
        return mid;
      }
      mid = (left + right) / 2.0;
    }
    return mid;
  }
  else {
    double left = 0.1;
    double right = 0.9;
    while (fLambdaHA(left, p) < 0){
      left = pow(left, 2.0);
    }
    while (fLambdaHA(right, p) > 0){
      right = sqrt(right);
    }
    double mid = (left + right) / 2.0;
    while (right - left > threshold){
      if (fLambdaHA(mid, p) > 0){
        left = mid;
      }
      else if (fLambdaHA(mid, p) < 0){
        right = mid;
      }
      else {
        return mid;
      }
      mid = (left + right) / 2.0;
    }
    return mid;
  }
}


double findRootNewtonHA(double p, double threshold){
  //use Newton-Raphson method to solve for the lambda
  if (p == 0.5){
    return 1.0;
  }
  else {
    double xold = findRoughRootHA(p, 0.1);
    double xnew = xold - fLambdaHA(xold, p) / fLambdaHADer(xold, p);
    while (std::abs(xold - xnew) > threshold){
      xold = xnew;
      xnew = xold - fLambdaHA(xold, p) / fLambdaHADer(xold, p);
      ////Rcout << std::abs(xold - xnew) << "\n";
    }
    return xnew;
  }
}



double LLqHA(double sigma, double q, arma::colvec resid, IntegerVector g, int ind, double c, double d, double p, double lambda){
  int n = g.size();
  int count = 0;
  double ll = 0.0;
  double temp = -q / sigma;
  if (temp > 0){
    ll -= 2 * log((1.0 - p) / lambda / p + exp(-temp)) + 2 * temp;
  }
  else {
    ll -= 2 * log(1.0 + (1.0 - p) * exp(temp) / lambda / p);
  }
  for (int i=0; i<n; i++){
    if (g(i) == ind) {
      count++;
      temp = -(q + resid(i)) / sigma;
      if (temp > 0){
        ll -= 2 * log((1.0 - p) / p + exp(-temp)) + 2 * temp;
      }
      else{
        ll -= 2 * log(1.0 + (1.0 - p) * exp(temp) / p);
      }
    }
  }
  ll += -(count + 1.0) * q / sigma;
  return ll;
}

double LLsigmaHA(double sigma, double q, arma::colvec resid, IntegerVector g, int ind, double c, double d, double p, double lambda){
  int n = g.size();
  int count = 0;
  double sumR = 0.0;
  double ll = 0;
  double temp = -q / sigma;
  if (temp > 0) {
    ll -= 2 * log(exp(-temp) + (1.0 - p) / lambda / p) +  2 * temp;
  }
  else {
    ll -= 2 * log(1.0 + (1.0 - p) * exp(temp) / lambda / p);
  }

  for (int i=0; i<n; i++){
    if (g(i) == ind) {
      count++;
      sumR += resid(i);
      temp = -(q + resid(i)) / sigma;
      if (temp > 0) {
        ll -= 2 * log(exp(-temp) + (1.0 - p) / p) + 2 * temp;
      }
      else {
        ll -= 2 * log(1.0 + (1.0 - p) * exp(temp) / p);
      }
    }
  }
  ll += -((count + 1.0) * q + d + sumR) / sigma - (c + count + 2.0) * log(sigma);
  return ll;
}

double LLsigmaHAInv(double sigmaInv, double q, arma::colvec resid, IntegerVector ind, double c, double d, double p, double lambda){
  int n = ind.size();
  double sumR = 0.0;
  for (int i=0; i<n; i++){
    sumR += resid(ind(i));
  }
  double ll = -((n + 1.0) * q + d + sumR) * sigmaInv + (c + n) * log(sigmaInv);
  double temp = -q * sigmaInv;
  if (temp > 0){
    ll -= 2 * log((1.0 - p) / p / lambda + exp(-temp)) + 2 * temp;
  }
  else {
    ll -= 2 * log(1.0 + (1.0 - p) * exp(temp) / lambda / p);
  }
  for (int i=0; i<n; i++){
    temp = -(q + resid(ind(i))) * sigmaInv;
    if (temp > 0){
      ll -= 2 * log((1.0 - p) / p + exp(-temp)) + 2 * temp;
    }
    else{
      ll -= 2 * log(1.0 + (1.0 - p) * exp(temp) / p);
    }
  }
  return ll;
}


double LLbetaHA(double beta, double var, int k, NumericVector qStar, IntegerVector g, NumericVector sigh, NumericVector resid, double p, NumericVector sigmaStar, arma::mat X){
  int n = resid.size();
  if (var <=0 ){
    throw Rcpp::exception("Variance should be positive!");
  }
  double ll = -pow(beta,2) / 2 / var;
  for (int i=0; i<n; i++){
    double temp = (resid(i) + qStar(g(i))*sigh(i)) / sigmaStar(g(i)) / sigh(i);
    if (temp > 0) {
      ll -= temp + 2 * log(1.0 + (1.0 - p) * exp(-temp) / p );
    }
    else {
      ll -= 2 * log(exp(temp) + (1.0 - p) / p) - temp;
    }
  }
  return ll;
}




int findDupHA(NumericVector sigma, NumericVector q, double vsigma, double vq, int n){
  //if value is in the vector values (of length n not necessary equal to value.size()), return the index, otherwise return -1
  for (int i=0; i<n; i++){
    if (vsigma == sigma(i) && vq == q(i)){
      return i;
    }
  }
  return -1;
}

IntegerVector findClusterHA(IntegerVector s, int ind){
  int num = 0;
  int n = s.size();
  IntegerVector clust(n);
  for (int i=0; i<n; i++){
    if (s(i) == ind){
      clust(num) = i;
      num++;
    }
  }
  IntegerVector cluster(num);
  for (int i=0; i<num; i++){
    cluster(i) = clust(i);
  }
  return cluster;
}

//sample from consecutive integers with replacement according to given probabilities

IntegerVector sampleWRHA(int start, int end, int size, NumericVector probs) {
  int n = probs.size();
  if (n != end-start+1) {
    throw(Rcpp::exception("Something is wrong with sampleWRHA 1!"));
  }
  IntegerVector samples(size);
  double temp;
  NumericVector endpts(n+1);
  endpts(0) = 0.0;
  for (int i=1; i<n+1; i++) {
    endpts(i) = endpts(i-1) + probs(i-1);
  }
  if (endpts(n) != 1.0) {
    //need to normalize probs
    for (int i=1; i<n; i++) {
      endpts(i) /= endpts(n);
    }
    endpts(n) = 1.0;
    //throw(Rcpp::exception("Something is wrong with sampleWRHA 2!"));
  }
  for (int i=0; i<size; i++) {
    temp = ::Rf_runif(0.0, 1.0);
    for (int j=0; j<n; j++) {
      if (temp>endpts(j) && temp<=endpts(j+1)) {
        samples(i) = j + start;
        break;
      }
    }
  }
  return samples;
}

//sample one element from consecutive integers with replacement according to given probabilities

int sampleOneHA(int start, int end, arma::colvec probs) {
  //test
  /*
  for (int i=0; i<probs.size(); i++) {
    Rcout<<probs(i)<<"\t";
  }
  Rcout<<"\n";*/
  //test
  int n = probs.size();
  if (n != end-start+1) {
    throw(Rcpp::exception("Something is wrong with sampleWRHA 1!"));
  }
  double temp;
  NumericVector endpts(n+1);
  endpts(0) = 0.0;
  for (int i=1; i<n+1; i++) {
    endpts(i) = endpts(i-1) + probs(i-1);
    //Rcout<<endpts(i)<<"\n";
  }
  //test
  //Rcout<<endpts(n)<<"  ***********\n";
  //test
  if (endpts(n) != 1.0) {
    //need to normalize probs
    for (int i=1; i<n; i++) {
      endpts(i) /= endpts(n);
      //Rcout<<endpts(i)<<"\n";
    }
    endpts(n) = 1.0;
    //throw(Rcpp::exception("Something is wrong with sampleWRHA 2!"));
  }
  temp = ::Rf_runif(0.0, 1.0);
  for (int j=0; j<n; j++) {
    if (temp>endpts(j) && temp<=endpts(j+1)) {
      return j + start;
      break;
    }
  }
  return -1;
}

//compute the stick-breaking weights
NumericVector makeProbsHA(NumericVector v) {
  int n = v.size();
  NumericVector probs(n);
  probs(0) = v(0);
  double temp = v(0);
  for (int i=1; i<n-1; i++) {
    probs(i) = v(i);
    for (int j=0; j<i; j++) {
      probs(i) *= 1.0 - v(j);
    }
    temp += probs(i);
  }
  probs(n-1) = 1.0 - temp;
  return probs;
}

//' BQR with Heteroskedasticity based on DPM of Logistic Distribution (What does A mean?)
//'
//' Bayesian quantile regression with heteroskedasticity term based on Dirichlet process mixture of Logistic Distribution.
//'
//' @param Xr Matrix of covariate.
//' @param Yr Vector of response.
//' @param p Quantile of interest.
//' @param nsim Number of simulation.
//' @param initCoef Initial value (vector) of the model coefficients.
//' @param sampleQ A boolean variable which determines how to calculate the location shift.
//'   (true means using sample quantile, false means using the density estimation)
//' @param burn Number of burn-in.
//' @param plotDensity A boolean variable which decides whether to plot the density estimation or not.
//' @param naux Number of the auxiliary variable.
//' @param threshold The threshold of the error of the solution to the equation of the lambda.
//' @param hasHeter Boolean indication whether heteroskedasticity is present in the model.
//' @return List of model parameters including the coefficients.
//' @export
//'
//[[Rcpp::export]]
List bqrDPMLHA (NumericMatrix Xr, NumericVector Yr, double p, int nsim, NumericVector initCoef, bool sampleQ = true, int burn = 5000, bool plotDensity = false, int nF=10,  double threshold=0.0000001, bool hasHeter = true) {
  //initialization
  // nsim is the number of simulation
  // nF is the number of components in the finite approximation
  // p is the quantile regression
  // X is the design matrix with each row correspnding to an observation (the first column with all ones should be removed)
  // Y is the response vector
  // initCoef is the initial value for the regression coefficients
  // sampleQ is a flag which determines how to calculate the location shift. (true means using sample quantile, false means using the density estimation)
  // threshold is the threshold of the error of the solution to the equation of the lambda
  if (p<0 || p>1){
    throw Rcpp::exception("the percentile should be between 0 and 1!");
  }


  //double tuneP = 1.0; //tuning parameter for the proposal distribution for the MH update of gamma
  double lambda = findRootNewtonHA(p, threshold);
  //lambda = 1;
  //Rcout << lambda << "\n";
  //throw Rcpp::exception("the percentile should be between 0 and 1!");
  int size = Yr.size(); //sample size
  int ncov = Xr.ncol(); //number of covariate + 1
  if (size != Xr.nrow()){
    throw Rcpp::exception("the dimension of the design matrix and the response does not match!");
  }
  arma::mat X(Xr.begin(), size, ncov, false);
  arma::colvec Y(Yr.begin(), size, false);

  RNGScope scope;         // Initialize Random number generator
  int i;
  int j;
  int sim;

  //fixed hyperparameters
  double c = 2.0;  //shape parameter for the base measure (inverse gamma) of the first DP
  NumericVector sdDiag = rep(10000.0, ncov);  //square root of the diagnal elements of the covariance matrix (diagonal) of the prior (multivariate normal)
                                            //of the regression coeffcients
  NumericVector mu = rep(0.0, ncov);  //mean vector of the prior (multivariate normal) of the regression coefficients
  double a1 = 1.0;   //shape parameter for the Gamma prior for the precision parameters of DP
  double b1 = 1.0;//10.0 / size;  //rate parameter for the Gamma prior for the precision parameters of DP
  double a2 = -1.0; //shape parameter for the Gamma prior for d
  double b2 = -1.0; //rate parameter for the Gamma prior for d
  for (i=0; i<size; i++){
    if (Y(i) > 0 && Y(i) > b2){
      b2 = Y(i);
    }
    if (Y(i) < 0 && a2 + Y(i) < 0){
      a2 = -Y(i);
    }
  }
  b2 = 1.0 / (a2 + b2);//1.0 / 10000.0;
  a2 = 1.0;

  //intermediate parameters
  NumericVector sigmaStar(nF);   //unique values for the DP
  NumericVector qStar(nF);
  NumericVector sigmaScale(size);
  NumericVector qScale(size);

  //initialize parameters to be updated
  arma::colvec beta = initCoef;  //regression coefficient
  arma::colvec gamma(ncov);
  gamma(0) = 1.0;
  for (i=1; i<ncov; i++) {
    gamma(i) = 0.0;
  }
  double d = 0.0;  //scale parameter for the base measure (inverse gamma) of the DP
  NumericVector resid(size);   //residual
  NumericVector sigh(size, 1.0);   //multiplier to the error


  resid = Y - X * beta;   //calculate the residual
  for (i=0; i<size; i++) {
    resid(i) /= sigh(i);
  }
  for (i=0; i<size; i++){
    d += resid(i);
  }
  d /= size;
  double var = 0;
  for (i=0; i<size; i++){
    var += pow(resid(i) - d, 2.0);
  }
  var /= size;
  d = sqrt(3.0 * var / 3.14159 / 3.14159);


  double alpha = 1.0;  //the precision parameter for the DP
  //NumericVector sigma(size); //scale parameter of the logistic distribution
  //NumericVector q(size); //quantile parameter of the logistc distribution

  for (i=0; i<nF; i++){
    sigmaStar(i) = 1.0 / ::Rf_rgamma(c, 1.0 / d);
    double u = R::runif(0.0, 1.0);
    qStar(i) = -sigmaStar(i) * log(lambda) + sigmaStar(i) * log(u * (1.0 - p) / p / (1.0 - u));
  }

  NumericVector v = rbeta(nF, 1.0, alpha);
  v(nF-1) = 1.0;
  NumericVector probs = makeProbsHA(v);
  IntegerVector g = sampleWRHA(0,nF-1, size, probs);


  //matrix to restore the mcmc samples
  NumericMatrix betaRec(nsim, ncov);
  NumericMatrix gammaRec(nsim, ncov);
  NumericMatrix sigmaRec(nsim, nF);
  NumericMatrix qRec(nsim, nF);
  NumericVector dRec(nsim);
  NumericVector nClusterRec(nsim);
  NumericVector nClusterTRec(nsim);
  NumericVector alphaRec(nsim);
  NumericVector interceptAdj(nsim);
  //NumericMatrix clusterRec(nsim, size);
  NumericMatrix nRejRec(nsim, 2 + ncov); // keep track average number of rejections for location parameters, scale parameters and regression coefficients
  NumericVector nAcceptGamma(ncov, 0.0);
  NumericVector nAttemptGamma(ncov, 0.0);
  NumericVector att(3+ncov, 0.0);
  NumericVector acc(3+ncov, 0.0);
  NumericVector can(3+ncov, 0.25);
  NumericVector cansigh(size);
  NumericVector canResid(size);

  NumericVector xGrid(4001);
  xGrid(0) = -20.0;
  for (i=1; i<4001; i++) {
    xGrid(i) = xGrid(i-1) + 0.01;
  }
  NumericVector sumdense(4001, 0.0);
  //ofstream betaRecFile;
  //std::string filename = path + "betaLogisticRecFile.txt";
  //betaRecFile.open (filename.c_str());
  //ofstream dRecFile;
  //ofstream


  arma::colvec lll(nF);
  double temp;
  double mhRate;
  double aveN = 0.0;




  //Rcout << "initializing ******************\n";



  for (sim=0; sim<nsim; sim++){
    //Rcout << "*************************************************\n";
    //Rcout<<sim<<":"<<beta(0)<<"\t"<<beta(1)<<"\t"<<beta(2)<<"\n";
    //Rcout<<"n\t"<<n<<"\n";

    //Rcout << "updating regression coefficients...\n";
    for (i=0; i<size; i++) {
      resid(i) *= sigh(i);
    }
    for (int k=0; k<ncov; k++){
      //Rcout<<"regression coefficients # " <<(k+1)<<"\n";
      att(3+k)++;
      double canbeta = ::Rf_rnorm(beta(k), 2*can(3+k));
      for (i=0; i<size; i++) {
        canResid(i) = resid(i) + X(i,k) * (beta(k) - canbeta);
      }

      mhRate = LLbetaHA(canbeta, sdDiag(k)*sdDiag(k), k, qStar, g, sigh, canResid, p, sigmaStar, X) - LLbetaHA(beta(k), sdDiag(k)*sdDiag(k), k, qStar, g, sigh, resid, p, sigmaStar, X);
      if (!(mhRate!=mhRate)) {
        //Rcout<<cangamma<<"\t"<<mhRate<<"\n";
        if (::Rf_runif(0,1) < exp(mhRate)) {
          beta(k) = canbeta;
	  resid = clone(canResid);
          acc(3+k)++;
        }
        else {
          nRejRec(sim, 2+k)++;
        }
      }
    }


    //update gamma
    if (hasHeter) {
      for (i=1; i<ncov; i++) {
        att(2)++;
        double cangamma = ::Rf_rnorm(gamma(i), 2*can(2));
        bool hasNeg = false;

        for (j=0; j<size; j++) {
          cansigh(j) = sigh(j) - X(j, i) * (gamma(i) - cangamma);
          if (cansigh(j) < 0.0) {
            hasNeg = true;
            break;
          }
        }

        if (hasNeg) {
          continue;
        }
        else {
          mhRate = ::Rf_dnorm4(cangamma, 0.0, sdDiag(i), 1) - ::Rf_dnorm4(gamma(i), 0.0, sdDiag(i), 1);
          for (j=0; j<size; j++) {
            mhRate += lLogisticHA(resid(j), p, -qStar(g(j))*cansigh(j), sigmaStar(g(j))*cansigh(j)) - lLogisticHA(resid(j), p, -qStar(g(j))*sigh(j), sigmaStar(g(j))*sigh(j));
          }
        }
        if (!(mhRate!=mhRate)) {
          //Rcout<<cangamma<<"\t"<<mhRate<<"\n";
          if (::Rf_runif(0,1) < exp(mhRate)) {
            gamma(i) = cangamma;
            sigh = clone(cansigh);
            nAcceptGamma(i)++;
            acc(2)++;
          }
        }
      }
    }

    //Rcout << "updating the conFiguration g.. \n";
    for (i=0; i<size; i++) {
      for (int j=0; j<nF; j++) {
        lll(j) = lLogisticHA(resid(i), p, -qStar(j)*sigh(i), sigmaStar(j)*sigh(i)) + log(probs(j));
      }
      g(i) = sampleOneHA(0,nF-1,arma::exp(lll-arma::max(lll)));
    }

    if (hasHeter) {
      for (i=0; i<size; i++) {
        resid(i) /= sigh(i);
      }
    }

    //Rcout << "updating unique sigma and q...\n";
    nRejRec(sim, 0) = 0.0;
    nRejRec(sim, 1) = 0.0;
    for (i=0; i<nF; i++){
      att(0)++;
      double cansigma = ::Rf_rnorm(sigmaStar(i), 2*can(0));
      mhRate = LLsigmaHA(cansigma, qStar(i), resid, g, i, c, d, p, lambda) - LLsigmaHA(sigmaStar(i), qStar(i), resid, g, i, c, d, p, lambda);
      if (!(mhRate!=mhRate)) {
        //Rcout<<cangamma<<"\t"<<mhRate<<"\n";
        if (::Rf_runif(0,1) < exp(mhRate)) {
          sigmaStar(i) = cansigma;
          acc(0)++;
        }
        else {
          nRejRec(sim, 1)++;
        }
      }

     // Rcout<<"********q\n";

      /*result = adrejSamplerq(sigmaStar(i), resid, cluster, c, d, p, lambda);
      qStar(i) = result(0);
      nRejRec(sim, 0) = nRejRec(sim, 0) + result(1);*/
      att(1)++;
      double canq = ::Rf_rnorm(qStar(i), 2*can(1));
      mhRate = LLqHA(sigmaStar(i), canq, resid, g, i, c, d, p, lambda) - LLqHA(sigmaStar(i), qStar(i), resid, g, i, c, d, p, lambda);
      if (!(mhRate!=mhRate)) {
        //Rcout<<cangamma<<"\t"<<mhRate<<"\n";
        if (::Rf_runif(0,1) < exp(mhRate)) {
          qStar(i) = canq;
          acc(1)++;
        }
        else {
          nRejRec(sim, 0)++;
        }
      }

      ////Rcout << qStar(i) << "\n";
    }
    //nRejRec(sim, 0) = nRejRec(sim, 0) / n;
    //nRejRec(sim, 1) = nRejRec(sim, 1) / n;

    //Rcout << "updating the conFiguation...\n";
    IntegerVector clusterSize(nF, 0);
    IntegerVector largerCount(nF, 0);
    for (i=0; i<size; i++) {
      clusterSize(g(i))++;
      for (int j=0; j<g(i); j++) {
        largerCount(j)++;
      }
    }
    for (i=0; i<nF; i++) {
      if (clusterSize(i) > 0) {
        nClusterRec(sim)++;
      }
      if (clusterSize(i) > 0.05 * size) {
        nClusterTRec(sim)++;
      }
    }
    temp = 0.0;
    for (j=0; j<nF-1; j++) {
      v(j) = ::Rf_rbeta(clusterSize(j)+1.0, largerCount(j)+alpha);
      temp += log(1.0 - v(j));
    }
    v(nF-1) = 1.0;
    probs = makeProbsHA(v);

    //Rcout << "updating the precision parameters... \n";
    //Gibbs sampling for the precision parameter of the first DP
    alpha = ::Rf_rgamma(nF-1.0+a1, 1.0 / (b1-temp));
    //Rcout<<D<<"\n";
    if (alpha != alpha) {
      alpha = 1.0;
    }
    if (alpha < 0.001) {
      alpha = 0.001;
    }

    //Rcout << "updating the scale parameters in the base measure... \n";
    //Gibbs sampling for the scale parameter in the base measure of the DP
    temp = 0.0;
    for (j=0; j<nF; j++){
      temp += 1.0 / sigmaStar(j);
     // //Rcout<<sigmaStar(j) << "\t";
    }
    //Rcout<<"\n";
    d = ::Rf_rgamma(a2 + nF * c, 1.0 / (b2 + temp));
   // //Rcout<<d<<"\t" << a2 + n * c << "\t" << 1.0 / (b2 + temp) <<"\t"<<b2<<"\n";




    for (i=0; i<3+ncov; i++) {
      if (att(i)>200 && 2*sim<burn) {
        can(i) = can(i) * (acc(i)/att(i)<0.2?0.5:1) * (acc(i)/att(i)>0.7?1.5:1);
        acc(i) = 1.0;
        att(i) = 1.0;
      }
    }

    //betaRec.row(sim) = Rcpp::as<Rcpp::NumericVector>(wrap(beta));

    //calculate the intercept
    if (sampleQ) {
      interceptAdj(sim) = adrejSamplerBetaHA(resid, p);
    }
    else {
      interceptAdj(sim) = findRootNewtonQHA(probs, qStar, sigmaStar, p, 0.000001);
    }
    if (interceptAdj(sim) != interceptAdj(sim)) {

      throw Rcpp::exception("NAN produced!!!");
    }
    //Rcout << "Finished" <<"\n";
    //free(gammaAdj);
    //free(tauAdj);
    //free(sigmaAdj);
    if (plotDensity) {
      if (sampleQ) {
        temp = findRootNewtonQHA(probs, qStar, sigmaStar, p, 0.000001);
      }
      else {
        temp = interceptAdj(sim);
      }
      if (sim > burn) {
        for (int j=0; j<nF; j++) {
          for (i=0; i<4001; i++) {
            sumdense(i) += (probs(j)*dLogisticHA(xGrid(i), p, -qStar(j)-temp, sigmaStar(j)))/(nsim-burn);
          }
        }
      }
    }

    betaRec.row(sim) = as<NumericVector>(wrap(beta));
    betaRec(sim, 0) = beta(0) + interceptAdj(sim) * gamma(0);
    for (i=1; i<ncov; i++) {
      betaRec(sim, i) = beta(i) + interceptAdj(sim) * gamma(i);
    }
    gammaRec.row(sim) = as<NumericVector>(wrap(gamma));
    sigmaRec.row(sim) = sigmaStar;
    qRec.row(sim) = qStar-beta(0)+interceptAdj(sim);

    //dRec(sim) = d;
    //clusterRec.row(sim) = g;
    alphaRec(sim) = alpha;
    //Rcout << "*************************************************\n";
    //Rcout << "\n";
  }
  //betaRecFile.close();
  aveN /= nsim;
  return List::create(Rcpp::Named("gamma") = gammaRec, Rcpp::Named("gammaAccRate") = nAcceptGamma/nsim, Rcpp::Named("dense.mean") = sumdense, Rcpp::Named("xGrid")=xGrid, Rcpp::Named("interceptAdj")=interceptAdj, Rcpp::Named("nRej") = nRejRec, Rcpp::Named("nCluster") = nClusterRec, Rcpp::Named("nClusterT") = nClusterTRec, Rcpp::Named("aveN") = aveN, //Rcpp::Named("beta1dx") = graph1x, Rcpp::Named("beta1dy") = graph1y, Rcpp::Named("beta2dx") = graph2x, Rcpp::Named("beta2dy") = graph2y, Rcpp::Named("beta1s") = graphS1, Rcpp::Named("beta2s") = graphS2,
  Rcpp::Named("d") = dRec, Rcpp::Named("lambda")=lambda, Rcpp::Named("alpha") = alphaRec, Rcpp::Named("beta") = betaRec, Rcpp::Named("sigma") = sigmaRec, Rcpp::Named("q") = qRec);       //Rcpp::Named("cluster") = clusterRec,  Rcpp::Named("nClusterT") = nClusterTRec,    // Return to R
}
