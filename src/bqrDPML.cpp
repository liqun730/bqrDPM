// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <string>
#include <iostream>
#include <fstream>
//#include <Rcpp.h>

using namespace Rcpp;
using namespace arma;
using namespace std;

const double thresholdInterval = -0.0000000001; //used in the adaptive reject sampling. If the length of the interval is smaller than this value,the interval will no longer be divided.

//calculate the sample quantile
double sampleQuantileL(NumericVector array, double p) {
  NumericVector data = clone(array);
  int n = (int)(data.size()*p);
  std::nth_element(data.begin(), data.begin()+n, data.end());
  double result = data(n);
  std::nth_element(data.begin(), data.begin()+n-1, data.end());
  result = (data(n) + data(n-1))/2;
  return result;
}

//the equation used to solve for the quantile
double fLambda(double x, double p){
  return x * log(x) + 1.0 - x - p * pow(x - 1.0, 2.0);
}

double dLogistic(double x, double p, double tau, double sigma) {
  double temp = -(x-tau)/sigma;
  if (temp > 0) {
    return exp(log((1.0-p)/p) - temp - log(sigma) - 2.0 * log((1.0-p)/p + exp(-temp)));
  }
  else {
    return exp(log((1.0-p)/p) + temp - log(sigma) - 2.0 * log(1.0 + exp(temp)*(1.0-p)/p));
  }
}

//the derivative of the function used to solve for the quantile
double fLambdaDer(double x, double p){
  return log(x) - 2 * p * (x - 1.0);
}

//the equation used to solve for the quantile of mixture of logistic distribution
double fq(double x, NumericVector gamma, NumericVector tau, NumericVector sigma, double p) {
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
  for (int i=0; i<n; i++) {
    value += gamma(i) / (p + (1.0-p) * exp(-(x-tau(i))/sigma(i)));
  }
  value -= 1.0;
  return value;
}

//the derivative of the equation used to solve for the quantile of mixture of logistic distribution
double fqDer(double x, NumericVector gamma, NumericVector tau, NumericVector sigma, double p) {
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
  for (int i=0; i<n; i++) {
    temp = (tau(i)-x)/sigma(i) - 2 * log(p+(1.0-p)*exp(-(x-tau(i))/sigma(i)));
    if (temp != temp) {
      throw Rcpp::exception("Something is wrong with fqDer!!!");
    }
    else {
      value += (1.0-p)/sigma(i) * exp(temp);
    }

    //value += gamma(i) * (1.0-p) * exp(-(x-tau(i))/sigma(i)) / sigma(i) / pow(p + (1.0-p) * exp(-(x-tau(i))/sigma(i)), 2.0);
  }
  return value;
}


double findRoughRootQ(NumericVector gamma, NumericVector tau, NumericVector sigma, double p, double threshold, double maxTau, double minTau){
  double left = minTau;
  double right = maxTau;
  if (fq(left, gamma, tau, sigma, p) > 0 || fq(right, gamma, tau, sigma, p) < 0) {
    throw Rcpp::exception("Something is wrong with findRoughRootQ!!!");
  }
  double mid = (left + right) / 2.0;
  while (right - left > threshold){
    if (fq(mid, gamma, tau, sigma, p) > 0){
      right = mid;
    }
    else if (fq(mid, gamma, tau, sigma, p) < 0){
      left = mid;
    }
    else {
      return mid;
    }
    mid = (left + right) / 2.0;
  }
  return mid;
}


double findRootNewtonQ(NumericVector gamma, NumericVector tau, NumericVector sigma, double p, double threshold){
  //use Newton method to solve the quantile of mixture of logistic distribution
  int n = gamma.size();
  if (n==1) {
    return tau(0);
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
  if (tau(0) > tau(1)) {
    maxTau = tau(0);
    minTau = tau(1);
  }
  else {
    maxTau = tau(1);
    minTau = tau(0);
  }
  for (int i=2; i< n; i++) {
    if (tau(i)>maxTau) {
      maxTau = tau(i);
    }
    if (tau(i)<minTau) {
      minTau = tau(i);
    }
  }

  //use the output of midpoint method as the initial value
  double xold = findRoughRootQ(gamma, tau, sigma, p, 0.00001, maxTau, minTau);
  double xnew = xold - fq(xold, gamma, tau, sigma, p) / fqDer(xold, gamma, tau, sigma, p);
  //Rcout << xold << "\t" << xnew << "\t" << fq(xold, gamma, tau, sigma, p) << "\t" << fqDer(xold, gamma, tau, sigma, p) << "\n";
  while (std::abs(xold - xnew) > threshold){
    xold = xnew;
    xnew = xold - fq(xold, gamma, tau, sigma, p) / fqDer(xold, gamma, tau, sigma, p);
    //Rcout << xold << "\t" << xnew << "\t" << fq(xold, gamma, tau, sigma, p) << "\t" << fqDer(xold, gamma, tau, sigma, p) << "\n";
  }
  if (xnew != xnew || xnew > maxTau || xnew < minTau) {
    return findRoughRootQ(gamma, tau, sigma, p, threshold, maxTau, minTau);
  }
  else {
    return xnew;
  }
}


//normalize a vector (which has been taken logrithm) such that they sum to one
NumericVector normalize(NumericVector x, int n){
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
double findRoughRoot(double p, double threshold){
  // p is not equal to 0.5
  // threshold controls when to stop the iteration
  if (p < 0.5){
    double left = 1.1;
    double right = 5.0;
    while (fLambda(left, p) < 0){
      left = sqrt(left);
    }
    while (fLambda(right, p) > 0){
      right = pow(right, 2.0);
    }
    double mid = (left + right) / 2.0;
    while (right - left > threshold){
      if (fLambda(mid, p) > 0){
        left = mid;
      }
      else if (fLambda(mid, p) < 0){
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
    while (fLambda(left, p) < 0){
      left = pow(left, 2.0);
    }
    while (fLambda(right, p) > 0){
      right = sqrt(right);
    }
    double mid = (left + right) / 2.0;
    while (right - left > threshold){
      if (fLambda(mid, p) > 0){
        left = mid;
      }
      else if (fLambda(mid, p) < 0){
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


double findRootNewton(double p, double threshold){
  //use Newton-Raphson method to solve for the lambda
  if (p == 0.5){
    return 1.0;
  }
  else {
    double xold = findRoughRoot(p, 0.1);
    double xnew = xold - fLambda(xold, p) / fLambdaDer(xold, p);
    while (std::abs(xold - xnew) > threshold){
      xold = xnew;
      xnew = xold - fLambda(xold, p) / fLambdaDer(xold, p);
      ////Rcout << std::abs(xold - xnew) << "\n";
    }
    return xnew;
  }
}



double LLq(double sigma, double q, arma::colvec resid, IntegerVector ind, double c, double d, double p, double lambda){
  int n = ind.size();
  double ll = -(n + 1.0) * q / sigma;
  double temp = -q / sigma;
  if (temp > 0){
    ll -= 2 * log((1.0 - p) / lambda / p + exp(-temp)) + 2 * temp;
  }
  else {
    ll -= 2 * log(1.0 + (1.0 - p) * exp(temp) / lambda / p);
  }
  for (int i=0; i<n; i++){
    temp = -(q + resid(ind(i))) / sigma;
    if (temp > 0){
      ll -= 2 * log((1.0 - p) / p + exp(-temp)) + 2 * temp;
    }
    else{
      ll -= 2 * log(1.0 + (1.0 - p) * exp(temp) / p);
    }
  }
  return ll;
}

double LLsigmaInv(double sigmaInv, double q, arma::colvec resid, IntegerVector ind, double c, double d, double p, double lambda){
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


double LLbeta(double beta, double var, int k, NumericVector q, NumericVector u, double p, NumericVector sigma, arma::mat X){
  int n = u.size();
  if (var <=0 ){
    throw Rcpp::exception("Variance should be positive!");
  }
  double ll = -pow(beta,2) / 2 / var;
  for (int i=0; i<n; i++){
    double temp = (u(i) - X(i,k) * beta + q(i)) / sigma(i);
    if (temp > 0) {
      ll -= temp + 2 * log(1.0 + (1.0 - p) * exp(-temp) / p );
    }
    else {
      ll -= 2 * log(exp(temp) + (1.0 - p) / p) - temp;
    }
  }
  return ll;
}



double LLqDer(double sigma, double q, arma::colvec resid, IntegerVector ind, double c, double d, double p, double lambda){
  int n = ind.size();
  double lld = -(n + 1.0) / sigma;
  double temp = -q / sigma;
  if (temp > 0){
    lld += 2 * (1.0-p) / lambda / p / sigma / (exp(-temp) + (1.0-p)/p/lambda);
  }
  else{
    lld +=  2 * (1.0-p) * exp(temp) / p / lambda / sigma / (1.0 + (1.0-p) * exp(temp) / p / lambda);
  }
  ////Rcout<<lambda<<"\t"<<sigma<<"\t"<<p<<"\t"<<q<<"  &&&&&&&&&&&&    "<<temp <<"\t"<< (1.0 + (1.0 - p) * temp / lambda / p)<<"\t"<<lld<<"\n";
  for (int i=0; i<n; i++){
    temp = -(resid(ind(i)) + q) / sigma;
    if (temp > 0){
      lld += 2 * (1.0 - p) / p / sigma / (exp(-temp) + (1.0-p)/p);
    }
    else{
      lld += 2 * (1.0 - p) * exp(temp) / p / sigma / (1.0 + (1.0-p)*exp(temp)/p);
    }
  }
  return lld;
}

double LLsigmaInvDer(double sigmaInv, double q, arma::colvec resid, IntegerVector ind, double c, double d, double p, double lambda){
  int n = ind.size();
  double sumR = 0.0;
  for (int i=0; i<n; i++){
    sumR += resid(ind(i));
  }
  double lld = -(sumR + (n + 1.0) * q + d) + (c + n) / sigmaInv;
  double temp = -q * sigmaInv;
  if (temp > 0){
    lld += 2 * q * (1.0 - p) / p / lambda / (exp(-temp) + (1.0-p)/p/lambda);
  }
  else {
    lld += 2 * q * (1.0 - p) * exp(temp) / p / lambda / (1.0 + (1.0-p)*exp(temp)/p/lambda);
  }
  for (int i=0; i<n; i++){
    temp = -(q + resid(ind(i))) * sigmaInv;
    if (temp > 0){
      lld += 2 * (resid(ind(i)) + q) * (1.0 - p) / p / (exp(-temp) + (1.0-p)/p);
    }
    else {
      lld += 2 * (resid(ind(i)) + q) * (1.0 - p) * exp(temp) / p / (1.0 + (1.0-p)*exp(temp)/p);
    }
  }
  return lld;
}

double LLbetaDer(double beta, double var, int k, NumericVector q, NumericVector u, double p, NumericVector sigma, arma::mat X){
  int n = u.size();
  if (var <=0 ){
    throw Rcpp::exception("Variance should be positive!");
  }
  double lld = -beta / var;
  for (int i=0; i<n; i++){
    double temp = (u(i) - X(i,k) * beta + q(i)) / sigma(i);
    ////Rcout<<lld<<"~1\t";
    lld += X(i,k) / sigma(i);
    ////Rcout<<lld<<"~2\t";

    if (temp > 0) {
      lld -= 2 * (1.0 - p) * X(i,k) * exp(-temp) / p / sigma(i) / (1.0 + (1.0 - p) * exp(-temp) / p);
    }
    else {
      lld -= 2 * (1.0 - p) * X(i,k) / p / sigma(i) / (exp(temp) + (1.0 - p) / p);
    }
    /*
    if (std::isinf(2 * X(i,k) / sigma(i) * (1.0 - p) * exp(-temp) / p)){
    lld -= 2 * X(i,k) / sigma(i);
    }
    else {
    lld -= 2 * (1.0 - p) * X(i,k) * exp(-temp) / p / sigma(i) / (1.0 + (1.0 - p) * exp(-temp) / p);
    }*/
    ////Rcout<<lld<<"~3\t";
    if (std::isinf(lld)) {
      //Rcout<<sigma(i) << "\t" <<p <<"\t" << temp<<"Got INF!\n";
      //Rcout<<2 * (1.0 - p) * X(i,k) / p / sigma(i) / (exp(temp) + (1.0 - p) / p)<<"\n";
      throw Rcpp::exception("Got INF!");
    }
  }
  return lld;
  }

NumericVector intercept(NumericVector x, NumericVector y, NumericVector slope, int nKnots){
  //find the consecutive (ordered by x) x values of intercept of lines passing through the given points with given slope
  //points is ordered by x values from smalldfsdfsdfsdfest to largest
  NumericVector interc(nKnots-1);
  double temp;
  double temp1;
  double temp2;
  for (int i=0; i<nKnots-1; i++){
    if (slope(i+1) == slope(i)){
      interc(i) = (x(i) + x(i+1)) / 2.0;
      /*Rcout<<x(i)<<"\t"<<y(i)<<"\t"<<slope(i)<<"\n";
      Rcout<<x(i+1)<<"\t"<<y(i+1)<<"\t"<<slope(i+1)<<"\n";
      Rcout<<"list\n";
      for (int j=0; j<nKnots; j++) {
      Rcout<<x(j)<<"\t"<<y(j)<<"\t"<<slope(j)<<"\n";
      }

      throw Rcpp::exception("There is something wrong with the intercept function1!");
      Rcout<<"There is something wrong with the intercept function1!\n";*/
    }
    else {
      //interc(i) = (y(i+1) - y(i) + slope(i) * x(i) - slope(i+1) * x(i+1)) / (slope(i) - slope(i+1));
      if (slope(i) != 0) {
        temp = slope(i+1) / slope(i);
        temp1 = (y(i) - y(i+1)) / slope(i) - (x(i)-x(i+1));
        temp2 = (temp - 1.0);
      }
      else {
        temp = slope(i) / slope(i+1);
        temp1 = (y(i) - y(i+1)) / slope(i+1) - temp * (x(i) - x(i+1));
        temp2 = 1.0 - temp;
      }

      interc(i) = x(i+1) + temp1 / temp2;
      //debug
      /* if (interc(i) != interc(i)) {
      Rcout<<x(i)<<"\t"<<y(i)<<"\t"<<slope(i)<<"\n";
      Rcout<<x(i+1)<<"\t"<<y(i+1)<<"\t"<<slope(i+1)<<"\n";
      Rcout<<log(temp1)<<"\n";
      throw Rcpp::exception("NA is produced when calculate the intercept!");
  }*/
      //debug

      //interc(i) = (y(i) - y(i+1) - slope(i) * x(i) + slope(i+1) * x(i+1)) / (slope(i+1) - slope(i));
      if (x(i+1) > x(i) && (interc(i) <= x(i) || interc(i) >= x(i+1))){
        interc(i) = (x(i) + x(i+1)) / 2.0;
      }
      if (x(i+1) < x(i) && (interc(i) >= x(i) || interc(i) <= x(i+1))){
        interc(i) = (x(i) + x(i+1)) / 2.0;
        throw Rcpp::exception("There is something wrong with the intercept function3!");
        Rcout<<"There is something wrong with the intercept function3!\n";
      }
}
    ////Rcout << "intercept xvalues "<< interc(i) << "\t" << (y(i) - y(i+1) - slope(i) * x(i) + slope(i+1) * x(i+1)) <<"\t" << slope(i+1) - slope(i) << "\n";
    }


  for (int i=1; i<nKnots-1; i++) {
    if (interc(i) < interc(i-1)) {
      for (int j=0; j<nKnots; j++) {
        Rcout<<x(j)<<"\t";
      }
      Rcout<<"\n";
      for (int j=0; j<nKnots-1; j++) {
        Rcout<<interc(j)<<"\t";
      }
      Rcout<<"\n";
      throw Rcpp::exception("There is something wrong with the intercept function!");
    }

    if (interc(i) != interc(i)) {
      throw Rcpp::exception("There is something wrong with the intercept function! NAN is produced!");
    }
  }
  return interc;
  }

NumericVector arsSample(bool isPositiveOnly, NumericVector knotsI, NumericVector knotsS, NumericVector knotsB, NumericVector prob, int nKnots){
  //the adaptive rejection sampling when the knots are given
  //the first element of returned vector is the sample drawn from the cover distribution.
  //the second element of the returned vector is the derivative evaluated at the sample drawn.
  //the third element of the returned vector indices which part the sample belongs to.
  double u = R::runif(0.0, 1.0);
  NumericVector sample(3);
  ////Rcout<<"prob = "<<u <<"\t"<<1-  prob(nKnots-1)<<"\n";
  if (u <= prob(0)){
    double uf = R::runif(0.0, 1.0);
    if (isPositiveOnly) {
      sample(0) = knotsI(0) + log(uf + (1.0 - uf) * exp(-knotsS(0) * knotsI(0))) / knotsS(0);
    }
    else {
      sample(0) = knotsI(0) + log(uf) / knotsS(0);
    }
    sample(1) = knotsS(0) * sample(0) + knotsB(0);
    sample(2) = double(0);
    // //Rcout<<1<<"\t"<<sample(0)<<"\t"<<knotsI(0)<<"\t"<<knotsS(0)<<"\t"<<knotsB(0)<<"\n";
    ////Rcout<<1<<"\t"<<sample(0)<<"\t"<<u<<"\t"<<uf<<"\n";
  }
  else if (u + prob(nKnots-1) > 1){
    double uf = R::runif(0.0, 1.0);
    sample(0) = knotsI(nKnots-2) + log(1.0 - uf) / knotsS(nKnots-1);
    sample(1) = knotsS(nKnots-1) * sample(0) + knotsB(nKnots-1);
    sample(2) = double(nKnots-1);
    // //Rcout<<2<<"\t"<<sample(0)<<"\t"<<knotsI(nKnots-2)<<"\t"<<knotsS(nKnots-1)<<"\t"<<knotsB(nKnots-1)<<"\n";
    ////Rcout<<2<<"\t"<<sample(0)<<"\t"<<u<<"\t"<<uf<<"\n";
  }
  else {
    double cumProb = prob(0);
    for (int j=1; j<nKnots-1; j++){
      if (u > cumProb && u <= cumProb + prob(j)){
        double uf = R::runif(0.0, 1.0);
        //sample(0) =  log(uf * exp(knotsS(j) * knotsI(j)) + (1.0 - uf) * exp(knotsS(j) * knotsI(j-1))) / knotsS(j);
        if (knotsS(j) == 0) {
          sample(0) = knotsI(j) * uf + (knotsI(j-1) * (1.0 - uf));
        }
        else {
          double a = knotsS(j) * knotsI(j);
          double b = knotsS(j) * knotsI(j-1);
          if (a == 0) {
            if (b > 0) {
              sample(0) = b + log((1.0 - uf) + uf * exp(a - b));
            }
            else{
              sample(0) = a + log(uf + (1.0 - uf) * exp(b - a));
            }
          }
          else if (b == 0) {
            if (a > 0) {
              sample(0) = a + log(uf + (1.0 - uf) * exp(b - a));
            }
            else {
              sample(0) = b + log((1.0 - uf) + uf * exp(a - b));
            }
          }
          else if (a < 0 && b <0) {
            if (a < b) {
              sample(0) = b + log((1.0 - uf) + uf * exp(a - b));
            }
            else {
              sample(0) = a + log(uf + (1.0 - uf) * exp(b - a));
            }
          }
          else if (a>0 && b>0) {
            if (a < b) {
              sample (0) = b + log((1.0 - uf) + uf * exp(a - b));
            }
            else {
              sample(0) = a + log(uf + (1.0 - uf) * exp(b - a));
            }
          }
          else if (a>0 && b<0) {
            sample(0) = a + log(uf + (1.0 - uf) * exp(b - a));
          }
          else {
            sample (0) = b + log((1.0 - uf) + uf * exp(a - b));
          }
          sample(0) = sample(0) / knotsS(j);
        }
        sample(1) = knotsS(j) * sample(0) + knotsB(j);
        sample(2) = double(j);
        break;
      }
      cumProb += prob(j);
    }
  }
  return sample;
}


bool calProbAndUpdateKnots(int which, double var, double sigma, arma::colvec resid, IntegerVector ind, double c, double d, double p, double q, double lambda, int k, NumericVector Q, NumericVector U, NumericVector Sigma, arma::mat X ,double u, double yD, int nKnots, NumericVector knotsX, NumericVector knotsY, NumericVector knotsS, NumericVector knotsB, NumericVector knotsI, NumericVector auc, NumericVector sample) {
  int loc = int(sample(2)); // indicate which parts the sample is drawn from
  for (int i=nKnots; i>loc+1; i--){
    knotsX(i) = knotsX(i-1);
    knotsY(i) = knotsY(i-1);
    knotsS(i) = knotsS(i-1);
    knotsB(i) = knotsB(i-1);
    auc(i) = auc(i-1);
    knotsI(i-1) = knotsI(i-2);
  }
  if (loc == 0){
    if (knotsX(loc) > sample(0)){
      knotsX(loc+1) = knotsX(loc);
      knotsY(loc+1) = knotsY(loc);
      knotsS(loc+1) = knotsS(loc);
      knotsB(loc+1) = knotsB(loc);
      knotsX(loc) = sample(0);
      knotsY(loc) = yD;
      if (which == 1) {
        knotsS(loc) = LLqDer(sigma, sample(0), resid, ind, c, d, p, lambda);
      }
      else if (which == 2) {
        knotsS(loc) = LLsigmaInvDer(sample(0), q, resid, ind, c, d, p, lambda);
      }
      else if (which == 3) {
        knotsS(loc) = LLbetaDer(sample(0), var, k, Q, U, p, Sigma, X);
      }
      else {
        throw Rcpp::exception("The integer argument which can only be 1, 2, 3!!!");
      }

      knotsB(loc) = knotsY(loc) - knotsS(loc) * knotsX(loc);
      if (knotsS(loc+1) - knotsS(loc) == 0){
        knotsI(loc) = (knotsX(loc) + knotsX(loc+1)) / 2.0;
      }
      else {
        knotsI(loc) = (knotsY(loc) - knotsY(loc+1) - knotsS(loc) * knotsX(loc) + knotsS(loc+1) * knotsX(loc+1)) / (knotsS(loc+1) - knotsS(loc));
        if (knotsX(loc+1) > knotsX(loc) && (knotsI(loc) <= knotsX(loc) || knotsI(loc) >= knotsX(loc+1))){
          knotsI(loc) = (knotsX(loc) + knotsX(loc+1)) / 2.0;
        }
        if (knotsX(loc+1) < knotsX(loc) && (knotsI(loc) >= knotsX(loc) || knotsI(loc) <= knotsX(loc+1))){
          knotsI(loc) = (knotsX(loc) + knotsX(loc+1)) / 2.0;
        }
      }


      if (knotsS(loc+1) - knotsS(loc) == 0){
        throw Rcpp::exception("Something is wrong!\tA 3");
      }
      if (knotsS(loc+1) == 0){
        //auc(loc+1) = exp(knotsB(loc+1)) * (knotsI(loc+1) - knotsI(loc));
        auc(loc+1) = knotsB(loc+1) + log(knotsI(loc+1) - knotsI(loc));
      }
      else {
        //auc(loc+1) = (exp(knotsB(loc+1) + knotsS(loc+1) * knotsI(loc+1)) - exp(knotsB(loc+1) + knotsS(loc+1) * knotsI(loc))) / knotsS(loc+1);
        //auc(loc+1) = (exp(knotsB(loc+1) + knotsS(loc+1) * knotsI(loc+1)) * (1.0 - exp(knotsS(loc+1) * (knotsI(loc) - knotsI(loc+1))))) / knotsS(loc+1);
        if (knotsS(loc+1) > 0){
          auc(loc+1) = knotsB(loc+1) + knotsS(loc+1) * knotsI(loc+1) + log(1.0 - exp(knotsS(loc+1) * (knotsI(loc) - knotsI(loc+1)))) - log(knotsS(loc+1));
        }
        else {
          auc(loc+1) = knotsB(loc+1) + knotsS(loc+1) * knotsI(loc) + log(1.0 - exp(knotsS(loc+1) * (knotsI(loc+1) - knotsI(loc)))) - log(-knotsS(loc+1));
        }
      }
      if (knotsS(loc) == 0){
        throw Rcpp::exception("Something is wrong! 4 ");
      }
      //auc(loc) = exp(knotsB(loc) + knotsS(loc) * knotsI(loc)) / knotsS(loc);
      auc(loc) = knotsB(loc) + knotsS(loc) * knotsI(loc) - log(knotsS(loc)); // knotsS(loc) must be positive
      // //Rcout <<"@@@@" <<auc(0)<<"\t"<<auc(1)<<"\n";
    }
    else {
      knotsX(loc+1) = sample(0);
      knotsY(loc+1) = yD;
      if (which == 1) {
        knotsS(loc+1) = LLqDer(sigma, sample(0), resid, ind, c, d, p, lambda);
      }
      else if (which == 2) {
        knotsS(loc+1) = LLsigmaInvDer(sample(0), q, resid, ind, c, d, p, lambda);
      }
      else if (which == 3) {
        knotsS(loc+1) = LLbetaDer(sample(0), var, k, Q, U, p, Sigma, X);
      }
      else {
        throw Rcpp::exception("The integer argument which can only be 1, 2, 3!!!");
      }

      knotsB(loc+1) = knotsY(loc+1) - knotsS(loc+1) * knotsX(loc+1);
      if (knotsS(loc+2) - knotsS(loc+1) == 0){
        knotsI(loc+1) = (knotsX(loc+1) + knotsX(loc+2)) / 2.0;
      }
      else {
        knotsI(loc+1) = (knotsY(loc+1) - knotsY(loc+2) - knotsS(loc+1) * knotsX(loc+1) + knotsS(loc+2) * knotsX(loc+2)) / (knotsS(loc+2) - knotsS(loc+1));
        if (knotsX(loc+2) > knotsX(loc+1) && (knotsI(loc+1) <= knotsX(loc+1) || knotsI(loc+1) >= knotsX(loc+2))){
          knotsI(loc+1) = (knotsX(loc+1) + knotsX(loc+2)) / 2.0;
        }
        if (knotsX(loc+2) < knotsX(loc+1) && (knotsI(loc+1) >= knotsX(loc+1) || knotsI(loc+1) <= knotsX(loc+2))){
          knotsI(loc+1) = (knotsX(loc+1) + knotsX(loc+2)) / 2.0;
        }
      }

      if (knotsS(loc+1) - knotsS(loc) == 0){
        knotsI(loc) = (knotsX(loc) + knotsX(loc+1)) / 2.0;
      }
      else {
        knotsI(loc) = (knotsY(loc) - knotsY(loc+1) - knotsS(loc) * knotsX(loc) + knotsS(loc+1) * knotsX(loc+1)) / (knotsS(loc+1) - knotsS(loc));
        if (knotsX(loc+1) > knotsX(loc) && (knotsI(loc) <= knotsX(loc) || knotsI(loc) >= knotsX(loc+1))){
          knotsI(loc) = (knotsX(loc) + knotsX(loc+1)) / 2.0;
        }
        if (knotsX(loc+1) < knotsX(loc) && (knotsI(loc) >= knotsX(loc) || knotsI(loc) <= knotsX(loc+1))){
          knotsI(loc) = (knotsX(loc) + knotsX(loc+1)) / 2.0;
        }
      }

      if (knotsS(loc+2) - knotsS(loc+1) == 0){
        //Rcout<<sample(0)<<"\t"<<sample(1)<<"\t"<<knotsX(loc+1)<<"\t"<<yD<<"\t"<<knotsY(loc+1)<<"\t"<<exp(yD - sample(1))<<"\t"<<u<<"\n";
        //Rcout << loc <<"\n";
        for (int i=0; i<nKnots; i++) {
          //Rcout<<knotsX(i) <<"\t";
        }
        //Rcout <<"knotsX\n";
        for (int i=0; i<nKnots; i++) {
          //Rcout<<knotsS(i) <<"\t";
        }
        //Rcout <<"knotsS\n";
        for (int i=0; i<nKnots; i++) {
          //Rcout<<knotsI(i) <<"\t";
        }
        //Rcout <<"knotsI\n";
        //Rcout <<"Please Check for Bugs!";
        for (int i=0; i<1; i++) {
          //Rcout << "ATTENTION!!\n";
        }
        if (sample(0) == 0 && sample(1) == 0 && sample(2) == 0) {
          throw Rcpp::exception("Something is wrong!\tB 5");
        }
        return true;
        throw Rcpp::exception("Something is wrong!\tB 5");
      }
      if (knotsS(loc+1) - knotsS(loc) == 0){
        throw Rcpp::exception("Something is wrong!\tC 6");
      }

      if (knotsS(loc) == 0){
        throw Rcpp::exception("Something is wrong! 7");
      }
      //auc(loc) = exp(knotsB(loc) + knotsS(loc) * knotsI(loc)) / knotsS(loc);
      auc(loc) = knotsB(loc) + knotsS(loc) * knotsI(loc) - log(knotsS(loc)); // knotsS(loc) must be positive
      if (knotsS(loc+2) == 0){
        //auc(loc+2) = exp(knotsB(loc+2)) * (knotsI(loc+2) - knotsI(loc+1));
        auc(loc+2) = knotsB(loc+2) + log(knotsI(loc+2) - knotsI(loc+1));
      }
      else {
        //auc(loc+2) = (exp(knotsB(loc+2) + knotsS(loc+2) * knotsI(loc+2)) - exp(knotsB(loc+2) + knotsS(loc+2) * knotsI(loc+1))) / knotsS(loc+2);
        //auc(loc+2) = (exp(knotsB(loc+2) + knotsS(loc+2) * knotsI(loc+2)) * (1.0 - exp(knotsS(loc+2) * (knotsI(loc+1) - knotsI(loc+2))))) / knotsS(loc+2);
        if (knotsS(loc+2) > 0){
          auc(loc+2) = knotsB(loc+2) + knotsS(loc+2) * knotsI(loc+2) + log(1.0 - exp(knotsS(loc+2) * (knotsI(loc+1) - knotsI(loc+2)))) - log(knotsS(loc+2));
        }
        else {
          auc(loc+2) = knotsB(loc+2) + knotsS(loc+2) * knotsI(loc+1) + log(1.0 - exp(knotsS(loc+2) * (knotsI(loc+2) - knotsI(loc+1)))) - log(-knotsS(loc+2));
        }
      }
      if (knotsS(loc+1) == 0){
        //auc(loc+1) = exp(knotsB(loc+1)) * (knotsI(loc+1) - knotsI(loc));
        auc(loc+1) = knotsB(loc+1) + log(knotsI(loc+1) - knotsI(loc));
      }
      else {
        //auc(loc+1) = (exp(knotsB(loc+1) + knotsS(loc+1) * knotsI(loc+1)) - exp(knotsB(loc+1) + knotsS(loc+1) * knotsI(loc))) / knotsS(loc+1);
        //auc(loc+1) = (exp(knotsB(loc+1) + knotsS(loc+1) * knotsI(loc+1)) * (1.0 - exp(knotsS(loc+1) * (knotsI(loc) - knotsI(loc+1))))) / knotsS(loc+1);
        if (knotsS(loc+1) > 0){
          auc(loc+1) = knotsB(loc+1) + knotsS(loc+1) * knotsI(loc+1) + log(1.0 - exp(knotsS(loc+1) * (knotsI(loc) - knotsI(loc+1)))) - log(knotsS(loc+1));
        }
        else {
          auc(loc+1) = knotsB(loc+1) + knotsS(loc+1) * knotsI(loc) + log(1.0 - exp(knotsS(loc+1) * (knotsI(loc+1) - knotsI(loc)))) - log(-knotsS(loc+1));
        }
      }
      // //Rcout <<"@@@@" <<auc(0)<<"\t"<<auc(1)<<"\t"<<auc(2)<<"\n";
    }
  }
  else if (loc == nKnots-1){
    if (knotsX(loc) > sample(0)){
      knotsX(loc+1) = knotsX(loc);
      knotsY(loc+1) = knotsY(loc);
      knotsS(loc+1) = knotsS(loc);
      knotsB(loc+1) = knotsB(loc);
      knotsX(loc) = sample(0);
      knotsY(loc) = yD;
      if (which == 1) {
        knotsS(loc) = LLqDer(sigma, sample(0), resid, ind, c, d, p, lambda);
      }
      else if (which == 2) {
        knotsS(loc) = LLsigmaInvDer(sample(0), q, resid, ind, c, d, p, lambda);
      }
      else if (which == 3) {
        knotsS(loc) = LLbetaDer(sample(0), var, k, Q, U, p, Sigma, X);
      }
      else {
        throw Rcpp::exception("The integer argument which can only be 1, 2, 3!!!");
      }
      knotsB(loc) = knotsY(loc) - knotsS(loc) * knotsX(loc);
      if (knotsS(loc+1) - knotsS(loc) == 0){
        knotsI(loc) = (knotsX(loc) + knotsX(loc+1)) / 2.0;
      }
      else {
        knotsI(loc) = (knotsY(loc) - knotsY(loc+1) - knotsS(loc) * knotsX(loc) + knotsS(loc+1) * knotsX(loc+1)) / (knotsS(loc+1) - knotsS(loc));
        if (knotsX(loc+1) > knotsX(loc) && (knotsI(loc) <= knotsX(loc) || knotsI(loc) >= knotsX(loc+1))){
          knotsI(loc) = (knotsX(loc) + knotsX(loc+1)) / 2.0;
        }
        if (knotsX(loc+1) < knotsX(loc) && (knotsI(loc) >= knotsX(loc) || knotsI(loc) <= knotsX(loc+1))){
          knotsI(loc) = (knotsX(loc) + knotsX(loc+1)) / 2.0;
        }
      }
      //knotsI()knotsI()knotsI()knotsI()knotsI()knotsI()knotsI()knotsI()knotsI()knotsI()knotsI()knotsI()knotsI()knotsI()knotsI()knotsI()

      knotsI(loc-1) = (knotsY(loc-1) - knotsY(loc) - knotsS(loc-1) * knotsX(loc-1) + knotsS(loc) * knotsX(loc)) / (knotsS(loc) - knotsS(loc-1));
      if (knotsS(loc+1) - knotsS(loc) == 0){
        //Rcout << loc <<"\t"<<sample(0)<<"\t"<<sample(1)<<"\t"<<sample(2)<<"\n";
        for (int i=0; i<nKnots; i++) {
          //Rcout<<knotsX(i) <<"\t";
        }
        //Rcout <<"knotsX\n";
        for (int i=0; i<nKnots; i++) {
          //Rcout<<knotsS(i) <<"\t";
        }
        //Rcout <<"knotsS\n";
        for (int i=0; i<nKnots; i++) {
          //Rcout<<knotsI(i) <<"\t";
        }
        //Rcout <<"knotsI\n";

        //Rcout <<"Please Check for Bugs!";
        for (int i=0; i<1; i++) {
          //Rcout << "ATTENTION!!\n";
        }
        if (sample(0) == 0 && sample(1) == 0 && sample(2) == 0) {
          throw Rcpp::exception("Something is wrong!tD 8");
        }
        return true;
        throw Rcpp::exception("Something is wrong!\tD 8");
      }
      if (knotsS(loc) - knotsS(loc-1) == 0){
        //Rcout << loc <<"\t"<<sample(0)<<"\t"<<sample(1)<<"\t"<<sample(2)<<"\n";
        for (int i=0; i<nKnots; i++) {
          //Rcout<<knotsX(i) <<"\t";
        }
        //Rcout <<"knotsX\n";
        for (int i=0; i<nKnots; i++) {
          //Rcout<<knotsS(i) <<"\t";
        }
        //Rcout <<"knotsS\n";
        for (int i=0; i<nKnots; i++) {
          //Rcout<<knotsI(i) <<"\t";
        }
        //Rcout <<"knotsI\n";

        //Rcout <<"Please Check for Bugs!";
        for (int i=0; i<1; i++) {
          //Rcout << "ATTENTION!!\n";
        }
        if (sample(0) == 0 && sample(1) == 0 && sample(2) == 0) {
          throw Rcpp::exception("Something is wrong!\tE 9");
        }
        return true;
        throw Rcpp::exception("Something is wrong!\tE 9");
      }
      if (knotsS(loc+1) == 0){
        throw Rcpp::exception("Something is wrong! 10");
      }
      //auc(loc+1) = -exp(knotsB(loc+1) + knotsS(loc+1) * knotsI(loc)) / knotsS(loc+1);
      auc(loc+1) = knotsB(loc+1) + knotsS(loc+1) * knotsI(loc) - log(-knotsS(loc+1)); // knotsS(loc+1) must be negative
      if (knotsS(loc) == 0){
        //auc(loc) = exp(knotsB(loc)) * (knotsI(loc) - knotsI(loc-1));
        auc(loc) = knotsB(loc) + log(knotsI(loc) - knotsI(loc-1));
      }
      else {
        //auc(loc) = (exp(knotsB(loc) + knotsS(loc) * knotsI(loc)) - exp(knotsB(loc) + knotsS(loc) * knotsI(loc-1))) / knotsS(loc);
        //auc(loc) = (exp(knotsB(loc) + knotsS(loc) * knotsI(loc)) * (1.0 - exp(knotsS(loc) * (knotsI(loc-1) - knotsI(loc))))) / knotsS(loc);
        if (knotsS(loc) > 0){
          auc(loc) = knotsB(loc) + knotsS(loc) * knotsI(loc) + log(1.0 - exp(knotsS(loc) * (knotsI(loc-1) - knotsI(loc)))) - log(knotsS(loc));
        }
        else {
          auc(loc) = knotsB(loc) + knotsS(loc) * knotsI(loc-1) + log(1.0 - exp(knotsS(loc) * (knotsI(loc) - knotsI(loc-1)))) - log(-knotsS(loc));
        }
      }
      if (knotsS(loc-1) == 0){
        //auc(loc-1) = exp(knotsB(loc-1)) * (knotsI(loc-1) - knotsI(loc-2));
        auc(loc-1) = knotsB(loc-1) + log(knotsI(loc-1) - knotsI(loc-2));
      }
      else {
        //auc(loc-1) = (exp(knotsB(loc-1) + knotsS(loc-1) * knotsI(loc-1)) - exp(knotsB(loc-1) + knotsS(loc-1) * knotsI(loc-2))) / knotsS(loc-1);
        //auc(loc-1) = (exp(knotsB(loc-1) + knotsS(loc-1) * knotsI(loc-1)) * (1.0 - exp(knotsS(loc-1) * (knotsI(loc-2) - knotsI(loc-1))))) / knotsS(loc-1);
        if (knotsS(loc-1) > 0){
          auc(loc-1) = knotsB(loc-1) + knotsS(loc-1) * knotsI(loc-1) + log(1.0 - exp(knotsS(loc-1) * (knotsI(loc-2) - knotsI(loc-1)))) - log(knotsS(loc-1));
        }
        else {
          auc(loc-1) = knotsB(loc-1) + knotsS(loc-1) * knotsI(loc-2) + log(1.0 - exp(knotsS(loc-1) * (knotsI(loc-1) - knotsI(loc-2)))) - log(-knotsS(loc-1));
        }
      }
    }
    else {
      knotsX(loc+1) = sample(0);
      knotsY(loc+1) = yD;
      if (which == 1) {
        knotsS(loc+1) = LLqDer(sigma, sample(0), resid, ind, c, d, p, lambda);
      }
      else if (which == 2) {
        knotsS(loc+1) = LLsigmaInvDer(sample(0), q, resid, ind, c, d, p, lambda);
      }
      else if (which == 3) {
        knotsS(loc+1) = LLbetaDer(sample(0), var, k, Q, U, p, Sigma, X);
      }
      else {
        throw Rcpp::exception("The integer argument which can only be 1, 2, 3!!!");
      }
      knotsB(loc+1) = knotsY(loc+1) - knotsS(loc+1) * knotsX(loc+1);
      knotsI(loc) = (knotsY(loc) - knotsY(loc+1) - knotsS(loc) * knotsX(loc) + knotsS(loc+1) * knotsX(loc+1)) / (knotsS(loc+1) - knotsS(loc));
      if (knotsS(loc+1) - knotsS(loc) == 0){
        throw Rcpp::exception("Something is wrong!\tF 11");
      }
      if (knotsS(loc+1) == 0){
        throw Rcpp::exception("Something is wrong! 12");
      }
      //auc(loc+1) = -exp(knotsB(loc+1) + knotsS(loc+1) * knotsI(loc)) / knotsS(loc+1);
      auc(loc+1) = knotsB(loc+1) + knotsS(loc+1) * knotsI(loc) - log(-knotsS(loc+1)); // knotsS(loc+1) must be negative
      if (knotsS(loc) == 0){
        //auc(loc) = exp(knotsB(loc)) * (knotsI(loc) - knotsI(loc-1));
        auc(loc) = knotsB(loc) + log(knotsI(loc) - knotsI(loc-1));
      }
      else {
        //auc(loc) = (exp(knotsB(loc) + knotsS(loc) * knotsI(loc)) - exp(knotsB(loc) + knotsS(loc) * knotsI(loc-1))) / knotsS(loc);
        //auc(loc) = (exp(knotsB(loc) + knotsS(loc) * knotsI(loc)) * (1.0 - exp(knotsS(loc) * (knotsI(loc-1) - knotsI(loc))))) / knotsS(loc);
        if (knotsS(loc) > 0){
          auc(loc) = knotsB(loc) + knotsS(loc) * knotsI(loc) + log(1.0 - exp(knotsS(loc) * (knotsI(loc-1) - knotsI(loc)))) - log(knotsS(loc));
        }
        else {
          auc(loc) = knotsB(loc) + knotsS(loc) * knotsI(loc-1) + log(1.0 - exp(knotsS(loc) * (knotsI(loc) - knotsI(loc-1)))) - log(-knotsS(loc));
        }
      }
      ////Rcout<<"---------   "<<auc(loc)<<"\t"<<auc(loc+1)<<"\t"<<knotsB(loc+1)<<"\t"<<knotsS(loc+1)<<"\t"<<knotsI(loc)<<"\t"<<knotsX(loc)<<"\t"<<knotsY(loc)<<"\t"<<sigma<<"\n";
      //aucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucaucauc
    }
  }
  else if (knotsX(loc) > sample(0)){
    knotsX(loc+1) = knotsX(loc);
    knotsY(loc+1) = knotsY(loc);
    knotsS(loc+1) = knotsS(loc);
    knotsB(loc+1) = knotsB(loc);
    knotsX(loc) = sample(0);
    knotsY(loc) = yD;
    if (which == 1) {
      knotsS(loc) = LLqDer(sigma, sample(0), resid, ind, c, d, p, lambda);
    }
    else if (which == 2) {
      knotsS(loc) = LLsigmaInvDer(sample(0), q, resid, ind, c, d, p, lambda);
    }
    else if (which == 3) {
      knotsS(loc) = LLbetaDer(sample(0), var, k, Q, U, p, Sigma, X);
    }
    else {
      throw Rcpp::exception("The integer argument which can only be 1, 2, 3!!!");
    }


    knotsB(loc) = knotsY(loc) - knotsS(loc) * knotsX(loc);
    knotsI(loc) = (knotsY(loc) - knotsY(loc+1) - knotsS(loc) * knotsX(loc) + knotsS(loc+1) * knotsX(loc+1)) / (knotsS(loc+1) - knotsS(loc));

    knotsI(loc-1) = (knotsY(loc-1) - knotsY(loc) - knotsS(loc-1) * knotsX(loc-1) + knotsS(loc) * knotsX(loc)) / (knotsS(loc) - knotsS(loc-1));

    if (knotsS(loc+1) - knotsS(loc) == 0){
      //Rcout << loc <<"\t"<<sample(0)<<"\t"<<sample(1)<<"\t"<<sample(2)<<"\n";
      for (int i=0; i<nKnots; i++) {
        //Rcout<<knotsX(i) <<"\t";
      }
      //Rcout <<"knotsX\n";
      for (int i=0; i<nKnots; i++) {
        //Rcout<<knotsS(i) <<"\t";
      }
      //Rcout <<"knotsS\n";
      for (int i=0; i<nKnots; i++) {
        //Rcout<<knotsI(i) <<"\t";
      }
      //Rcout <<"knotsI\n";

      //Rcout <<"Please Check for Bugs!";
      for (int i=0; i<1; i++) {
        //Rcout << "ATTENTION!!\n";
      }
      if (sample(0) == 0 && sample(1) == 0 && sample(2) == 0) {
        throw Rcpp::exception("Something is wrong!\tG 14");
      }
      return true;
      throw Rcpp::exception("Something is wrong!\tG 14");
    }
    if (knotsS(loc) - knotsS(loc-1) == 0){
      //Rcout << loc <<"\t"<<sample(0)<<"\t"<<sample(1)<<"\t"<<sample(2)<<"\n";
      for (int i=0; i<nKnots; i++) {
        //Rcout<<knotsX(i) <<"\t";
      }
      //Rcout <<"knotsX\n";
      for (int i=0; i<nKnots; i++) {
        //Rcout<<knotsS(i) <<"\t";
      }
      //Rcout <<"knotsS\n";
      for (int i=0; i<nKnots; i++) {
        //Rcout<<knotsI(i) <<"\t";
      }
      //Rcout <<"knotsI\n";
      //Rcout <<"Please Check for Bugs!";
      for (int i=0; i<1; i++) {
        //Rcout << "ATTENTION!!\n";
      }
      if (sample(0) == 0 && sample(1) == 0 && sample(2) == 0) {
        throw Rcpp::exception("Something is wrong!\tH 15");
      }
      return true;
      throw Rcpp::exception("Something is wrong!\tH 15");
    }
    if (knotsS(loc+1) == 0){
      //auc(loc+1) = exp(knotsB(loc+1)) * (knotsI(loc+1) - knotsI(loc));
      auc(loc+1) = knotsB(loc+1) + log(knotsI(loc+1) - knotsI(loc));
    }
    else {
      //auc(loc+1) = (exp(knotsB(loc+1) + knotsS(loc+1) * knotsI(loc+1)) - exp(knotsB(loc+1) + knotsS(loc+1) * knotsI(loc))) / knotsS(loc+1);
      //auc(loc+1) = (exp(knotsB(loc+1) + knotsS(loc+1) * knotsI(loc+1)) * (1.0 - exp(knotsS(loc+1) * (knotsI(loc) - knotsI(loc+1))))) / knotsS(loc+1);
      if (knotsS(loc+1) > 0){
        auc(loc+1) = knotsB(loc+1) + knotsS(loc+1) * knotsI(loc+1) + log(1.0 - exp(knotsS(loc+1) * (knotsI(loc) - knotsI(loc+1)))) - log(knotsS(loc+1));
      }
      else {
        auc(loc+1) = knotsB(loc+1) + knotsS(loc+1) * knotsI(loc) + log(1.0 - exp(knotsS(loc+1) * (knotsI(loc+1) - knotsI(loc)))) - log(-knotsS(loc+1));
      }
    }
    if (knotsS(loc) == 0){
      //auc(loc) = exp(knotsB(loc)) * (knotsI(loc) - knotsI(loc-1));
      auc(loc) = knotsB(loc) + log(knotsI(loc) - knotsI(loc-1));
    }
    else {
      //auc(loc) = (exp(knotsB(loc) + knotsS(loc) * knotsI(loc)) - exp(knotsB(loc) + knotsS(loc) * knotsI(loc-1))) / knotsS(loc);
      //auc(loc) = (exp(knotsB(loc) + knotsS(loc) * knotsI(loc)) * (1.0 - exp(knotsS(loc) * (knotsI(loc-1) - knotsI(loc))))) / knotsS(loc);
      if (knotsS(loc) > 0){
        auc(loc) = knotsB(loc) + knotsS(loc) * knotsI(loc) + log(1.0 - exp(knotsS(loc) * (knotsI(loc-1) - knotsI(loc)))) - log(knotsS(loc));
      }
      else {
        auc(loc) = knotsB(loc) + knotsS(loc) * knotsI(loc-1) + log(1.0 - exp(knotsS(loc) * (knotsI(loc) - knotsI(loc-1)))) - log(-knotsS(loc));
      }
    }
    if (loc == 1){
      //auc(loc-1) = exp(knotsB(loc-1) + knotsS(loc-1) * knotsI(loc-1)) / knotsS(loc-1);
      auc(loc-1) = knotsB(loc-1) + knotsS(loc-1) * knotsI(loc-1) - log(knotsS(loc-1)); // knotsS(loc-1) must be positive
    }
    else {
      if (knotsS(loc-1) == 0){
        //auc(loc-1) = exp(knotsB(loc-1)) * (knotsI(loc-1) - knotsI(loc-2));
        auc(loc-1) = knotsB(loc-1) + log(knotsI(loc-1) - knotsI(loc-2));
      }
      else {
        //auc(loc-1) = (exp(knotsB(loc-1) + knotsS(loc-1) * knotsI(loc-1)) - exp(knotsB(loc-1) + knotsS(loc-1) * knotsI(loc-2))) / knotsS(loc-1);
        //auc(loc-1) = (exp(knotsB(loc-1) + knotsS(loc-1) * knotsI(loc-1)) * (1.0 - exp(knotsS(loc-1) * (knotsI(loc-2) - knotsI(loc-1))))) / knotsS(loc-1);
        if (knotsS(loc-1) > 0){
          auc(loc-1) = knotsB(loc-1) + knotsS(loc-1) * knotsI(loc-1) + log(1.0 - exp(knotsS(loc-1) * (knotsI(loc-2) - knotsI(loc-1)))) - log(knotsS(loc-1));
        }
        else {
          auc(loc-1) = knotsB(loc-1) + knotsS(loc-1) * knotsI(loc-2) + log(1.0 - exp(knotsS(loc-1) * (knotsI(loc-1) - knotsI(loc-2)))) - log(-knotsS(loc-1));
        }
      }
    }
  }
  else {

    knotsX(loc+1) = sample(0);
    knotsY(loc+1) = yD;
    if (which == 1) {
      knotsS(loc+1) = LLqDer(sigma, sample(0), resid, ind, c, d, p, lambda);
    }
    else if (which == 2) {
      knotsS(loc+1) = LLsigmaInvDer(sample(0), q, resid, ind, c, d, p, lambda);
    }
    else if (which == 3) {
      knotsS(loc+1) = LLbetaDer(sample(0), var, k, Q, U, p, Sigma, X);
    }
    else {
      throw Rcpp::exception("The integer argument which can only be 1, 2, 3!!!");
    }
    //if (loc == 7) {
    //  //Rcout << "~~~~~~~~~\t" << knotsS(loc) << "\t" << knotsS(loc+1) << "\t" << knotsS(loc+2) << "\t" << knotsX(loc-1) << "\t" <<knotsX(loc) <<"\t"<<knotsX(loc+1) << "\n";
    //}
    knotsB(loc+1) = knotsY(loc+1) - knotsS(loc+1) * knotsX(loc+1);
    knotsI(loc+1) = (knotsY(loc+1) - knotsY(loc+2) - knotsS(loc+1) * knotsX(loc+1) + knotsS(loc+2) * knotsX(loc+2)) / (knotsS(loc+2) - knotsS(loc+1));
    knotsI(loc) = (knotsY(loc) - knotsY(loc+1) - knotsS(loc) * knotsX(loc) + knotsS(loc+1) * knotsX(loc+1)) / (knotsS(loc+1) - knotsS(loc));
    if (knotsS(loc+2) - knotsS(loc+1) == 0){
      //Rcout << loc <<"\t"<<sample(0)<<"\t"<<sample(1)<<"\t"<<sample(2)<<"\n";
      for (int i=0; i<nKnots; i++) {
        //Rcout<<knotsX(i) <<"\t";
      }
      //Rcout <<"knotsX\n";
      for (int i=0; i<nKnots; i++) {
        //Rcout<<knotsS(i) <<"\t";
      }
      //Rcout <<"knotsS\n";
      for (int i=0; i<nKnots; i++) {
        //Rcout<<knotsI(i) <<"\t";
      }
      //Rcout <<"knotsI\n";
      //Rcout <<"Please Check for Bugs!";
      for (int i=0; i<1; i++) {
        //Rcout << "ATTENTION!!\n";
      }
      if (sample(0) == 0 && sample(1) == 0 && sample(2) == 0) {
        throw Rcpp::exception("Something is wrong!\tI 16");
      }
      return true;
      throw Rcpp::exception("Something is wrong!\tI 16");
    }
    if (knotsS(loc+1) - knotsS(loc) == 0){
      //Rcout << loc << "\t" <<sample(0)<<"\t"<<sample(1)<<"\t" << sample(2) <<"\n";
      for (int i=0; i<nKnots; i++) {
        //Rcout<<knotsX(i) <<"\t";
      }
      //Rcout <<"knotsX\n";
      for (int i=0; i<nKnots; i++) {
        //Rcout<<knotsS(i) <<"\t";
      }
      //Rcout <<"knotsS\n";
      for (int i=0; i<nKnots; i++) {
        //Rcout<<knotsI(i) <<"\t";
      }
      //Rcout <<"knotsI\n";
      //Rcout <<"Please Check for Bugs!";
      for (int i=0; i<1; i++) {
        //Rcout << "ATTENTION!!\n";
      }
      if (sample(0) == 0 && sample(1) == 0 && sample(2) == 0) {
        throw Rcpp::exception("Something is wrong!\tJ 17");
      }
      return true;
      throw Rcpp::exception("Something is wrong!\tJ 17");
    }
    if (loc == nKnots-2){
      //auc(loc+2) = -exp(knotsB(loc+2) + knotsS(loc+2) * knotsI(loc+1)) / knotsS(loc+2);
      auc(loc+2) = knotsB(loc+2) + knotsS(loc+2) * knotsI(loc+1) - log(-knotsS(loc+2)); //knotsS(loc+2) must be negetive
    }
    else{
      if (knotsS(loc+2) == 0){
        //auc(loc+2) = exp(knotsB(loc+2)) * (knotsI(loc+2) - knotsI(loc+1));
        auc(loc+2) = knotsB(loc+2) + log(knotsI(loc+2) - knotsI(loc+1));
      }
      else {
        //auc(loc+2) = (exp(knotsB(loc+2) + knotsS(loc+2) * knotsI(loc+2)) - exp(knotsB(loc+2) + knotsS(loc+2) * knotsI(loc+1))) / knotsS(loc+2);
        //auc(loc+2) = (exp(knotsB(loc+2) + knotsS(loc+2) * knotsI(loc+2)) * (1.0 - exp(knotsS(loc+2) * (knotsI(loc+1) - knotsI(loc+2))))) / knotsS(loc+2);
        if (knotsS(loc+2) > 0){
          auc(loc+2) = knotsB(loc+2) + knotsS(loc+2) * knotsI(loc+2) + log(1.0 - exp(knotsS(loc+2) * (knotsI(loc+1) - knotsI(loc+2)))) - log(knotsS(loc+2));
        }
        else {
          auc(loc+2) = knotsB(loc+2) + knotsS(loc+2) * knotsI(loc+1) + log(1.0 - exp(knotsS(loc+2) * (knotsI(loc+2) - knotsI(loc+1)))) - log(-knotsS(loc+2));
        }
      }
    }
    if (knotsS(loc+1) == 0){
      //auc(loc+1) = exp(knotsB(loc+1)) * (knotsI(loc+1) - knotsI(loc));
      auc(loc+1) = knotsB(loc+1) + log(knotsI(loc+1) - knotsI(loc));
    }
    else {
      //auc(loc+1) = (exp(knotsB(loc+1) + knotsS(loc+1) * knotsI(loc+1)) - exp(knotsB(loc+1) + knotsS(loc+1) * knotsI(loc))) / knotsS(loc+1);
      //auc(loc+1) = (exp(knotsB(loc+1) + knotsS(loc+1) * knotsI(loc+1)) * (1.0 - exp(knotsS(loc+1) * (knotsI(loc) - knotsI(loc+1))))) / knotsS(loc+1);
      if (knotsS(loc+1) > 0){
        auc(loc+1) = knotsB(loc+1) + knotsS(loc+1) * knotsI(loc+1) + log(1.0 - exp(knotsS(loc+1) * (knotsI(loc) - knotsI(loc+1)))) - log(knotsS(loc+1));
      }
      else {
        auc(loc+1) = knotsB(loc+1) + knotsS(loc+1) * knotsI(loc) + log(1.0 - exp(knotsS(loc+1) * (knotsI(loc+1) - knotsI(loc)))) - log(-knotsS(loc+1));
      }
    }
    if (knotsS(loc) == 0){
      //auc(loc) = exp(knotsB(loc)) * (knotsI(loc) - knotsI(loc-1));
      auc(loc) = knotsB(loc) + log(knotsI(loc) - knotsI(loc-1));
    }
    else {
      //auc(loc) = (exp(knotsB(loc) + knotsS(loc) * knotsI(loc)) - exp(knotsB(loc) + knotsS(loc) * knotsI(loc-1))) / knotsS(loc);
      //auc(loc) = (exp(knotsB(loc) + knotsS(loc) * knotsI(loc)) * (1.0 - exp(knotsS(loc) * (knotsI(loc-1) - knotsI(loc))))) / knotsS(loc);
      if (knotsS(loc) > 0){
        auc(loc) = knotsB(loc) + knotsS(loc) * knotsI(loc) + log(1.0 - exp(knotsS(loc) * (knotsI(loc-1) - knotsI(loc)))) - log(knotsS(loc));
      }
      else {
        auc(loc) = knotsB(loc) + knotsS(loc) * knotsI(loc-1) + log(1.0 - exp(knotsS(loc) * (knotsI(loc) - knotsI(loc-1)))) - log(-knotsS(loc));
      }
    }

  }
  for (int i=1; i<nKnots; i++) {
    if (knotsI(i) < knotsI(i-1)) {
      //Rcout << i <<"\t"<<knotsI(i-1) <<"\t"<<knotsI(i) << "\t" <<knotsI(i-1) - knotsI(i) << "\n";
      //Rcout << "The intercept are messed up!\n";
      return true;
      throw Rcpp::exception("The intercept are messed up!");
    }
  }
  return false;
}


void initializeAdRejSample(int nKnots, NumericVector knotsX, NumericVector knotsY, NumericVector knotsS, NumericVector knotsB, NumericVector knotsI, NumericVector auc){
  //auc(0) = exp(knotsB(0) + knotsS(0) * knotsI(0)) / knotsS(0);

  auc(0) = knotsB(0) + knotsS(0) * knotsI(0) - log(knotsS(0));
  if (knotsS(nKnots-1) == 0){
    throw Rcpp::exception("Something is wrong! 2");
  }
  //auc(nKnots-1) = -exp(knotsB(nKnots-1) + knotsS(nKnots-1) * knotsI(nKnots-2)) / knotsS(nKnots-1);
  auc(nKnots-1) = knotsB(nKnots-1) + knotsS(nKnots-1) * knotsI(nKnots-2) - log(-knotsS(nKnots-1));
  //norm = auc(0) + auc(nKnots-1);
  for (int i=1; i<nKnots-1; i++){
    if (knotsS(i) == 0){
      //auc(i) = exp(knotsB(i)) * (knotsI(i) - knotsI(i-1));
      auc(i) = knotsB(i) + log(knotsI(i) - knotsI(i-1));
    }
    else{
      //auc(i) = (exp(knotsB(i) + knotsS(i) * knotsI(i)) - exp(knotsB(i) + knotsS(i) * knotsI(i-1))) / knotsS(i);
      //auc(i) = (exp(knotsB(i) + knotsS(i) * knotsI(i)) * (1.0 - exp(knotsS(i) * (knotsI(i-1) - knotsI(i))))) / knotsS(i);
      if (knotsS(i) > 0){
        auc(i) = knotsB(i) + knotsS(i) * knotsI(i) + log(1.0 - exp(knotsS(i) * (knotsI(i-1) - knotsI(i)))) - log(knotsS(i));
      }
      else {
        auc(i) = knotsB(i) + knotsS(i) * knotsI(i-1) + log(1.0 - exp(knotsS(i) * (knotsI(i) - knotsI(i-1)))) - log(-knotsS(i));
      }
      // //Rcout<<"initial area "<<i<<"\t"<<auc(i)<<"\t"<<knotsB(i)<<"\t"<<knotsS(i)<<"\t"<<knotsI(i)<<"\n";
    }
    //norm += auc(i);
  }

}


NumericVector adrejSamplerq(double sigma, arma::colvec resid, IntegerVector ind, double c, double d, double p, double lambda){
  //adaptive rejective sampling for q, return the sample and number of rejections
  int maxNumKnots = 100;
  NumericVector knotsX(maxNumKnots);  // x
  NumericVector knotsY(maxNumKnots);  // y
  NumericVector knotsS(maxNumKnots);  // slope
  NumericVector knotsB(maxNumKnots);  // intecept b in y = sx+b
  NumericVector knotsI(maxNumKnots);  // endPoints of intervals of piecewise linear function
  NumericVector knotsXBU(maxNumKnots);  // x
  NumericVector knotsYBU(maxNumKnots);  // y
  NumericVector knotsSBU(maxNumKnots);  // slope
  NumericVector knotsBBU(maxNumKnots);  // intecept b in y = sx+b
  NumericVector knotsIBU(maxNumKnots);  // endPoints of intervals of piecewise linear function
  NumericVector auc(maxNumKnots);     // area under each piece of the exponential curve
  NumericVector prob(maxNumKnots);    // probability of each exponential part
  NumericVector temp;
  int nKnots = 10; //current number of knots
  NumericVector result(2);
  double left = -5.0;
  while (LLqDer(sigma, left, resid, ind, c, d, p, lambda) < 0){
    left = -pow(left, 2.0);
  }
  double right = 5.0;
  while (LLqDer(sigma, right, resid, ind, c, d, p, lambda) > 0){
    right = pow(right, 2.0);
  }
  ////Rcout<<left<<"\t"<<right<<"\n";
  double h = (right - left) / (nKnots - 1.0);
  knotsX(0) = left;
  knotsX(nKnots-1) = right;
  for (int i=1; i<nKnots-1; i++){
    knotsX(i) = knotsX(i-1) + h;
  }
  for (int i=0; i<nKnots; i++){
    knotsY(i) = LLq(sigma, knotsX(i), resid, ind, c, d, p, lambda);
    knotsS(i) = LLqDer(sigma, knotsX(i), resid, ind, c, d, p, lambda);
  }
  //calculate the intercept of consecutive lines
  temp = intercept(knotsX, knotsY, knotsS, nKnots);
  // //Rcout<<"intervals endpts ";
  for (int i=0; i<nKnots-1; i++){
    knotsI(i) = temp(i);
    // //Rcout<<knotsI(i)<<"\t";
  }
  // //Rcout<<"\n";
  //calculate the intercept parameter of each line equations
  for (int i=0; i<nKnots; i++){
    knotsB(i) = knotsY(i) - knotsS(i) * knotsX(i);
  }
  //double norm;  //normalizer for the cover distribution
  if (knotsS(0) == 0){
    for (int i=0; i<nKnots; i++) {
      //Rcout<<knotsS(i) <<"\t";
    }
    //Rcout<<"knotsS\n";
    for (int i=0; i<nKnots; i++) {
      //Rcout<<knotsX(i) <<"\t";
    }
    //Rcout<<"knotsX\n";
    //Rcout<<sigma<<"\n";
    throw Rcpp::exception("Something is wrong! 1");
  }

  initializeAdRejSample(nKnots, knotsX, knotsY, knotsS, knotsB, knotsI, auc);
  prob = normalize(auc, nKnots);
  //for (int i=0; i<nKnots; i++){
  // //Rcout<<prob(i)<<"\t";
  //}

  // //Rcout <<"IIIIPPPP\n";
  //for (int i=0; i<nKnots; i++){
  //  //Rcout<<auc(i)<<"\t";
  //}
  // //Rcout <<"IIIIAAAA\n";

  NumericVector sample = arsSample(false, knotsI, knotsS, knotsB, prob, nKnots);
  double u = R::runif(0.0, 1.0);
  double yD = LLq(sigma, sample(0), resid, ind, c, d, p, lambda);
  //int count = 0;
  NumericVector null;
  arma::mat empty;
  int countFail = 0;
  while (u > exp(yD - sample(1)) && nKnots < maxNumKnots && countFail < 5){

    //add the rejected sample into the knots if the length of the interval is not too small.

    if (sample(0)==0||sample(0) == nKnots-1||(sample(0) > 0 && sample(0) < nKnots-1 && knotsI(sample(0)) - knotsI(sample(0) - 1) > thresholdInterval)) {

    }
    bool hasProblem = calProbAndUpdateKnots(1, 0.0, sigma, resid, ind, c, d, p, 0.0, lambda, 0, null, null, null, empty, u, yD, nKnots, knotsX, knotsY, knotsS, knotsB, knotsI, auc, sample);
    if (hasProblem) {
      //Rcout<<"q!";
      return 1.0 / 0.0;
    }
    if (hasProblem) {
      knotsX = knotsXBU;
      knotsY = knotsYBU;
      knotsI = knotsIBU;
      knotsS = knotsSBU;
      knotsB = knotsBBU;
      countFail++;
      //Rcout << "try # " << countFail << "\n";
      break;
    }
    else {
      nKnots++;
      knotsXBU = knotsX;
      knotsYBU = knotsY;
      knotsSBU = knotsS;
      knotsIBU = knotsI;
      knotsBBU = knotsB;
      prob = normalize(auc, nKnots);
    }
    bool isDup = true;
    int count = 0;
    while (isDup && count < 5) {
      isDup = false;
      sample = arsSample(false, knotsI, knotsS, knotsB, prob, nKnots);
      for (int i=0; i<nKnots; i++) {
        if (sample(0) == knotsX(i)) {
          isDup = true;
          //Rcout<<"WTFWTFWTFWTFWTFWTF\n";
          //throw Rcpp::exception("WTFWTFWTFWTFWTFWTF");
          break;
        }
      }
      count ++;
    }
    if (count == 100) {
      break;
    }
    //sample = arsSample(false, knotsI, knotsS, knotsB, prob, nKnots);
    u = R::runif(0.0, 1.0);
    yD = LLq(sigma, sample(0), resid, ind, c, d, p, lambda);
  }
  if (nKnots == maxNumKnots && u > exp(yD - sample(1))){
    throw Rcpp::exception("Exceed the maximum number of knots");
  }
  result(0) = sample(0);
  result(1) = nKnots - 10.0;
  return result;
}

NumericVector adrejSamplerSigma(double q, arma::colvec resid, IntegerVector ind, double c, double d, double p, double lambda){
  //adaptive rejective sampling for sigma, return the sample and the number of rejection
  int maxNumKnots = 100;
  NumericVector knotsX(maxNumKnots);  // x
  NumericVector knotsY(maxNumKnots);  // y
  NumericVector knotsS(maxNumKnots);  // slope
  NumericVector knotsB(maxNumKnots);  // intecept b in y = sx+b
  NumericVector knotsI(maxNumKnots);  // endPoints of intervals of piecewise linear function
  NumericVector knotsXBU(maxNumKnots);  // x
  NumericVector knotsYBU(maxNumKnots);  // y
  NumericVector knotsSBU(maxNumKnots);  // slope
  NumericVector knotsBBU(maxNumKnots);  // intecept b in y = sx+b
  NumericVector knotsIBU(maxNumKnots);  // endPoints of intervals of piecewise linear function
  NumericVector auc(maxNumKnots);     // area under each piece of the exponential curve
  NumericVector prob(maxNumKnots);    // probability of each exponential part
  NumericVector temp;
  NumericVector result(2);
  int nKnots = 10; //current number of knots
  double left = 0.1;
  while (LLsigmaInvDer(left, q, resid, ind, c, d, p, lambda) < 0){
    left = pow(left, 2.0);
  }
  double right = 5.0;
  while (LLsigmaInvDer(right, q, resid, ind, c, d, p, lambda) > 0){
    right = pow(right, 2.0);
  }
  ////Rcout<<left<<"\t"<<right<<"\n";
  double h = (right - left) / (nKnots - 1.0);
  knotsX(0) = left;
  knotsX(nKnots-1) = right;
  for (int i=1; i<nKnots-1; i++){
    knotsX(i) = knotsX(i-1) + h;
  }
  for (int i=0; i<nKnots; i++){
    knotsY(i) = LLsigmaInv(knotsX(i),q, resid, ind, c, d, p, lambda);
    knotsS(i) = LLsigmaInvDer(knotsX(i), q, resid, ind, c, d, p, lambda);
  }
  //calculate the intercept of consecutive lines
  temp = intercept(knotsX, knotsY, knotsS, nKnots);
  for (int i=0; i<nKnots-1; i++){
    knotsI(i) = temp(i);
  }
  //calculate the intercept parameter of each line equations
  for (int i=0; i<nKnots; i++){
    knotsB(i) = knotsY(i) - knotsS(i) * knotsX(i);
  }
  if (knotsS(0) == 0){
    throw Rcpp::exception("Something is wrong! 19");
  }

  initializeAdRejSample(nKnots, knotsX, knotsY, knotsS, knotsB, knotsI, auc);
  prob = normalize(auc, nKnots);

  ////Rcout<<"%%%%%%%%/t"<<nKnots<<"\n";
  NumericVector sample = arsSample(true, knotsI, knotsS, knotsB, prob, nKnots);
  double u = R::runif(0.0, 1.0);
  double yD = LLsigmaInv(sample(0),q, resid, ind, c, d, p, lambda);
  NumericVector null;
  arma::mat empty;
  //int count = 0;
  int countFail = 0;
  if (sample(0) < 0) {
    for (int i=0; i<nKnots; i++) {
      //Rcout<<knotsX(i) <<"\t";
    }
    //Rcout << "knotsX\n";
    for (int i=0; i<nKnots; i++) {
      //Rcout<<knotsS(i) <<"\t";
    }
    //Rcout << "knotsS\n";
    for (int i=0; i<nKnots-1; i++) {
      //Rcout<<knotsI(i) <<"\t";
    }
    //Rcout << "knotsI\n";
    for (int i=0; i<nKnots; i++) {
      //Rcout<<prob(i) <<"\t";
    }
    //Rcout << "prob\n";
    throw Rcpp::exception("Got negative sigma!!");
  }
  while (u > exp(yD - sample(1)) && nKnots < maxNumKnots && countFail < 5){
    ////Rcout<<sample(0)<<"\t"<<sample(1)<<"\t"<<sample(2)<<"~~~~~~~~~~~"<<"\n";
    //count++;
    //add the rejected sample into the knots

    bool hasProblem = calProbAndUpdateKnots(2, 0.0, 0.0, resid, ind, c, d, p, q, lambda, 1, null, null, null, empty, u, yD, nKnots, knotsX, knotsY, knotsS, knotsB, knotsI, auc, sample);
    if (hasProblem) {
      //Rcout<<"sigma!";
      return 1.0 / 0.0;
    }
    if (hasProblem) {
      knotsX = knotsXBU;
      knotsY = knotsYBU;
      knotsI = knotsIBU;
      knotsS = knotsSBU;
      knotsB = knotsBBU;
      countFail++;
      //Rcout << "try # " << countFail << "\n";
      break;
    }
    else {
      nKnots++;
      knotsXBU = knotsX;
      knotsYBU = knotsY;
      knotsSBU = knotsS;
      knotsIBU = knotsI;
      knotsBBU = knotsB;
      prob = normalize(auc, nKnots);
    }
    bool isDup = true;
    int count = 0;
    while (isDup && count < 5) {
      isDup = false;
      sample = arsSample(true, knotsI, knotsS, knotsB, prob, nKnots);
      for (int i=0; i<nKnots; i++) {
        if (sample(0) == knotsX(i)) {
          isDup = true;
          //Rcout<<"WTFWTFWTFWTFWTFWTF\n";
          //Rcout<<sample(0)<<"\n";
          //throw Rcpp::exception("WTFWTFWTFWTFWTFWTF");
          break;
        }
      }
      count ++;
    }

    if (count == 100) {
      break;
    }


    //sample = arsSample(true, knotsI, knotsS, knotsB, prob, nKnots);
    u = R::runif(0.0, 1.0);
    yD = LLsigmaInv(sample(0), q, resid, ind, c, d, p, lambda);
  }
  if (nKnots == maxNumKnots && u > exp(yD - sample(1))){
    throw Rcpp::exception("Exceed the maximum number of knots");
  }
  if (sample(0) <= 0) {
    for (int i=0; i<nKnots; i++) {
      //Rcout<<knotsX(i) <<"\t";
    }
    //Rcout << "knotsX\n";
    for (int i=0; i<nKnots-1; i++) {
      //Rcout<<knotsI(i) <<"\t";
    }
    //Rcout << "knotsI\n";
    throw Rcpp::exception("Got non-positive sigma!!");
  }
  result(0) = 1 / sample(0);
  result(1) = nKnots - 10.0;
  return result;
}


NumericVector adrejSamplerBeta(int k, double var, NumericVector q, NumericVector U, double p, NumericVector sigma, arma::mat X){
  //adaptive rejective sampling for beta, return the sample and the number of rejections
  int maxNumKnots = 100;
  NumericVector knotsX(maxNumKnots);  // x
  NumericVector knotsY(maxNumKnots);  // y
  NumericVector knotsS(maxNumKnots);  // slope
  NumericVector knotsB(maxNumKnots);  // intecept b in y = sx+b
  NumericVector knotsI(maxNumKnots);  // endPoints of intervals of piecewise linear function
  NumericVector knotsXBU(maxNumKnots);  // x
  NumericVector knotsYBU(maxNumKnots);  // y
  NumericVector knotsSBU(maxNumKnots);  // slope
  NumericVector knotsBBU(maxNumKnots);  // intecept b in y = sx+b
  NumericVector knotsIBU(maxNumKnots);  // endPoints of intervals of piecewise linear function
  NumericVector auc(maxNumKnots);     // area under each piece of the exponential curve
  NumericVector prob(maxNumKnots);    // probability of each exponential part
  NumericVector temp;
  NumericVector result(2);
  int nKnots = 10; //current number of knots
  double left = -5.0;
  while (LLbetaDer(left, var, k, q, U, p, sigma, X) < 0){
    left = -pow(left, 2.0);
  }
  double right = 5.0;
  while (LLbetaDer(right, var, k, q, U, p, sigma, X) > 0){
    right = pow(right, 2.0);
  }
  ////Rcout<<left<<"\t"<<right<<"\n";
  double h = (right - left) / (nKnots - 1.0);
  knotsX(0) = left;
  knotsX(nKnots-1) = right;
  for (int i=1; i<nKnots-1; i++){
    knotsX(i) = knotsX(i-1) + h;
  }
  for (int i=0; i<nKnots; i++){
    knotsY(i) = LLbeta(knotsX(i), var, k, q, U, p, sigma, X);
    knotsS(i) = LLbetaDer(knotsX(i), var, k, q, U, p, sigma, X);
  }
  //calculate the intercept of consecutive lines
  temp = intercept(knotsX, knotsY, knotsS, nKnots);
  for (int i=0; i<nKnots-1; i++){
    knotsI(i) = temp(i);
  }
  //calculate the intercept parameter of each line equations
  for (int i=0; i<nKnots; i++){
    knotsB(i) = knotsY(i) - knotsS(i) * knotsX(i);
  }
  if (knotsS(0) == 0){
    throw Rcpp::exception("Something is wrong! 38");
  }


  initializeAdRejSample(nKnots, knotsX, knotsY, knotsS, knotsB, knotsI, auc);
  prob = normalize(auc, nKnots);


  NumericVector sample = arsSample(false, knotsI, knotsS, knotsB, prob, nKnots);
  double u = R::runif(0.0, 1.0);
  double yD = LLbeta(sample(0), var, k, q, U, p, sigma, X);
  arma::colvec null;
  IntegerVector empty;
  int countFail = 0;
  while (u > exp(yD - sample(1)) && nKnots < maxNumKnots && countFail<5){
    ////Rcout<<sample(0)<<"\t"<<sample(1)<<"\t"<<sample(2)<<"\t"<<count<<"\n";
    //count++;
    //add the rejected sample into the knots

    bool hasProblem = calProbAndUpdateKnots(3, var, 0.0, null, empty, 0.0, 0.0, p, 0.0, 0.0, k, q, U, sigma, X, u, yD, nKnots, knotsX, knotsY, knotsS, knotsB, knotsI, auc, sample);
    if (hasProblem) {
      ////Rcout<<"beta!";
      if (std::isinf(1.0 / 0.0)){
        //Rcout<<"RIGHT!\n";
      }
      //throw Rcpp::exception("WRONG!");
      return 1.0 / 0.0;
    }


    if (hasProblem) {
      knotsX = knotsXBU;
      knotsY = knotsYBU;
      knotsI = knotsIBU;
      knotsS = knotsSBU;
      knotsB = knotsBBU;
      countFail++;
      //Rcout << "try # " << countFail << "\n";
      break;
    }
    else {
      nKnots++;
      knotsXBU = knotsX;
      knotsYBU = knotsY;
      knotsSBU = knotsS;
      knotsIBU = knotsI;
      knotsBBU = knotsB;
      prob = normalize(auc, nKnots);
    }
    bool isDup = true;
    int count = 0;
    while (isDup && count < 5) {
      isDup = false;
      sample = arsSample(false, knotsI, knotsS, knotsB, prob, nKnots);
      for (int i=0; i<nKnots; i++) {
        if (sample(0) == knotsX(i)) {
          isDup = true;
          //Rcout<<"WTFWTFWTFWTFWTFWTF\n";
          //throw Rcpp::exception("WTFWTFWTFWTFWTFWTF");
          break;
        }
      }
      count ++;
    }

    if (count == 100) {
      break;
    }

    u = R::runif(0.0, 1.0);
    yD = LLbeta(sample(0), var, k, q, U, p, sigma, X);
    ////Rcout<<sample(0)<<"\t"<<sample(1)<<"\t"<<sample(2)<<"\t"<<exp(yD - sample(1)) << "\t"<<u<<"\tSample\n";

  }
  if (nKnots == maxNumKnots && u > exp(yD - sample(1))){
    throw Rcpp::exception("Exceed the maximum number of knots");
  }
  result(0) = sample(0);
  result(1) = nKnots - 10.0;
  return result;
}

int findDup(NumericVector sigma, NumericVector q, double vsigma, double vq, int n){
  //if value is in the vector values (of length n not necessary equal to value.size()), return the index, otherwise return -1
  for (int i=0; i<n; i++){
    if (vsigma == sigma(i) && vq == q(i)){
      return i;
    }
  }
  return -1;
}

IntegerVector findCluster(IntegerVector s, int ind){
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

//' BQR Based on DPM of Logistic Distribution
//'
//' Bayesian quantile regression with Dirichlet process mixture of Logistic Distribution.
//' It is the main and recommended method of this package.
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
//' @return List of model parameters including the coefficients.
//' @export
//'
//[[Rcpp::export]]
List bqrDPML (NumericMatrix Xr, NumericVector Yr, double p, int nsim, NumericVector initCoef, bool sampleQ = true, int burn = 5000, bool plotDensity = false, int naux=10,  double threshold=0.0000001) {
  //initialization
  // nsim is the number of simulation
  // naux is the number of the auxiliary variable
  // p is the quantile regression
  // X is the design matrix with each row correspnding to an observation (the first column with all ones should be removed)
  // Y is the response vector
  // initCoef is the initial value for the regression coefficients
  // sampleQ is a flag which determines how to calculate the location shift. (true means using sample quantile, false means using the density estimation)
  // threshold is the threshold of the error of the solution to the equation of the lambda
  if (p<0 || p>1){
    throw Rcpp::exception("the percentile should be between 0 and 1!");
  }

  double lambda = findRootNewton(p, threshold);
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
  NumericVector sdDiag = rep(10000.0, ncov);  //diagnal elements of the covariance matrix (diagonal) of the prior (multivariate normal)
  //of the regression coeffcients
  //NumericVector sdDiag = rep(0.0001, ncov);
  //sdDiag(0) = 10000.0;
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
  int n = 0;   //number of clusters of the DP
  IntegerVector s(size);   //configuation parameters for the DP
  NumericVector sigmaStar(size*2);   //unique values for the DP
  NumericVector qStar(size*2);
  IntegerVector clusterSize(size*2);   //cluster size for the DP
  NumericVector sigmaAux(naux);
  NumericVector qAux(naux);

  //initialize parameters to be updated
  arma::colvec beta = initCoef;  //regression coefficient
  double d = 0.0;  //scale parameter for the base measure (inverse gamma) of the DP
  NumericVector resid(size);   //residual
  resid = Y - X * beta;   //calculate the residual
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
  NumericVector sigma(size); //scale parameter of the logistic distribution
  NumericVector q(size); //quantile parameter of the logistc distribution

  for (i=0; i<size; i++){
    sigma(i) = 1.0 / ::Rf_rgamma(c, 1.0 / d);
    double u = R::runif(0.0, 1.0);
    q(i) = -sigma(i) * log(lambda) + sigma(i) * log(u * (1.0 - p) / p / (1.0 - u));
  }

  double ita = R::runif(0.0, 1.0);   //the augment parameter to help update the precision parameters

  //matrix to restore the mcmc samples
  NumericMatrix betaRec(nsim, ncov);
  NumericMatrix sigmaRec(nsim, size);
  NumericMatrix qRec(nsim, size);
  NumericVector dRec(nsim);
  NumericVector nClusterRec(nsim);
  NumericVector nClusterTRec(nsim);
  NumericVector alphaRec(nsim);
  NumericVector interceptAdj(nsim);
  NumericMatrix clusterRec(nsim, size);
  NumericMatrix nRejRec(nsim, 2 + ncov); // keep track average number of rejections for the adaptive rejection sampling


  NumericVector xGrid(4001);
  xGrid(0) = -20.0;
  for (int i=1; i<4001; i++) {
    xGrid(i) = xGrid(i-1) + 0.01;
  }
  NumericVector sumdense(4001, 0.0);
  //ofstream betaRecFile;
  //std::string filename = path + "betaLogisticRecFile.txt";
  //betaRecFile.open (filename.c_str());
  //ofstream dRecFile;
  //ofstream


  //Gibbs sampler
  NumericVector prob(size*2);   //probabilities used to draw from the mixture
  NumericVector probAux(naux);  //used to draw from the mixture
  double temp;
  //double temp1;
  IntegerVector tempVec(size);
  double aveN = 0.0;

  //NumericVector tempVecD(size+1);
  IntegerVector tempVec1(size);

  NumericVector result(2); // temporarily store the output of the adaptive rejection sample and number of rejections

  //calculate the initial number of clusters and configuation parameters
  n = 0;
  sigmaStar(0) = sigma(0);
  qStar(0) = q(0);
  s(0) = 0;
  clusterSize(0) = 1;
  n = 1;
  for (i=1; i<size; i++){
    s(i) = findDup(sigmaStar, qStar, sigma(i), q(i), n);
    if (s(i) == -1){
      sigmaStar(n) = sigma(i);
      qStar(n) = q(i);
      s(i) = n;
      clusterSize(n) = 1;
      n++;
    }
    else{
      clusterSize(s(i))++;
    }
  }

  //Rcout << "initializing ******************\n";



  for (sim=0; sim<nsim; sim++){
    //Rcout << "*************************************************\n";
    //Rcout<<sim<<"\n";
    //Rcout<<"n\t"<<n<<"\n";
    //if (sim%1000 == 0) {
    //   Rcout<<sim<<"\n";
    //}

    //Rcout << "updating unique sigma and q...\n";
    nRejRec(sim, 0) = 0.0;
    nRejRec(sim, 1) = 0.0;
    for (i=0; i<n; i++){
      IntegerVector cluster = findCluster(s, i);
      ////Rcout<<i<<"\n";
      result = adrejSamplerSigma(qStar(i), resid, cluster, c, d, p, lambda);
      sigmaStar(i) = result(0);
      nRejRec(sim, 1) = nRejRec(sim, 1) + result(1);
      if (sigmaStar(i) == 0) {
        throw Rcpp::exception("NEGATIVE SIGMA!!!");
      }


      result = adrejSamplerq(sigmaStar(i), resid, cluster, c, d, p, lambda);
      qStar(i) = result(0);
      nRejRec(sim, 0) = nRejRec(sim, 0) + result(1);
      ////Rcout << qStar(i) << "\n";
    }
    nRejRec(sim, 0) = nRejRec(sim, 0) / n;
    nRejRec(sim, 1) = nRejRec(sim, 1) / n;

    //Rcout << "updating the configuation...\n";
    for (i=0; i<size; i++){
      ////Rcout<<"sample "<<i<<"\n";
      if (clusterSize(s(i)) == 1){
        sigmaAux(0) = sigma(i);
        qAux(0) = q(i);
      }
      else {
        sigmaAux(0) = 1.0 / ::Rf_rgamma(c, 1.0 / d);
        double u = R::runif(0.0, 1.0);
        qAux(0) = -sigmaAux(0) * log(lambda) + sigmaAux(0) * log(u * (1.0 - p) / p / (1.0 - u));
      }

      for (j=1; j<naux; j++){
        sigmaAux(j) = 1.0 / ::Rf_rgamma(c, 1.0 / d);
        double u = R::runif(0.0, 1.0);
        qAux(j) = -sigmaAux(j) * log(lambda) + sigmaAux(j) * log(u * (1.0 - p) / p / (1.0 - u));
      }
      temp = 0.0;
      for (j=0; j<n; j++){
        if (s(i) == j){
          prob(j) = (clusterSize(j) - 1.0) * dLogistic(resid(i), p, -qStar(j), sigmaStar(j));
        }
        else {
          prob(j) = clusterSize(j) * dLogistic(resid(i), p, -qStar(j), sigmaStar(j));
        }
        temp += prob(j);
      }
      for (j=0; j<naux; j++){
        probAux(j) = alpha * dLogistic(resid(i), p, -qAux(j), sigmaAux(j)) / naux;
        temp += probAux(j);
      }
      if (temp == 0){
        throw Rcpp::exception("Something is wrong! 57");
      }
      double probOld = 0.0;
      for (j=0; j<n; j++){
        prob(j) /= temp;
        probOld += prob(j);
      }
      ////Rcout<<"old Prob = "<<probOld<<"\n" ;
      for (j=0; j<naux; j++){
        probAux(j) /= temp;
      }

      temp = R::runif(0.0, 1.0);
      if (temp <= prob(0)){
        sigma(i) = sigmaStar(0);
        q(i) = qStar(0);
        clusterSize(s(i))--;
        clusterSize(0)++;
      }
      else if (temp + probAux(naux-1) > 1){
        sigma(i) = sigmaAux(naux-1);
        q(i) = qAux(naux-1);
        int ind = findDup(sigmaStar, qStar, sigma(i), q(i), n);
        if (ind == -1) {
          sigmaStar(n) = sigma(i);
          qStar(n) = q(i);
          clusterSize(n) = 1;
          n++;
          clusterSize(s(i))--;
        }
        else {
          clusterSize(ind)++;
          clusterSize(s(i))--;
        }
      }
      else{
        double cumProb = prob(0);
        bool selected = false;
        for (j=1; j<n; j++){
          if (temp > cumProb && temp <= cumProb + prob(j)){
            sigma(i) = sigmaStar(j);
            q(i) = qStar(j);
            clusterSize(s(i))--;
            clusterSize(j)++;
            selected = true;
            break;
          }
          cumProb += prob(j);
        }
        if (!selected) {
          for(j=0; j<naux-1; j++){
            if (temp > cumProb && temp <= cumProb + probAux(j)){
              sigma(i) = sigmaAux(j);
              q(i) = qAux(j);
              int ind = findDup(sigmaStar, qStar, sigma(i), q(i), n);
              if (ind == -1) {
                sigmaStar(n) = sigma(i);
                qStar(n) = q(i);
                clusterSize(n) = 1;
                n++;
                clusterSize(s(i))--;
              }
              else {
                clusterSize(ind)++;
                clusterSize(s(i))--;
              }
              break;
            }
            cumProb += probAux(j);
          }
        }
      }
    }

    //Rcout<<"calculating the number of clusters and configuation parameters...\n";
    //calculate number of clusters and configuation parameters after the above update
    n = 0;
    sigmaStar(0) = sigma(0);
    qStar(0) = q(0);
    s(0) = 0;
    clusterSize(0) = 1;
    n = 1;
    for (i=1; i<size; i++){
      s(i) = findDup(sigmaStar, qStar, sigma(i), q(i), n);
      if (s(i) == -1){
        sigmaStar(n) = sigma(i);
        qStar(n) = q(i);
        s(i) = n;
        clusterSize(n) = 1;
        n++;
      }
      else{
        clusterSize(s(i))++;
      }
    }

    aveN += n;


    //Rcout << "updating the precision parameters... \n";
    //Gibbs sampling for the precision parameter of the first DP
    temp = (a1 + n -1) / (a1 + n - 1 + size*(b1 - log(ita)));
    if (R::runif(0.0, 1.0) <= temp) {
      alpha = ::Rf_rgamma(a1 + n, 1.0 / (b1 - log(ita)));
    }
    else {
      alpha = ::Rf_rgamma(a1 + n - 1.0, 1.0 / (b1 - log(ita)));
    }
    ita = ::Rf_rbeta(alpha+1.0, size);

    //Rcout << "updating the scale parameters in the base measure... \n";
    //Gibbs sampling for the scale parameter in the base measure of the DP
    temp = 0.0;
    for (j=0; j<n; j++){
      temp += 1.0 / sigmaStar(j);
      // //Rcout<<sigmaStar(j) << "\t";
    }
    //Rcout<<"\n";
    d = ::Rf_rgamma(a2 + n * c, 1.0 / (b2 + temp));
    // //Rcout<<d<<"\t" << a2 + n * c << "\t" << 1.0 / (b2 + temp) <<"\t"<<b2<<"\n";

    //Rcout << "updating regression coefficients...\n";
    //Gibbs sampling for the regression coefficients
    NumericVector U(size);
    for (int k=0; k<ncov; k++){
      //Rcout<<"regression coefficients # " <<(k+1)<<"\n";
      for (i=0; i<size; i++){
        U(i) = resid(i) + X(i,k) * beta(k);
      }
      result = adrejSamplerBeta(k, sdDiag(k), q, U, p, sigma, X);
      beta(k) = result(0);
      nRejRec(sim, 2+k) = result(1);
      if (std::isinf(beta(k)) && k==1) {
        for (int i=0; i<10000; i++) {
          //graph2x(i) = -2 + i*3.0 / 10000.0;
          //graph2y(i) = LLbeta(graph2x(i), sdDiag(k), k, q, U, p, sigma, X);
          //graphS2(i) = adrejSamplerBeta(k, sdDiag(k), q, U, p, sigma, X);
          throw Rcpp::exception("The regression coefficients are inite!");
        }
        //return List::create(Rcpp::Named("beta1dx") = graph1x, Rcpp::Named("beta1dy") = graph1y, Rcpp::Named("beta2dx") = graph2x, Rcpp::Named("beta2dy") = graph2y, Rcpp::Named("beta1s") = graphS1, Rcpp::Named("beta2s") = graphS2,Rcpp::Named("beta") = beta, Rcpp::Named("aveN") = aveN, // Rcpp::Named("sigma1") = sigma1Rec, Rcpp::Named("sigma2") = sigma2Rec,
        //Rcpp::Named("d") = dRec, Rcpp::Named("alpha") = alphaRec, Rcpp::Named("sigma") = sigmaRec, Rcpp::Named("q") = qRec);
      }
      if (std::isinf(beta(k)) && k==0) {
        for (int i=0; i<10000; i++) {
          //graph1x(i) = -2 + i*5.0 / 10000.0;
          //graph1y(i) = LLbeta(graph1x(i), sdDiag(k), k, q, U, p, sigma, X);
          //graphS1(i) = adrejSamplerBeta(k, sdDiag(k), q, U, p, sigma, X);
          throw Rcpp::exception("The regression coefficients are inite!");
        }
        //return List::create(Rcpp::Named("beta1dx") = graph1x, Rcpp::Named("beta1dy") = graph1y, Rcpp::Named("beta2dx") = graph2x, Rcpp::Named("beta2dy") = graph2y, Rcpp::Named("beta1s") = graphS1, Rcpp::Named("beta2s") = graphS2,Rcpp::Named("beta") = beta, Rcpp::Named("aveN") = aveN, // Rcpp::Named("sigma1") = sigma1Rec, Rcpp::Named("sigma2") = sigma2Rec,
        //Rcpp::Named("d") = dRec, Rcpp::Named("alpha") = alphaRec, Rcpp::Named("sigma") = sigmaRec, Rcpp::Named("q") = qRec);
      }
      //Rcout << beta(k) <<"\n";
      resid = Y - X * beta;   //update the residual
    }
    //betaRec.row(sim) = Rcpp::as<Rcpp::NumericVector>(wrap(beta));

    //calculate the location shift
    NumericVector gammaAdj(n);
    NumericVector tauAdj(n);
    NumericVector sigmaAdj(n);
    double sumAdj = 0.0;
    temp = 0.0;
    for (i=0; i<n-1; i++) {
      gammaAdj(i) = double(clusterSize(i)) / (double) size;
      if (gammaAdj(i)>0.1) {
        temp++;
      }
      sumAdj += gammaAdj(i);
      tauAdj(i) = beta(0)-qStar(i);
      //Rcout<<gammaAdj(i)<<"\n";
      sigmaAdj(i) = sigmaStar(i);
    }
    gammaAdj(n-1) = 1.0 - sumAdj;
    if (gammaAdj(n-1) > 0.1) {
      temp++;
    }
    tauAdj(n-1) = beta(0)-qStar(n-1);
    sigmaAdj(n-1) = sigmaStar(n-1);
    if (sampleQ) {
      interceptAdj(sim) = sampleQuantileL(resid, p) + beta(0);
    }
    else {
      interceptAdj(sim) = findRootNewtonQ(gammaAdj, tauAdj, sigmaAdj, p, 0.00000001);
    }
    if (interceptAdj(sim) != interceptAdj(sim)) {
      Rcout<< interceptAdj(sim) << "\t" << n <<"\n";
      Rcout << "gamma";
      for (i=0; i<n; i++) {
        Rcout <<"\t" << gammaAdj(i);
      }
      Rcout<<"\n";
      Rcout << "tau";
      for (i=0; i<n; i++) {
        Rcout <<"\t" << tauAdj(i);
      }
      Rcout<<"\n";
      Rcout << "sigma";
      for (i=0; i<n; i++) {
        Rcout <<"\t" << sigmaAdj(i);
      }
      Rcout<<"\n";
      throw Rcpp::exception("NAN produced!!!");
    }
    //Rcout << "Finished" <<"\n";
    //free(gammaAdj);
    //free(tauAdj);
    //free(sigmaAdj);
    if (plotDensity) {
      if (sampleQ) {
        temp = findRootNewtonQ(gammaAdj, tauAdj, sigmaAdj, p, 0.00000001);
      }
      else {
        temp = interceptAdj(sim);
      }
      if (sim > burn) {
        for (int j=0; j<n; j++) {
          for (int i=0; i<4001; i++) {
            sumdense(i) += (gammaAdj(j)*dLogistic(xGrid(i), p, tauAdj(j)-temp, sigmaAdj(j)))/(nsim-burn);
          }
        }
      }
    }

    betaRec.row(sim) = as<NumericVector>(wrap(beta));
    betaRec(sim, 0) = interceptAdj(sim);
    sigmaRec.row(sim) = sigma;
    qRec.row(sim) = q-beta(0)+interceptAdj(sim);

    dRec(sim) = d;
    clusterRec.row(sim) = s;
    nClusterRec(sim) = n;
    nClusterTRec(sim) = temp;
    alphaRec(sim) = alpha;
    //Rcout << "*************************************************\n";
    //Rcout << "\n";
  }
  //betaRecFile.close();
  aveN /= nsim;
  return List::create(Rcpp::Named("dense.mean") = sumdense, Rcpp::Named("xGrid")=xGrid, Rcpp::Named("interceptAdj")=interceptAdj, Rcpp::Named("nRej") = nRejRec, Rcpp::Named("nCluster") = nClusterRec, Rcpp::Named("nClusterT") = nClusterTRec, Rcpp::Named("cluster") = clusterRec, Rcpp::Named("aveN") = aveN, //Rcpp::Named("beta1dx") = graph1x, Rcpp::Named("beta1dy") = graph1y, Rcpp::Named("beta2dx") = graph2x, Rcpp::Named("beta2dy") = graph2y, Rcpp::Named("beta1s") = graphS1, Rcpp::Named("beta2s") = graphS2,
    Rcpp::Named("d") = dRec, Rcpp::Named("lambda")=lambda, Rcpp::Named("alpha") = alphaRec, Rcpp::Named("beta") = betaRec, Rcpp::Named("sigma") = sigmaRec, Rcpp::Named("q") = qRec);           // Return to R
}
