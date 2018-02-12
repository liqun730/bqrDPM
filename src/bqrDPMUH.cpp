// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <string>
#include <iostream>
#include <fstream>
//#include <Rcpp.h>

using namespace Rcpp;
using namespace arma;
using namespace std;

void insertSortUH(int k, NumericVector values, IntegerVector ind, int n, int left){
  //insert k-th value (we know it is new) into vector values keeping values sorted
  //left the mid calculated in findDupUH1
  //this function is used together with findDupUH1 to do the unique sort
  //we know ind is not empty
  if (left == 0){
    if (values(k) < values(ind(0))){
      for (int i=n; i>0; i--){
        ind(i) = ind(i-1);
      }
      ind(0) = k;
    }
    else if (values(k) < values(ind(n-1))){
      for (int i=n; i>1; i--){
        ind(i) = ind(i-1);
      }
      ind(1) = k;
    }
    else {
      ind(n) = k;
    }
  }
  else if (left == n-2 && n > 2){
    if (values(k) < values(ind(n-1))){
      ind(n) = ind(n-1);
      ind(n-1) = k;
    }
    else {
      ind(n) = k;
    }
  }
  else {
    for (int i=n; i>left+1; i--){
      ind(i) = ind(i-1);
    }
    ind(left+1) = k;
  }
}

void testUH (NumericVector values){
  values(0) = 0;
}

int findDupUH(NumericVector values, double value, int n){
  //if value is in the vector values (of length n not necessary equal to value.size()), return the index, otherwise return -1
  for (int i=0; i<n; i++){
    if (value == values(i)){
      return i;
    }
  }
  return -1;
}

int findDupUH1(NumericVector values, IntegerVector ind, int k, int n){
  //if k-th value is in the sorted subset of the vector values (of length n indexed by ind), return the index, otherwise return -1
  int left = 0;
  int right = n-1;
  int mid = int(floor((left + right)/2.0));
  while (left + 1 < right){
    ////Rcout <<left << "\t" << mid <<"\t" <<right <<"\n";
    if (values(k) < values(ind(mid))){
      right = mid;
      mid = int(floor((left + right) / 2.0));
    }
    else if (values(k) > values(ind(mid))){
      left = mid;
      mid = int(floor((left + right) / 2.0));
    }
    else {
      return mid;
    }
  }
  if (values(k) == values(ind(left))){
    return left;
  }
  if (values(k) == values(ind(right))){
    return right;
  }
  return -mid-1;
}

double kernelDensUH(double x, double left, double right, double p){
  // the kernel density (step function with pth quantile zero)
  if (x + left <= 0){
    return 0;
  }
  else if (x < 0){
    return p / left;
  }
  else if (x < right){
    return (1-p) / right;
  }
  else{
    return 0;
  }
}

double lkernelDensUH(double x, double left, double right, double p){
  // the kernel density (step function with pth quantile zero)
  if (x + left <= 0){
    return -1.0/0.0;
  }
  else if (x < 0){
    return log(p) - log(left);
  }
  else if (x < right){
    return log(1-p) - log(right);
  }
  else{
    return -1.0/0.0;
  }
}

IntegerVector findLowerBoundUH1(arma::colvec resid, int ind, IntegerVector s){
  // find the index for lower bound for the truncated inverse gamma used to update the unduplicated left endpoints
  // also return the number of samples that can be used to update the unduplicated left endpoints
  // resid is the residual vector
  // ind is the index of the unduplicated value
  // s is the configuation vector
  double max = -1.0;
  int n = resid.size();
  IntegerVector v(2); // the first element is the index of the maximum and the second element is the number of relevant sample
  v(1) = 0;
  for (int i=0; i<n; i++){
    if (s(i) == ind && resid(i) < 0){
      v(1)++;
      if (max + resid(i) < 0){
        max = -resid(i);
        v(0) = i;
      }
    }
  }
  return v;
}

IntegerVector findLowerBoundUH2(arma::colvec resid, int ind, IntegerVector s){
  // find the index for lower bound for the truncated inverse gamma used to update the unduplicated right endpoints
  // also return the number of samples that can be used to update the unduplicated right endpoints
  // resid is the residual vector
  // ind is the index of the unduplicated value
  // s is the configuation vector
  double max = -1.0;
  int n = resid.size();
  IntegerVector v(2); // the first element is the index of the maximum and the second element is the number of relevant sample
  v(1) = 0;
  for (int i=0; i<n; i++){
    if (s(i) == ind && resid(i) > 0){
      v(1)++;
      if (max - resid(i) < 0){
        max = resid(i);
        v(0) = i;
      }
    }
  }
  return v;
}

double f1UH(double p, double sigma1, double sigma2, double x){
  //help to draw the regression coefficients
  if (x < 0){
    return (1.0 - p) / sigma2;
  }
  else {
    return p / sigma1;
  }
}

double f2UH(double p, double sigma1, double sigma2, double x){
  //help to draw the regression coefficients
  if (x > 0){
    return (1.0 - p) / sigma2;
  }
  else {
    return p / sigma1;
  }
}

double trunGammaLeftUH(double a, double b, double t){
  // efficient sampling from left truncated gamma distribution with shape parameter a,
  //scale parameter b, truncated at t
  if (t < 0){
    throw Rcpp::exception("the truncated value should be nonnegative!");
  }
  if (t == 0){
    return ::Rf_rgamma(a, b);
  }
  double newb = b / t; // after this transformation, we can assume t=1
  int newa = int(a);
  NumericVector v(newa);
  NumericVector w(newa);
  v(0) = 1.0;
  w(0) = 1.0;
  for (int i=1; i<newa; i++){
    v(i) = v(i-1) * (a - i) * newb;
    w(i) = w(i-1) + v(i);
  }
  for (int i=0; i<newa; i++){
    w(i) = v(i) / w(newa - 1);
  }
  double u = R::runif(0.0, 1.0);
  double randSample;
  if (u < w(0)){
     randSample = ::Rf_rgamma(1.0, newb) * t + t;
     return randSample;
  }
  else {
    for (int i=1; i<newa; i++){
      if (u >= w(i-1) && u < w(i)){
        randSample = ::Rf_rgamma(i + 1.0, newb) * t + t;
        return randSample;
      }
    }
    randSample = ::Rf_rgamma(newa + 1.0, newb) * t + t;
    return randSample;
  }
}

double proposalFcnUH(NumericVector w, double a, int nc){
  // proposal function to generate samples from right truncated gamma
  double u = R::runif(0.0, 1.0);
  if (u <= w(0)){
    return ::Rf_rbeta(a, 1.0);
  }
  for (int i=1; i<nc; i++){
    if (u > w(i-1) && u <= w(i)){
      return ::Rf_rbeta(a, i + 1.0);
    }
  }
  return ::Rf_rbeta(a, double(nc));
}

double trunGammaRightUH(double a, double b, double t){
  // efficient sampling from right truncated gamma distribution with shape parameter a,
  //scale parameter b, truncated at t
  if (t <= 0){
    //Rcout << "t = \t" << t <<"\n";
    throw Rcpp::exception("the truncated value should be positive!");
  }
  double newb = b / t; // after this transformation, we can assume t=1
  double q95 = ::Rf_qnorm5(0.95, 0.0, 1.0, 1, 0);
  int nc = int(pow(q95 + ::sqrt(pow(q95, 2.0) + 4 / newb), 2.0) / 4); //number of mixture component needed to have 95% acceptance probability
  ////Rcout<<"N = \t"<<nc<< "\t" << "newb = \t" << newb << "\t b = \t" <<b<< "\t t = \t" <<t <<"\n";
  NumericVector w(nc);
  NumericVector v(nc);
  w(0) = 1.0;
  v(0) = 1.0;
  for (int i=1; i<nc; i++){
    v(i) = v(i-1) / (a + i) / newb;
    w(i) = w(i-1) + v(i);
  }
  for (int i=0; i<nc; i++){
    w(i) = w(i) / w(nc-1);
  }
  double u;
  double x;
  double M = 0.0;
  double rho;
  for (int i=0; i<nc; i++){
    M += 1.0 / pow(newb, i) / ::Rf_gammafn(i + 1.0);
  }
  M = 1.0 / M;
  while (true){
    u = ::Rf_runif(0.0, 1.0);
    x = proposalFcnUH(w, a, nc);
    rho = 0.0;
    for (int i=0; i<nc; i++){
      rho += pow(1.0 - x, i) / ::Rf_gammafn(i + 1.0) / pow(newb, i);
    }
    rho *= exp(x / newb);
    rho = 1.0 / rho;
    if (u * M <= rho){
      return x * t;
    }
  }
}

double trunNormalLRUH (double left, double right, double mu, double sigma){
  //draw random sample from a truncated normal distribution
  if (left > right){
    throw Rcpp::exception("the left truncated value should be no bigger than the right truncated value");
  }
  if (left == right){
    return left;
  }
  double u = ::Rf_runif(0.0, 1.0);
  double lprob = Rf_pnorm5((left - mu) / sigma, 0.0, 1.0, 1, 0);
  double rprob = Rf_pnorm5((right - mu) / sigma, 0.0, 1.0, 1, 0);
  double randSample = mu + sigma * ::Rf_qnorm5(u * (rprob - lprob) + lprob , 0.0, 1.0, 1, 0);
  //if the tail is too far away from the mean, then just sample from uniform distribution
  if (randSample < left || randSample > right){
    return ::Rf_runif(left, right);
  }
  else {
    return randSample;
  }
}


double normalizeUH(NumericVector x, double y, int n){
  //x (of length n) and y combined as a vector of logrithm of the values to be nornmalized
  //values of x are modified through reference, while the value of normalized y is returned
  double max = y;
  for (int i=0; i<n; i++){
    if (x(i) > max){
      max = x(i);
    }
  }
  double denom = 0.0;
  for (int i=0; i<n; i++){
    denom += exp(x(i) - max);
  }
  denom += exp(y - max);
  //Rcout<<denom<<"\t"<<1/denom<<"\n";
  for (int i=0; i<n; i++){
    x(i) = exp(x(i) - max - log(denom));
    ////Rcout<<x(i) <<"-->"<<prob(i) << "\t";
  }
 // //Rcout<<"PROB\n";
 /*for (int i=0; i<n; i++) {
   Rcout<<x(i)<<"\t";
 }
 Rcout<<exp(y - max - log(denom))<<"\n";
 */
  return exp(y - max - log(denom));
}

//' BQR with Heteroskedasticity based on DPM of Uniform Distribution
//'
//' Bayesian quantile regression with Heteroskedasticity based on Dirichlet process mixture of uniform distribution.
//'
//' @param Xr Matrix of covariate.
//' @param Yr Vector of response.
//' @param p Quantile of interest.
//' @param nsim Number of simulation.
//' @param initCoef Initial value (vector) of the model coefficients.
//' @param burn Number of burn-in.
//' @param plotDensity A boolean variable which decides whether to plot the density estimation or not.
//' @param tol The tolerance value for the length of the interval to treat the normal density in this interval as constant.
//' @param hasHeter Boolean indicating if Heteroskedasticity is present. Default is True.
//' @return List of model parameters including the coefficients.
//' @export
//'
//[[Rcpp::export]]
List bqrDPMUH (NumericMatrix Xr, NumericVector Yr, double p, int nsim, NumericVector initCoef, int burn = 5000, bool plotDensity = false, double tol=0.0000001, bool hasHeter = true) {
  // nsim is the number of simulation
  // p is the quantile regression
  // X is the design matrix with each row correspnding to an observation
  // Y is the response vector
  // initCoef is the initial value for the regression coefficients
  // tol is the tolerance value for the length of the interval to treat the normal density in this interval as constant
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
  int j;
  int sim;


  //fixed hyperparameters
  double c1 = 2.0;  //shape parameter for the base measure (inverse gamma) of the first DP
  double c2 = 2.0;  //shape parameter for the base measure (inverse gamma) of the second DP
  NumericVector sdDiag = rep(10000.0, ncov);  //diagnal elements of the covariance matrix (diagonal) of the prior (multivariate normal)
                                            //of the regression coeffcients
  NumericVector mu = rep(0.0, ncov);  //mean vector of the prior (multivariate normal) of the regression coefficients
  double a1 = 1.0;   //shape parameter for the Gamma prior for the precision parameters of DP
  double b1 = 1.0;//10.0 / size;  //rate parameter for the Gamma prior for the precision parameters of DP
  double a2 = -1.0; //shape parameter for the Gamma prior for d1, d2
  double b2; //rate parameter for the Gamma prior for d1, d2
  for (i=0; i<size; i++){
    if (Y(i) > 0 && Y(i) > a2){
      a2 = Y(i);
    }
    if (Y(i) < 0 && a2 + Y(i) < 0){
      a2 = -Y(i);
    }
  }
  b2 = 1.0 / a2;
  a2 = 1.0;

  //intermediate parameters
  int n1 = 0;   //number of clusters of the first DP
  int n2 = 0;   //number of clusters of the second DP
  IntegerVector s1(size);   //configuation parameters for the first DP
  IntegerVector s2(size);   //configuation parameters for the second DP
  NumericVector sigmaStar1(size*2);   //unique values for the first DP
  NumericVector sigmaStar2(size*2);   //unique values for the second DP
  IntegerVector clusterSize1(size*2);   //cluster size for the first DP
  IntegerVector clusterSize2(size*2);   //cluster size for the second DP


  arma::colvec gamma(ncov);
  gamma(0) = 1.0;
  for (i=1; i<ncov; i++) {
    gamma(i) = 0.0;
  }
  //debug
  //gamma(1) = -1.0/1.1;
  //debug



  NumericVector sigh(size, 1.0);
  sigh = X * gamma;
  //initialize parameters to be updated
  arma::colvec beta = initCoef;  //regression coefficient
  double d1 = 0.0;  //scale parameter for the base measure (inverse gamma) of the first DP
  double d2 = 0.0;  //scale parameter for the base measure (inverse gamma) of the second DP
  arma::colvec resid(size);   //residual
  resid = Y - X * beta;   //calculate the residual
  for (i=0; i<size; i++) {
    resid(i) /= sigh(i);
  }
  for (i=0; i<size; i++){
    if (resid(i) < 0){
      d1 -= resid(i);
      ////Rcout << d1 <<"\n";
      n1++;
    }
    else {
      d2 += resid(i);
      n2++;
    }
  }
  if (n1 == 0){
    if (n2 == 0){
      d1 = 3.0;
      d2 = 3.0;
    }
    else {
      d2 /= n2;
      d1 = d2;
    }
  }
  else {
    if (n2 == 0){
      d1 /= n1;
      d2 = d1;
    }
    else {
      d1 /= n1;
      d2 /= n2;
    }
  }

  ////Rcout << d1 <<"\t" << d2 <<"\t" << n1 <<"\t" << n2 << "\n";
  double alpha1 = 1.0;  //the precision parameter for the first DP
  double alpha2 = 1.0;  // the precision parameter for the second DP
  NumericVector sigma1(size);// = rgamma(size, c1, 1.0 / d1);   //left endpoint of the step function * (-1)
  NumericVector sigma2(size);// = rgamma(size, c2, 1.0 / d2);   //right endpoint of the step function
  NumericVector sigma1Scale(size);
  NumericVector sigma2Scale(size);



  for (i=0; i<size; i++){
    if (resid(i) < 0){
      sigma1(i) = -2.0 * resid(i);
      sigma2(i) = -2.0 * resid(i);
    }
    else if (resid(i) > 0){
      sigma1(i) = 2.0 * resid(i);
      sigma2(i) = 2.0 * resid(i);
    }
    else {
      sigma1(i) = 1.0 / ::Rf_rgamma(c1, 1.0 / d1);
      sigma2(i) = 1.0 / ::Rf_rgamma(c1, 1.0 / d1);
    }
  }

  double ita1 = R::runif(0.0, 1.0);   //the augment parameter to help update the precision parameters
  double ita2 = R::runif(0.0, 1.0);


  //matrix to restore the mcmc samples
  NumericMatrix gammaRec(nsim, ncov);
  NumericMatrix betaRec(nsim, ncov);
  NumericMatrix sigma1Rec(nsim, size);
  NumericMatrix sigma2Rec(nsim, size);
  NumericVector d1Rec(nsim);
  NumericVector d2Rec(nsim);
  NumericVector alpha1Rec(nsim);
  NumericVector alpha2Rec(nsim);

  NumericVector acc(ncov, 0.0);
  NumericVector att(ncov, 0.0);
  NumericVector can(ncov, 0.25);
  double mhRate;
  NumericVector nAcceptGamma(ncov, 0.0);
  NumericVector cansigh(size);

  NumericVector xGrid(4001);
  xGrid(0) = -20.0;
  for (int i=1; i<4001; i++) {
    xGrid(i) = xGrid(i-1) + 0.01;
  }
  NumericVector sumdense(4001, 0.0);
  //ofstream betaRecFile;
  //std::string filename = path + "betaStepRecFile.txt";
  //betaRecFile.open (filename.c_str());
  //ofstream d1RecFile;
  //ofstream d2RecFile;
  //ofstream

  //Gibbs sampler
  NumericVector prob(size*2);   //probabilities used to draw from the mixture
  double probCont;  //used to draw from the mixture
  double randSample;  //random sample
  double temp;
  //double temp1;
  IntegerVector tempVec(size);
  int tempInd;
  int tempInd1;
  int tempInd2;
  int numNoSupport = 0;
  double aveN1 = 0.0;
  double aveN2 = 0.0;

  //NumericVector tempVecD(size+1);
  IntegerVector tempVec1(size);

  //calculate the initial number of clusters and configuation parameters
  n1 = 0;
  n2 = 0;
  sigmaStar1(0) = sigma1(0);
  s1(0) = 0;
  clusterSize1(0) = 1;
  n1 = 1;
  sigmaStar2(0) = sigma2(0);
  s2(0) = 0;
  clusterSize2(0) = 1;
  n2 = 1;
  for (i=1; i<size; i++){
    s1(i) = findDupUH(sigmaStar1, sigma1(i), n1);
    if (s1(i) == -1){
      sigmaStar1(n1) = sigma1(i);
      s1(i) = n1;
      clusterSize1(n1) = 1;
      n1++;
    }
    else{
      clusterSize1(s1(i))++;
    }
    s2(i) = findDupUH(sigmaStar2, sigma2(i), n2);
    if (s2(i) == -1){
      sigmaStar2(n2) = sigma2(i);
      s2(i) = n2;
      clusterSize2(n2) = 1;
      n2++;
    }
    else{
      clusterSize2(s2(i))++;
    }
  }

  for (sim=0; sim<nsim; sim++){
    //Rcout << "*************************************************\n";
    //Rcout<<sim<<"\n";

    //Rcout << "updating the unduplicated left endpoints...\n";
    //Gibbs sampling for the unduplicated left endpoints
    int noInfo = 0;
    for (j=0; j<n1; j++){
      IntegerVector v = findLowerBoundUH1(resid, j, s1);
      if (v(1) == 0){
        noInfo++;
        sigmaStar1(j) = 1.0 / ::Rf_rgamma(c1, 1.0 / d1);
        ////Rcout << sigmaStar1(j) <<"\t" <<  d1  << "\n";
      }
      else {
        temp = ::Rf_pgamma(-1.0 / resid(v(0)) , c1 + v(1), 1.0 / d1, 1, 0);
        if (temp > 0.95){
          randSample = -1.0;
          while (randSample + resid(v(0)) < 0){
            randSample = 1.0 / ::Rf_rgamma(c1 + v(1), 1.0 / d1);
          }
        }
        else {
          randSample = 1.0 / trunGammaRightUH(c1 + v(1), 1.0 / d1, -1.0 / resid(v(0)));
        }
        ////Rcout << randSample << "\t" << -resid(v(0)) << "\n";

        sigmaStar1(j) = randSample;
      }
      ////Rcout << sigmaStar1(j) <<"\t" << n1 <<"\n";
    }
    //Rcout<<noInfo<<"\n";

    //Rcout << "updating the unduplicated right endpoints...\n";
    //Gibbs sampling for the unduplicated right endpoints
    noInfo = 0;
    for (j=0; j<n2; j++){
      IntegerVector v = findLowerBoundUH2(resid, j, s2);
      if (v(1) == 0){
        noInfo++;
        sigmaStar2(j) = 1.0 / ::Rf_rgamma(c2, 1.0 / d2);
      }
      else {
        temp = ::Rf_pgamma(1.0 / resid(v(0)) , c2 + v(1), 1.0 / d2, 1, 0);
        if (temp > 0.95){
          randSample = -1.0;
          while (randSample - resid(v(0)) < 0){
            randSample = 1.0 / ::Rf_rgamma(c2 + v(1), 1.0 / d2);
          }
        }
        else {
          randSample = 1.0 / trunGammaRightUH(c2 + v(1), 1.0 / d2, 1.0 / resid(v(0)));
        }
        ////Rcout << randSample << "\t" << resid(v(0)) << "\n";
        sigmaStar2(j) = randSample;
      }
    }
    //Rcout<<noInfo<<"\n";

    int wrong = 0;
    //Rcout << "updating left endpoints...\n";
    //gibbs sampling for the left endpoints of the step function
    for (i=0; i<size; i++){
      ////Rcout << i <<"\n";
      //calculate the weight for the continuous part
      if (resid(i) < sigma2(i)){
        if (resid(i) < 0){
          ////Rcout << -resid(i) << "\t" <<beta(0) <<"\t"<<beta(1)<<"\n";
          temp = ::Rf_pgamma(- 1.0 / resid(i), c1 + 1.0, 1.0 / d1, 1, 0);
          probCont = p * alpha1 * c1 * temp / d1;
          //draw from the truncated inverse gamma
          ////Rcout<<"t = \t"<<-1.0 / resid(i)<<"\n";
          if (temp > 0.95){
            randSample = -1.0;
            while (randSample + resid(i) < 0){
              randSample = 1.0 / ::Rf_rgamma(c1 + 1, 1.0 / d1);
            }
          }
          else {
            randSample = 1.0 / trunGammaRightUH(c1 + 1, 1.0 / d1, -1.0 / resid(i));
          }
          ////Rcout << randSample << "\t" << -resid(i) << "\n";
        }
        else{
          if (resid(i) < sigma2(i)){
            probCont = (1.0 - p) * alpha1 / sigma2(i);
          }
          else {
            probCont = 0.0;
          }
          randSample = 1.0 / ::Rf_rgamma(c1, 1.0 / d1);
        }
        ////Rcout <<"error\t"<< 1 <<"\t" <<n1;
        //calculate the weight for the discrete part
        for (j=0; j<n1; j++){
          if (j != s1(i)){
            prob(j) = clusterSize1(j) * kernelDensUH(resid(i), sigmaStar1(j), sigma2(i), p);
            ////Rcout << resid(i) << "\t" << -sigmaStar1(j) << "\t" <<sigma2(i)<<"\t" << kernelDensUH(resid(i), sigmaStar1(j), sigma2(i), p) << "\n";
          }
          else {
            prob(j) = (clusterSize1(j) - 1.0) * kernelDensUH(resid(i), sigmaStar1(j), sigma2(i), p);
          }
        }
        ////Rcout <<"error\t"<< 2;
        //normalize

        temp = probCont;
        for (j=0; j<n1; j++){
          temp += prob(j);
        }
        if (temp == 0){
          throw Rcpp::exception("Something is wrong1!");
        }

        probCont /= temp;
        if (probCont!=probCont) {
          throw Rcpp::exception("NAN is produced when calculating the probability 1!");
        }
        for (j=0; j<n1; j++){
          prob(j) /= temp;
          if (prob(j)!=prob(j)) {
            throw Rcpp::exception("NAN is produced when calculating the probability 1!");
          }
        }
        ////Rcout<<"~~~~~~~~~~~~~~"<<probCont<<"\n";
      ////Rcout <<"error\t"<< 3;
      //sampling from the mixture of continuous and dicrete distribution with probability available
        temp = R::runif(0.0, 1.0);
        if (temp <= probCont){
          sigma1(i) = randSample;
          int ind = findDupUH(sigmaStar1, sigma1(i), n1);
          if (ind == -1) {
            sigmaStar1(n1) = sigma1(i);
            ////Rcout<<"flag1!";
            clusterSize1(n1) = 1;
            clusterSize1(s1(i))--;
            ////Rcout<<"flag1! "<<n1;
            n1++;
          }
          else {
            clusterSize1(s1(i))--;
            clusterSize1(ind)++;
          }


          ////Rcout << sigma1(i) <<"\t";
        }
        else{
          double cumProb = probCont;
          for (j=0; j<n1; j++){
            if (temp > cumProb && temp <= cumProb + prob(j)){
              sigma1(i) = sigmaStar1(j);
              if (j != s1(i)) {
               // //Rcout<<"flag2";
                clusterSize1(s1(i))--;
                clusterSize1(j)++;
               // //Rcout<<"flag2"<<"\n";
              }
              ////Rcout << sigma1(i) <<"\t";
              break;
            }
            cumProb += prob(j);
          }
        }
      }
      else{
        wrong ++;
      }
      // if the parameter does not support the data, do not update
    }
    //Rcout <<wrong <<"\n";
    //Rcout << "updating right endpoints... \n";
    //gibbs sampling for the right endpoints of the step function
    //Rcout <<n1<<"\t"<<n2<<"\n";
    for (i=0; i<size; i++){
      //calculate the weight for the continuous part
      if (resid(i) + sigma1(i) > 0){
        if (resid(i) > 0){
          temp = ::Rf_pgamma(1.0 / resid(i), c2 + 1.0, 1.0 / d2, 1, 0);
          probCont = (1 - p) * alpha2 * c2 * temp / d2;
          //draw from the truncated inverse gamma
          if (temp > 0.95){
            randSample = -1.0;
            while (randSample - resid(i) < 0){
              randSample = 1.0 / ::Rf_rgamma(c2 + 1.0, 1.0 / d2);
            }
          }
          else {
            randSample = 1.0 / trunGammaRightUH(c2 + 1.0, 1.0 / d2, 1.0 / resid(i));
          }
          ////Rcout << randSample << "\t" << resid(i) << "\n";

        }
        else{
          if (resid(i) + sigma1(i) > 0){
            probCont = p * alpha2 / sigma1(i);
          }
          else {
            probCont = 0.0;
          }
          randSample = 1.0 / ::Rf_rgamma(c2, 1.0 / d2);
        }

        //calculate the weight for the discrete part
        for (j=0; j<n2; j++){
          if (j != s2(i)){
            prob(j) = clusterSize2(j) * kernelDensUH(resid(i), sigma1(i), sigmaStar2(j), p);
          }
          else {
            prob(j) = (clusterSize2(j) - 1) * kernelDensUH(resid(i), sigma1(i), sigmaStar2(j), p);
          }
        }
        //normalize
        temp = probCont;
        for (j=0; j<n2; j++){
          temp += prob(j);
        }

        if (temp == 0){
          throw Rcpp::exception("Something is wrong2!");
        }
        probCont /= temp;
        if (probCont!=probCont) {
          throw Rcpp::exception("NAN is produced when calculating the probability 2!");
        }
        for (j=0; j<n2; j++){
          prob(j) /= temp;
          if (prob(j)!=prob(j)) {
            throw Rcpp::exception("NAN is produced when calculating the probability 2!");
          }
        }


        ////Rcout<<"~~~~~~~~~~~~~~"<<probCont<<"\n";
        //sampling from the mixture of continuous and dicrete distribution with probability available
        temp = R::runif(0.0, 1.0);
        if (temp <= probCont){
          sigma2(i) = randSample;
          int ind = findDupUH(sigmaStar2, sigma2(i), n2);
          if (ind == -1) {
            sigmaStar2(n2) = sigma2(i);
            clusterSize2(n2) = 1;
            clusterSize2(s2(i))--;
            n2++;
          }
          else {
            clusterSize2(s2(i))--;
            clusterSize2(ind)++;
          }
        }
        else{
          double cumProb = probCont;
          for (j=0; j<n2; j++){
            if (temp > cumProb && temp <= cumProb + prob(j)){
              sigma2(i) = sigmaStar2(j);
              if (j != s2(i)) {
                clusterSize2(s2(i))--;
                clusterSize2(j)++;
              }
              break;
            }
            cumProb += prob(j);
          }
        }
      }
    }

    //Rcout<<"calculating the number of clusters and configuation parameters...\n";
    //calculate number of clusters and configuation parameters after the above update
    n1 = 0;
    n2 = 0;
    sigmaStar1(0) = sigma1(0);
    s1(0) = 0;
    clusterSize1(0) = 1;
    n1 = 1;
    sigmaStar2(0) = sigma2(0);
    s2(0) = 0;
    clusterSize2(0) = 1;
    n2 = 1;
    for (i=1; i<size; i++){
      s1(i) = findDupUH(sigmaStar1, sigma1(i), n1);
      if (s1(i) == -1){
        sigmaStar1(n1) = sigma1(i);
        s1(i) = n1;
        clusterSize1(n1) = 1;
        n1++;
      }
      else{
        clusterSize1(s1(i))++;
      }
      s2(i) = findDupUH(sigmaStar2, sigma2(i), n2);
      if (s2(i) == -1){
        sigmaStar2(n2) = sigma2(i);
        s2(i) = n2;
        clusterSize2(n2) = 1;
        n2++;
      }
      else{
        clusterSize2(s2(i))++;
      }
    }

    //Rcout<<"n1\t"<<n1<<"n2\t"<<n2<<"\n";
    aveN1 += n1;
    aveN2 += n2;


    //Rcout << "updating the precision parameters... \n";
    //Gibbs sampling for the precision parameter of the first DP

    temp = (a1 + n1 -1) / (a1 + n1 - 1 + size*(b1 - log(ita1)));
    if (R::runif(0.0, 1.0) <= temp) {
      alpha1 = ::Rf_rgamma(a1 + n1, 1.0 / (b1 - log(ita1)));
    }
    else {
      alpha1 = ::Rf_rgamma(a1 + n1 - 1.0, 1.0 / (b1 - log(ita1)));
    }
    ita1 = ::Rf_rbeta(alpha1+1.0, size);

    //Gibbs sampling for the precision parameter of the second DP

    temp = (a1 + n2 -1) / (a1 + n2 - 1 + size*(b1 - log(ita2)));
    if (R::runif(0.0, 1.0) <= temp) {
      alpha2 = ::Rf_rgamma(a1 + n2, 1.0 / (b1 - log(ita2)));
    }
    else {
      alpha2 = ::Rf_rgamma(a1 + n2 - 1.0, 1.0 / (b1 - log(ita2)));
    }
    ita2 = ::Rf_rbeta(alpha2+1.0, size);


    //Rcout << "updating the scale parameters in the base measure... \n";
    //Gibbs sampling for the scale parameter in the base measure of the first DP
    temp = 0.0;
    for (j=0; j<n1; j++){
      temp += 1.0 / sigmaStar1(j);
    }
    d1 = ::Rf_rgamma(a2 + n1 * c1, 1.0 / (b2 + temp));

    //Gibbs sampling for the scale parameter in the base measure of the second DP
    temp = 0.0;
    for (j=0; j<n2; j++){
      temp += 1.0 / sigmaStar2(j);
    }
    d2 = ::Rf_rgamma(a2 + n2 * c2, 1.0 / (b2 + temp));

    //Rcout << "updating regression coefficients...\n";
    //Gibbs sampling for the regression coefficients
    NumericVector u(size);
    for (i=0; i<size; i++) {
      resid(i) *= sigh(i);
      sigma1Scale(i) = sigma1(i) * sigh(i);
      sigma2Scale(i) = sigma2(i) * sigh(i);
    }


    for (int k=0; k<ncov; k++){
    //for (int k=ncov-1; k>-1; k--){
      for (i=0; i<size; i++){
        u(i) = (resid(i) + X(i,k) * beta(k)) / X(i,k);
      }
      //calculate the support of u values
      double lowerBound = -std::numeric_limits<double>::infinity();
      double upperBound = std::numeric_limits<double>::infinity();
      for (i=0; i<size; i++){
        if (X(i,k) < 0){
          temp = u(i) - sigma2Scale(i) / X(i,k);

          if (temp < upperBound){
            upperBound = temp;
          }
          temp = u(i) + sigma1Scale(i) / X(i,k);
          if (temp > lowerBound){
            lowerBound = temp;
          }
          ////Rcout<<lowerBound<<"<"<<upperBound<<"\n";
        }
        if (X(i,k) > 0){
          temp = u(i) + sigma1Scale(i) / X(i,k);

          if (temp < upperBound){
            upperBound = temp;
          }
          temp = u(i) - sigma2Scale(i) / X(i,k);
          if (temp > lowerBound){
            lowerBound = temp;
          }
          ////Rcout<<lowerBound<<"<"<<upperBound<<"\n";
        }
      }
      //Rcout<<lowerBound<<"\t"<<upperBound<<"\n";
      //if the likelihood is 0, we still update using the middle interval
      if (lowerBound > upperBound){
        numNoSupport++;
        temp = upperBound;
        upperBound = lowerBound;
        lowerBound = temp;
      }
      tempInd = 0;
      //pick u within the support
      for (i=0; i<size; i++){
        if (u(i) < upperBound && u(i) > lowerBound){
          tempVec(tempInd) = i;
          tempInd++;
        }
      }
      //no u values in the support
      if (tempInd == 0){
        beta(k) = trunNormalLRUH(lowerBound, upperBound, mu(k), sdDiag(k));
      }
      else {
        //find unique u values and sort them into an increasing sequence
        tempInd1 = 1;
        tempVec1(0) = tempVec(0);
        for (i=0; i<tempInd; i++){
          tempInd2 = findDupUH1(u, tempVec1, tempVec(i), tempInd1);
          if (tempInd2 < 0){
            insertSortUH(tempVec(i), u, tempVec1, tempInd1, -(tempInd2+1));
            tempInd1++;
          }
        }

        //draw random variable from each mixing part, and calculate the corresponding probability
        for (i=0; i<tempInd1-1; i++){
          ////Rcout<< i <<"\tout of\t"<<tempInd-1<<"\n";
          prob(i) = log(::Rf_pnorm5(u(tempVec1(i+1)), mu(k), sdDiag(k), 1, 0) - ::Rf_pnorm5(u(tempVec1(i)), mu(k), sdDiag(k), 1, 0));
          for (j=0; j<tempInd; j++){
            if (u(tempVec(j)) <= u(tempVec1(i)) && X(tempVec(j), k) != 0){
              prob(i) += log(f1UH(p, sigma1Scale(tempVec(j)), sigma2Scale(tempVec(j)), X(tempVec(j), k)));
            }
            if (u(tempVec(j)) >= u(tempVec1(i+1)) && X(tempVec(j), k) != 0){
              prob(i) += log(f2UH(p, sigma1Scale(tempVec(j)), sigma2Scale(tempVec(j)), X(tempVec(j), k)));
            }
          }
          //prob(i) = exp(prob(i));
        }

        probCont = log(::Rf_pnorm5(u(tempVec1(0)), mu(k), sdDiag(k), 1, 0) - ::Rf_pnorm5(lowerBound, mu(k), sdDiag(k), 1, 0));
        for (j=0; j<tempInd; j++){
          if (X(tempVec(j), k) != 0) {
            probCont += log(f2UH(p, sigma1Scale(tempVec(j)), sigma2Scale(tempVec(j)), X(tempVec(j), k)));
          }
        }
        //debug
        //Rcout<<"probCont1\t"<<probCont<<"\n";
        //debug
        //probCont = exp(probCont);
        //debug
        //Rcout<<"probCont2\t"<<probCont<<"\n";
        //debug

        ////Rcout<< "mixture component finished\n";
        ////Rcout << u(tempVec1(tempInd1-1)) << "\t" << upperBound << "\n";

        prob(tempInd1-1) = log(::Rf_pnorm5(upperBound, mu(k), sdDiag(k), 1, 0) - ::Rf_pnorm5(u(tempVec1(tempInd1-1)), mu(k), sdDiag(k), 1, 0));
        for (j=0; j<tempInd; j++){
          if (X(tempVec(j), k) != 0) {
            prob(tempInd1-1) += log(f1UH(p, sigma1Scale(tempVec(j)), sigma2Scale(tempVec(j)), X(tempVec(j), k)));
          }
        }
        //prob(tempInd1-1) = exp(prob(tempInd1-1));



        ////Rcout<< "mixture component finished\n";
        //normalize the probability
        probCont = normalizeUH(prob, probCont, tempInd1);

        /*Rcout<<"Regression Coefficient # "<<k<<"\n";
        Rcout<<"[("<<lowerBound<<","<<u(tempVec1(0))<<")~~"<<probCont<<"]\t";
        for (int i=0; i<tempInd1-1; i++) {
          Rcout<<"[("<<u(tempVec1(i))<<","<<u(tempVec1(i+1))<<")~~"<<prob(i)<<"]\t";
        }
        Rcout<<"[("<<u(tempVec1(tempInd1-1))<<","<<upperBound<<")~~"<<prob(tempInd1-1)<<"]\n"; */
        /*temp = probCont;
        for (i=0; i<tempInd1; i++){
          temp += prob(i);
        }
        if (temp == 0.0){
          Rcout<<probCont<<"\t";
          for (i=0; i<tempInd1; i++) {
            Rcout<<prob(i)<<"\t";
          }
          Rcout<<"\nu\n";
          for (i=0; i<tempInd1; i++) {
            Rcout<<u(tempVec1(i))<<"\t";
          }
          Rcout<<"\nbounds\t"<<upperBound<<"\t"<<lowerBound<<"\n";
          Rcout<<log(::Rf_pnorm5(u(tempVec1(0)), mu(k), sdDiag(k), 1, 0) - ::Rf_pnorm5(lowerBound, mu(k), sdDiag(k), 1, 0))<<"\t"<<u(tempVec1(0))<<"\t"<<lowerBound<<"\n";
          throw Rcpp::exception("Something is wrong3!");
        }
        else {
          probCont /= temp;
          for (i=0; i<tempInd1; i++){
            prob(i) /= temp;
          }
        }*/

        //sampling from the mixture of continuous and dicrete distribution with probability available
        temp = R::runif(0.0, 1.0);
        if (temp <= probCont){
          beta(k) = trunNormalLRUH(lowerBound, u(tempVec1(0)), mu(k), sdDiag(k));
        }
        else if (temp + prob(tempInd1-1) > 1){
          beta(k) = trunNormalLRUH(u(tempVec1(tempInd1-1)), upperBound, mu(k), sdDiag(k));
        }
        else{
          double cumProb = probCont;
          for (j=0; j<tempInd1-1; j++){
            if (temp > cumProb && temp <= cumProb + prob(j)){
              beta(k) = trunNormalLRUH(u(tempVec1(j)), u(tempVec1(j+1)), mu(k), sdDiag(k));
              break;
            }
            cumProb += prob(j);
          }
        }
      }
      //Rcout << beta(k) <<"\n";
      //Rcout<<k<<"      "<<tempInd1<<"     ["<<lowerBound<<",  "<<upperBound<<"]     "<<"\n";
      resid = Y - X * beta;   //update the residual
    }
    //betaRec.row(sim) = Rcpp::as<Rcpp::NumericVector>(wrap(beta));

    //update gamma

    if (hasHeter) {
      for (i=1; i<ncov; i++) {
        att[i]++;
        double cangamma = ::Rf_rnorm(gamma(i), 2*can[i]);
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
          bool hasZero = false;
          mhRate = ::Rf_dnorm4(cangamma, 0.0, sdDiag(i), 1) - ::Rf_dnorm4(gamma(i), 0.0, sdDiag(i), 1);
          for (j=0; j<size; j++) {
            if (kernelDensUH(resid(j), sigma1(j)*cansigh(j), sigma2(j)*cansigh(j), p) == 0) {
              hasZero = true;
              break;
            }
            mhRate += lkernelDensUH(resid(j), sigma1(j)*cansigh(j), sigma2(j)*cansigh(j), p) - lkernelDensUH(resid(j), sigma1(j)*sigh(j), sigma2(j)*sigh(j), p);
          }
          if (hasZero) {
            continue;
          }
        }
        if (!(mhRate!=mhRate)) {
          //Rcout<<cangamma<<"\t"<<mhRate<<"\n";
          if (::Rf_runif(0,1) < exp(mhRate)) {
            gamma(i) = cangamma;
            sigh = clone(cansigh);
            nAcceptGamma(i)++;
            acc[i]++;
          }
        }
      }


      for (i=0; i<size; i++) {
        resid(i) /= sigh(i);
      }
      for (i=0; i<ncov; i++) {
        if (att(i)>200 && sim<=burn) {
          can(i) = can(i) * (acc(i)/att(i)<0.2?0.5:1) * (acc(i)/att(i)>0.7?1.5:1);
          acc(i) = 1.0;
          att(i) = 1.0;
        }
      }
    }

    /* //debug
    gamma(1) = -1.0/1.1;
    sigh = X * gamma;
    for (i=0; i<size; i++) {
      resid(i) /= sigh(i);
    }*/

    NumericVector p1(n1);
    NumericVector p2(n2);
    double pSum = 0.0;
    for (int i=0; i<n1-1; i++) {
      p1(i) = (double) clusterSize1(i) / (double) size;
      pSum += p1(i);
    }
    p1(n1-1) = 1.0 - pSum;
    pSum = 0.0;
    for (int i=0; i<n2-1; i++) {
      p2(i) = (double) clusterSize2(i) / (double) size;
      pSum += p2(i);
    }
    p2(n2-1) = 1.0 - pSum;


    if (plotDensity) {
      if (sim > burn) {
        for (int j=0; j<n1; j++) {
          for (int i=0; i<4001; i++) {
            sumdense(i) += (p*p1(j)*((xGrid(i)>-sigmaStar1(j) && xGrid(i)<0)? 1.0:0.0)/sigmaStar1(j))/(nsim-burn);
          }
        }
        for (int j=0; j<n2; j++) {
          for (int i=0; i<4001; i++) {
            sumdense(i) += ((1.0-p)*p2(j)*((xGrid(i)<sigmaStar2(j) && xGrid(i)>=0)? 1.0:0.0)/sigmaStar2(j))/(nsim-burn);
          }
        }
      }
    }

    betaRec.row(sim) = as<NumericVector>(wrap(beta));
    gammaRec.row(sim) = as<NumericVector>(wrap(gamma));
    sigma1Rec.row(sim) = sigma1;
    sigma2Rec.row(sim) = sigma2;
    d1Rec(sim) = d1;
    d2Rec(sim) = d2;
    alpha1Rec(sim) = alpha1;
    alpha2Rec(sim) = alpha2;
    //Rcout << "*************************************************\n";
    //Rcout << "\n";
  }
  //betaRecFile.close();
  aveN1 /= nsim;
  aveN2 /= nsim;
  //test

  return List::create(Rcpp::Named("gamma") = gammaRec, Rcpp::Named("gammaAcceptRate") = nAcceptGamma/nsim, Rcpp::Named("dense.mean") = sumdense, Rcpp::Named("xGrid") = xGrid, Rcpp::Named("beta") = betaRec, Rcpp::Named("aveN1") = aveN1, Rcpp::Named("aveN2") = aveN2, Rcpp::Named("numNoSupport") = numNoSupport, Rcpp::Named("sigma1") = sigma1Rec, Rcpp::Named("sigma2") = sigma2Rec,
  Rcpp::Named("d1") = d1Rec, Rcpp::Named("d2") = d2Rec, Rcpp::Named("alpha1") = alpha1Rec, Rcpp::Named("alpha2") = alpha2Rec);           // Return to R
}



