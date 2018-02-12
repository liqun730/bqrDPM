// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <string>
#include <iostream>
#include <fstream>
//#include <Rcpp.h>

using namespace Rcpp;
using namespace arma;
using namespace std;

const double pi = 3.1415926;

//calculate the sample quantile
double sampleQuantileMN(NumericVector array, double p) {
  NumericVector data = clone(array);
  int n = (int)(data.size()*p);
  std::nth_element(data.begin(), data.begin()+n, data.end());
  double result = data(n);
  std::nth_element(data.begin(), data.begin()+n-1, data.end());
  result = (data(n) + data(n-1))/2;
  return result;
}

void insertSortMN(int k, NumericVector values, IntegerVector ind, int n, int left){
  //insert k-th value (we know it is new) into vector values keeping values sorted
  //left the mid calculated in findDupMN1
  //this function is used together with findDupMN1 to do the unique sort
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

void test (NumericVector values){
  values(0) = 0;
}

int findDupMN(NumericVector values1, NumericVector values2, double value1, double value2, int n){
  //if value is in the vector values (of length n not necessary equal to value.size()), return the index, otherwise return -1
  for (int i=0; i<n; i++){
    if (value1 == values1(i) && value2 == values2(i)){
      return i;
    }
  }
  return -1;
}

int findDupMN1(NumericVector values, IntegerVector ind, int k, int n){
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

NumericMatrix calcByClusterMN(NumericVector Yr, NumericVector sigma2, NumericVector mu, IntegerVector s) {
  NumericMatrix result = NumericMatrix(2, Yr.size());
  for (int i=0; i<Yr.size(); i++) {
    result(0, s(i)) += (Yr(i) - mu(i)) * (Yr(i) - mu(i));
    result(1, s(i)) += Yr(i);
  }
  return result;
}

//[[Rcpp::export]]
double calcQNormalMixMN(NumericVector prob, NumericVector sigma2Star, NumericVector muStar, double p, int n) {
  double qleft = ::Rf_qnorm5(p, muStar(0), sqrt(sigma2Star(0)), 1, 0);
  if (n==1) {
    return qleft;
  }
  double qright = qleft;
  double qmid = qleft;
  double temp;
  int i;
  for (i=1; i<n; i++) {
    temp = ::Rf_qnorm5(p, muStar(i), sqrt(sigma2Star(i)), 1, 0);
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
      temp += prob(i) * ::Rf_pnorm5(qmid, muStar(i), sqrt(sigma2Star(i)), 1, 0);
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

//' BQR based on DPM of Normal Distribution
//'
//' Bayesian quantile regression with Dirichlet process mixture of normal distribution.
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
//' @return List of model parameters including the coefficients.
//' @export
//'
//[[Rcpp::export]]
List bqrDPMN (NumericMatrix Xr, NumericVector Yr, double p, int nsim, NumericVector initCoef, bool sampleQ = true, int burn = 5000, bool plotDensity = false) {
  // nsim is the number of simulation
  // sampleQ is a flag which determines how to calculate the location shift. (true means using sample quantile, false means using the density estimation)
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
  int j;
  int sim;


  //fixed hyperparameters
  double c = 2.0;  //shape parameter for the base measure (inverse gamma) of the first DP
  arma::colvec varDiag = 100000000.0 * arma::ones(ncov);  //diagnal elements of the covariance matrix (diagonal) of the prior (multivariate normal) of the regression coeffcients
  double a2 = 1.0;
  double b2 = 1.0;
  double a1 = -1.0;
  double b1;
  for (i=0; i<size; i++){
    if (Y(i) > 0 && Y(i) > a1){
      a1 = Y(i);
    }
    if (Y(i) < 0 && a1 + Y(i) < 0){
      a1 = -Y(i);
    }
  }
  b1 = 1.0 / a1 / a1;
  a1 = 1.0;

  //intermediate parameters
  int n = size;   //number of clusters of DP
  IntegerVector s(size);   //configuation parameters for the first DP
  NumericVector muStar(size*2);
  NumericVector sigma2Star(size*2);   //unique values for the first DP
  IntegerVector clusterSize(size*2);   //cluster size for the first DP

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
  d = var / size;


  ////Rcout << d1 <<"\t" << d2 <<"\t" << n1 <<"\t" << n2 << "\n";
  double alpha = ::Rf_rgamma(a2, 1.0 / b2);  //the precision parameter for DP
  NumericVector sigma2(size);//
  sigma2 = 1.0 / rgamma(size, c, 1.0 / d);   //left endpoint of the step function * (-1)
  NumericVector mu(size);
  for(i=0; i<size; i++) {
    mu(i) = ::Rf_rnorm(0.0, sqrt(sigma2(i)));
  }

  double ita = R::runif(0.0, 1.0);   //the augment parameter to help update the precision parameters


  //matrix to restore the mcmc samples
  //NumericMatrix betaRec(nsim, ncov);
 // NumericMatrix sigma1Rec(nsim, size);
 // NumericMatrix sigma2Rec(nsim, size);
  NumericVector dRec(nsim);
  NumericVector alphaRec(nsim);
  NumericMatrix betaRec(nsim, ncov);
  NumericMatrix sigma2Rec(nsim, size);
  NumericMatrix muRec(nsim, size);
  IntegerVector clusterSizeRec(nsim);

  NumericVector xGrid(4001);
  xGrid(0) = -20.0;
  for (int i=1; i<4001; i++) {
    xGrid(i) = xGrid(i-1) + 0.01;
  }
  NumericVector sumdense(4001, 0.0);

  //temporary variables
  NumericMatrix tempm(2, size);
  arma::colvec normTemp(size);
  arma::colvec sigma2Temp(size);
  arma::mat covM(ncov,ncov);
  NumericVector prob(2*size);   //probabilities used to draw from the mixture
  double probCont;  //used to draw from the mixture
  double temp;


  //Gibbs sampler
  //calculate the initial number of clusters and configuation parameters
  n = 0;
  sigma2Star(0) = sigma2(0);
  muStar(0) = mu(0);
  s(0) = 0;
  clusterSize(0) = 1;
  n = 1;
  for (i=1; i<size; i++){
    s(i) = findDupMN(sigma2Star, muStar, sigma2(i), mu(i), n);
    if (s(i) == -1){
      sigma2Star(n) = sigma2(i);
      muStar(n) = mu(i);
      s(i) = n;
      clusterSize(n) = 1;
      n++;
    }
    else{
      clusterSize(s(i))++;
    }
  }

  for (sim=0; sim<nsim; sim++){
    //Rcout << "*************************************************\n";
    //Rcout<<sim<<"\n";
    //if (sim % 1000 == 0) {
    //   Rcout<<sim<<"\n";
    //}

    //Gibbs sampling for the regression coefficients
    for (i=0; i<size; i++) {
      normTemp(i) = (Yr(i) - mu(i)) / sigma2(i);
    }
    arma::colvec sigma2Temp(sigma2.begin(), size, false);
    covM = X.t() * arma::diagmat(1/sigma2Temp) * X + arma::diagmat(1/varDiag);
    covM = inv(covM);
    NumericVector tempV = rnorm(ncov);
    arma::colvec tempArmaV(tempV.begin(), ncov, false);
    //use Cholesky decomposition
    beta = covM*X.t()*normTemp + arma::chol(covM).t()*tempArmaV;

    //use eigen decomposition
    //arma::mat evec;
    //arma::colvec eval;
    //arma::eig_sym(eval, evec, covM);
    //Rcout << eval(0) <<"\n";
    //beta = covM*X.t()*normTemp + evec * arma::diagmat(arma::sqrt(eval)) *tempArmaV;


    resid = Y - X*beta;

    //Gibbs sampling for sigma2Star and muStar
    tempm = calcByClusterMN(resid, sigma2, mu, s);
    for (j=0; j<n; j++) {
      sigma2Star(j) = 1.0 / ::Rf_rgamma(c+0.5+clusterSize(j)/2, 1.0/(d+muStar(j)*muStar(j)/2+tempm(0,j)/2 ));
      muStar(j) = ::Rf_rnorm(tempm(1,j)/(clusterSize(j)+1.0), sqrt(sigma2Star(j)/(clusterSize(j)+1.0)));
    }

    //Gibbs sampling for sigma and mu
    for (i=0; i<size; i++) {
      probCont = alpha * pow(d, c) * ::Rf_gammafn(c+0.5) / ::Rf_gammafn(c) / 2 / sqrt(pi) / pow(d+resid(i)*resid(i)/4.0, c+0.5);
      //Rcout<<probCont << "\t" << pow(d+Yr(i)*Yr(i)/4.0-mu(i)*mu(i)/2.0, c+0.5) << "\t" << d+Yr(i)*Yr(i)/4.0-mu(i)*mu(i)/2.0<<"\n";
      for (j=0; j<n; j++) {
        if (j != s(i)){
          prob(j) = clusterSize(j) * ::Rf_dnorm4(resid(i), muStar(j), sqrt(sigma2Star(j)), 0);
        }
        else {
          prob(j) = (clusterSize(j) - 1.0) * ::Rf_dnorm4(resid(i), muStar(j), sqrt(sigma2Star(j)), 0);
        }
      }
      temp = probCont;
      for (j=0; j<n; j++){
        temp += prob(j);
      }
      if (temp == 0){
        throw Rcpp::exception("Something is wrong!");
      }

      probCont /= temp;
      for (j=0; j<n; j++){
        prob(j) /= temp;
      }
      temp = R::runif(0.0, 1.0);
      if (temp <= probCont){
        sigma2(i) = 1.0 / ::Rf_rgamma(c+1.0, 1.0/(d+mu(i)*mu(i)/2.0+(resid(i)-mu(i))*(resid(i)-mu(i))/2.0));
        mu(i) = ::Rf_rnorm(resid(i)/2.0, sqrt(sigma2(i)/2.0));
        int ind = findDupMN(sigma2Star, muStar, sigma2(i), mu(i), n);
        if (ind == -1) {
          sigma2Star(n) = sigma2(i);
          ////Rcout<<"flag1!";
          clusterSize(n) = 1;
          clusterSize(s(i))--;
          ////Rcout<<"flag1! "<<n1;
          n++;
        }
        else {
          clusterSize(s(i))--;
          clusterSize(ind)++;
        }
      }
      else{
        double cumProb = probCont;
        for (j=0; j<n; j++){
          if (temp > cumProb && temp <= cumProb + prob(j)){
            sigma2(i) = sigma2Star(j);
            mu(i) = muStar(j);
            if (j != s(i)) {
             // //Rcout<<"flag2";
              clusterSize(s(i))--;
              clusterSize(j)++;
             // //Rcout<<"flag2"<<"\n";
            }
            ////Rcout << sigma1(i) <<"\t";
            break;
          }
          cumProb += prob(j);
        }
      }
    }


    //Rcout<<"calculating the number of clusters and configuation parameters...\n";
    //calculate number of clusters and configuation parameters after the above update
    n = 0;
    sigma2Star(0) = sigma2(0);
    muStar(0) = mu(0);
    s(0) = 0;
    clusterSize(0) = 1;
    n = 1;
    for (i=1; i<size; i++){
      s(i) = findDupMN(sigma2Star, muStar, sigma2(i), mu(i), n);
      if (s(i) == -1){
        sigma2Star(n) = sigma2(i);
        muStar(n) = mu(i);
        s(i) = n;
        clusterSize(n) = 1;
        n++;
      }
      else{
        clusterSize(s(i))++;
      }
    }

    //Rcout << "updating the precision parameters... \n";
    //Gibbs sampling for the precision parameter of DP
    temp = (a2 + n -1) / (a2 + n - 1 + size*(b2 - log(ita)));
    if (R::runif(0.0, 1.0) <= temp) {
      alpha = ::Rf_rgamma(a2 + n, 1.0 / (b2 - log(ita)));
    }
    else {
      alpha = ::Rf_rgamma(a2 + n - 1.0, 1.0 / (b2 - log(ita)));
    }
    ita = ::Rf_rbeta(alpha+1.0, size);

    //Rcout << "updating the scale parameters in the base measure... \n";
    //Gibbs sampling for the scale parameter in the base measure of DP
    temp = 0.0;
    for (j=0; j<n; j++){
      temp += 1.0 / sigma2Star(j);
    }
    d = ::Rf_rgamma(a1 + n*c, 1.0 / (b1 + temp));


    //shift to guarantee p-th quantile 0 constaint
    double pSum = 0.0;
    for (j=0; j<n-1; j++) {
      prob(j) = (double) clusterSize(j) / (double) size;
      pSum += prob(j);
    }
    prob(n-1) = 1.0 - pSum;


    double intercept;
    if (sampleQ) {
      intercept = sampleQuantileMN(resid, p) + beta(0);
    }
    else {
      intercept = calcQNormalMixMN(prob, sigma2Star, muStar+beta(0), p, n);
    }
    //debug
    /*
    if (intercept < 0.001 && sim>20000) {
      Rcout<<"prob\n";
      for (i=0; i<n; i++) {
        Rcout<<prob(i) <<"\t";
      }
      Rcout<<"\n";
      Rcout<<"sigma2Star\n";
      for (i=0; i<n; i++) {
        Rcout<<sigma2Star(i) << "\t";
      }
      Rcout<<"\n";
      Rcout<<"muStar+beta(0)\n";
      for (i=0; i<n; i++) {
        Rcout<<muStar(i)+beta(0)<<"\t";
      }
      Rcout<<"\n";
      throw Rcpp::exception("Intecept calculation went wrong!!\n");
    }
    */


    if (plotDensity) {
      if (sampleQ) {
        intercept = calcQNormalMixMN(prob, sigma2Star, muStar+beta(0), p, n);
      }
      if (sim > burn) {
        for (j=0; j<n; j++) {
          for (i=0; i<4001; i++) {
            sumdense(i) += prob(j) * ::Rf_dnorm4(xGrid(i), muStar(j)+beta(0)-intercept, sqrt(sigma2Star(j)), 0) / (nsim-burn);
          }
        }
      }
    }

    betaRec.row(sim) = as<NumericVector>(wrap(beta));
    betaRec(sim, 0) = intercept;
    muRec.row(sim) = mu + beta(0) - intercept;
    sigma2Rec.row(sim) = sigma2;
    dRec(sim) = d;
    alphaRec(sim) = alpha;
    clusterSizeRec(sim) = n;
    //Rcout << "*************************************************\n";
    //Rcout << "\n";
  }
  //betaRecFile.close();
  //test

  return List::create(Rcpp::Named("beta") = betaRec, Rcpp::Named("dense.mean") = sumdense, Rcpp::Named("xGrid") = xGrid, Rcpp::Named("nCluster") = clusterSizeRec, Rcpp::Named("sigma2") = sigma2Rec, Rcpp::Named("mu") = muRec,
  Rcpp::Named("d") = dRec, Rcpp::Named("alpha") = alphaRec);           // Return to R
}



