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

void insertSortNHP(int k, NumericVector values, IntegerVector ind, int n, int left){
  //insert k-th value (we know it is new) into vector values keeping values sorted
  //left the mid calculated in findDupNHP1
  //this function is used together with findDupNHP1 to do the unique sort
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

void testNHP (NumericVector values){
  values(0) = 0;
}

int findDupNHP(NumericVector values1, NumericVector values2, double value1, double value2, int n){
  //if value is in the vector values (of length n not necessary equal to value.size()), return the index, otherwise return -1
  for (int i=0; i<n; i++){
    if (value1 == values1(i) && value2 == values2(i)){
      return i;
    }
  }
  return -1;
}

int findDupNHP1(NumericVector values, IntegerVector ind, int k, int n){
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

NumericMatrix calcByClusterNHP(NumericVector Yr, NumericVector sigma2, NumericVector mu, IntegerVector s) {
  NumericMatrix result = NumericMatrix(2, Yr.size());
  for (int i=0; i<Yr.size(); i++) {
    result(0, s(i)) += (Yr(i) - mu(i)) * (Yr(i) - mu(i));
    result(1, s(i)) += Yr(i);
  }
  return result;
}


double calcQNormalMixNHP(NumericVector prob, NumericVector sigma2Star, NumericVector muStar, double p, int n) {
  double qleft = ::Rf_qnorm5(p, muStar(0), sqrt(sigma2Star(0)), 1, 0);
  if (n==1) {
    return qleft;
  }
  double qright = qleft;
  double qmid = qright;
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

//' BQR based on DPM of Normal Distribution? (what is 'P'?)
//'
//' Bayesian quantile regression with heteroskedasticity with Dirichlet process mixture of normal distribution.
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
//' @param hasHeter Boolean indicating if Heteroskedasticity is present. Default is True.
//' @return List of model parameters including the coefficients.
//' @export
//'
//[[Rcpp::export]]
List bqrDPMNHP (NumericMatrix Xr, NumericVector Yr, int T, double p, int nsim, NumericVector initCoef, int burn = 5000, bool plotDensity = false, bool hasHeter = true) {
  // nsim is the number of simulation
  // Y is the response vector
  // T is the number of observations for each subject.
  if (p<0 || p>1){
    throw Rcpp::exception("the percentile should be between 0 and 1!");
  }

  int size = Yr.size(); //sample size
  if (size % T != 0) {
    throw Rcpp::exception("There are missing values for some observations");
  }
  int nsub = size / T; //number of subjects
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
  arma::colvec varDiag = 100000000.0 * arma::ones(ncov);  //diagnal elements of the covariance matrix (diagonal) of the prior (multivariate normal)
                                            //of the regression coeffcients
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
  NumericVector theta(nsub, 0.0); //fixed effects
  arma::colvec beta = initCoef;  //regression coefficient
  arma::colvec gamma(ncov);
  for (i=0; i<ncov; i++) {
    gamma(i) = 0.0;
  }
  NumericVector sigh(size, 1.0);
  double d = 0.0;  //scale parameter for the base measure (inverse gamma) of the DP
  NumericVector resid(size);   //residual
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
  NumericVector interceptRec(nsim);
  NumericVector alphaRec(nsim);
  NumericMatrix betaRec(nsim, ncov);
  NumericMatrix gammaRec(nsim, ncov);
  NumericMatrix thetaRec(nsim, nsub);
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
  arma::colvec varTemp(size);
  arma::colvec sigma2Temp(size);
  arma::mat covM(ncov,ncov);
  arma::colvec Theta(size);
  NumericVector U(size);
  NumericVector V(size);
  NumericVector prob(2*size);   //probabilities used to draw from the mixture
  double probCont;  //used to draw from the mixture
  double temp;
  double temp1;
  NumericVector acc(ncov, 0.0);
  NumericVector att(ncov, 0.0);
  NumericVector can(ncov, 0.25);
  double mhRate;
  NumericVector nAcceptGamma(ncov, 0.0);
  NumericVector cansigh(size);


  //Gibbs sampler
  //calculate the initial number of clusters and configuation parameters
  n = 0;
  sigma2Star(0) = sigma2(0);
  muStar(0) = mu(0);
  s(0) = 0;
  clusterSize(0) = 1;
  n = 1;
  for (i=1; i<size; i++){
    s(i) = findDupNHP(sigma2Star, muStar, sigma2(i), mu(i), n);
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
    if (sim%5000==0) {
      Rcout<<sim<<"\n";
    }

    //Gibbs sampling for the regression coefficients
    for (i=0; i<size; i++) {
      Theta(i) = theta(i/T);
      varTemp(i) = sigh(i)*sigh(i)*sigma2(i);
      normTemp(i) = (Yr(i) - Theta(i) - mu(i)*sigh(i)) / varTemp(i);
    }
    covM = X.t() * arma::diagmat(1/varTemp) * X + arma::diagmat(1/varDiag);
    covM = inv(covM);
    NumericVector tempV = rnorm(ncov);
    arma::colvec tempArmaV(tempV.begin(), ncov, false);
    beta = covM*X.t()*normTemp + arma::chol(covM).t()*tempArmaV;
    resid = Y - X*beta - Theta;



    //Update gamma
    if (hasHeter) {
      for (i=0; i<ncov; i++) {
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
          mhRate = ::Rf_dnorm4(cangamma, 0.0, sqrt(varDiag(i)), 1) - ::Rf_dnorm4(gamma(i), 0.0, sqrt(varDiag(i)), 1);
          for (j=0; j<size; j++) {
            mhRate += ::Rf_dnorm4(resid(j), mu(j)*cansigh(j), sqrt(sigma2(j))*cansigh(j), 1) - ::Rf_dnorm4(resid(j), mu(j)*sigh(j), sqrt(sigma2(j))*sigh(j), 1);
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

    //Gibbs sampling for sigma2Star and muStar
    tempm = calcByClusterNHP(resid, sigma2, mu, s);
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
        sigma2Star(n) = sigma2(i);
        muStar(n) = mu(i);
        clusterSize(n) = 1;
        clusterSize(s(i))--;
        n++;
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
      s(i) = findDupNHP(sigma2Star, muStar, sigma2(i), mu(i), n);
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

    //Rcout << "updating the fixed effects...\n";
    //Gibbs sampling for the fixed effects
    U = as<NumericVector>(wrap(Y - X*beta)) - sigh * mu;
    V = sigma2 * sigh * sigh;
    for (i=0; i<nsub; i++) {
      temp = 0.0;
      temp1 = 0.0;
      for (j=i*T; j<(i+1)*T; j++) {
      	temp += 1.0 / V(j);
      	temp1 += U(j) / V(j);
      }
      temp += 1.0 / varDiag(0);
      theta(i) = ::Rf_rnorm(temp1/temp, sqrt(1/temp));
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


    double intercept = calcQNormalMixNHP(prob, sigma2Star, muStar, p, n);
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
      if (sim > burn) {
        for (j=0; j<n; j++) {
          for (i=0; i<4001; i++) {
            sumdense(i) += prob(j) * ::Rf_dnorm4(xGrid(i), muStar(j)-intercept, sqrt(sigma2Star(j)), 0) / (nsim-burn);
          }
        }
      }
    }

    gammaRec.row(sim) = as<NumericVector>(wrap(gamma));
    interceptRec(sim) = intercept;
    for (i=0; i<ncov; i++) {
      betaRec(sim, i) = beta(i) + intercept * gamma(i);
    }
    muRec.row(sim) = mu + as<NumericVector>(wrap(Theta)) - intercept;
    sigma2Rec.row(sim) = sigma2;
    thetaRec.row(sim) = theta+intercept;
    dRec(sim) = d;
    alphaRec(sim) = alpha;
    clusterSizeRec(sim) = n;
    //Rcout << "*************************************************\n";
    //Rcout << "\n";
  }
  //betaRecFile.close();
  //test

  return List::create(Rcpp::Named("gammaAccRate") = nAcceptGamma/nsim, Rcpp::Named("gamma") = gammaRec, Rcpp::Named("intercept") = interceptRec, Rcpp::Named("beta") = betaRec, Rcpp::Named("dense.mean") = sumdense, Rcpp::Named("xGrid") = xGrid, Rcpp::Named("nCluster") = clusterSizeRec, Rcpp::Named("sigma2") = sigma2Rec, Rcpp::Named("mu") = muRec,
  Rcpp::Named("d") = dRec, Rcpp::Named("alpha") = alphaRec, Rcpp::Named("theta")=thetaRec);           // Return to R
}
