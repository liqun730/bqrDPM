// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <ctime>
//#include <Rcpp.h>

using namespace Rcpp;
//using namespace arma;
//using namespace std;

//define other functions called by bqrDPMMN

//compute the stick-breaking weights


NumericVector makeProbsMNA(NumericVector v) {
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

const double root2piMNA = sqrt(2*3.141592653589793238463);
double dNormMNA(double x, double mu, double sigma) {
  return exp(-pow((x-mu)/sigma, 2.0)/2)/sigma/root2piMNA;
}


//calculate the mixing probability of the mixture of two normal distributions

double makeQMNA(double mu1, double mu2, double sigma1, double sigma2, double tau) {
  return (tau - Rf_pnorm5(0.0, mu2, sigma2, 1, 0)) / (Rf_pnorm5(0.0, mu1, sigma1, 1, 0) - Rf_pnorm5(0.0, mu2, sigma2, 1, 0));
}

//Asymmetric Laplace density [1/lambda is reparametrized by lambda] (in the log scale) (single variable)

double fASLsMNA(double y, double mu, double lambda, double tau) {
  return log(lambda) - ((y>mu)? tau : (1.0-tau)) * std::abs(y-mu) * lambda;
}


double absTestMNA(double a, double b) {
  return std::abs(a-b);
}

//Asymmetric Laplace density [1/lambda is reparametrized by lambda] (in the log scale) (multiple variables)

double fASLmMNA(NumericVector y, double mu, double lambda, double tau) {
  int n = y.size();
  double val = double(n) * log(lambda);
  for (int i=0; i<n; i++) {
    val -= ((y(i)>mu)? tau : (1.0-tau)) * std::abs(y(i)-mu) * lambda;
  }
  return val;
}

//generate an asymmetric Laplace random variable

double rASLsMNA (double lambda, double tau) {
  //RNGScope scope;
  double posneg = ::Rf_rbinom(1.0,tau);
  double vpos = ::Rf_rexp(1.0/(tau*lambda));
  double vneg = ::Rf_rexp(1.0/((1.0-tau)*lambda));
  //Rcout<<posneg<<"\t"<<vpos<<"\t"<<vneg;
  return (1.0 - posneg) * vpos - posneg * vneg;
}


NumericMatrix rngCppMNA(const int N, double lambda, double tau) {
  //RNGScope scope;  	// ensure RNG gets set/reset
  NumericMatrix X(N, 3);
  X(_, 0) = rbinom(N,1.0,tau);
  X(_, 1) = rexp(N, tau*lambda);
  X(_, 2) = rexp(N, (1.0-tau)*lambda);
  return X;
}

//generate n asymmetric Laplace random variables

NumericVector rASLmMNA (int n, double lambda, double tau) {
  RNGScope scope;
  NumericVector posneg = rbinom(n,1.0,tau);
  NumericVector a = rexp(n, tau*lambda);
  NumericVector b = rexp(n, (1.0-tau)*lambda);
  return (1.0 - posneg) * a - posneg * b;
}


NumericVector testMNA (NumericVector a, NumericVector b) {
  return (1.0-a)*b;
}

//generate random variables from multivariate normal distribution

arma::mat mvrnormArmaMNA(int n, arma::vec mu, arma::mat sigma) {
  int ncols = sigma.n_cols;
  arma::mat Y = arma::randn(n, ncols);
  return arma::repmat(mu, 1, n).t() + Y * arma::chol(sigma);
}

//sample from consecutive integers with replacement according to given probabilities

IntegerVector sampleWRMNA(int start, int end, int size, NumericVector probs) {
  int n = probs.size();
  if (n != end-start+1) {
    throw(Rcpp::exception("Something is wrong with sampleWRMNA 1!"));
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
    //throw(Rcpp::exception("Something is wrong with sampleWRMNA 2!"));
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

int sampleOneMNA(int start, int end, arma::colvec probs) {
  //test
  /*
  for (int i=0; i<probs.size(); i++) {
    Rcout<<probs(i)<<"\t";
  }
  Rcout<<"\n";*/
  //test
  int n = probs.size();
  if (n != end-start+1) {
    throw(Rcpp::exception("Something is wrong with sampleWRMNA 1!"));
  }
  double temp;
  NumericVector endpts(n+1);
  endpts(0) = 0.0;
  for (int i=1; i<n+1; i++) {
    endpts(i) = endpts(i-1) + probs(i-1);
  }
  //test
  //Rcout<<endpts(n)<<"  ***********\n";
  //test
  if (endpts(n) != 1.0) {
    //need to normalize probs
    for (int i=1; i<n; i++) {
      endpts(i) /= endpts(n);
    }
    endpts(n) = 1.0;
    //throw(Rcpp::exception("Something is wrong with sampleWRMNA 2!"));
  }
  temp = ::Rf_runif(0.0, 1.0);
  for (int j=0; j<n; j++) {
    if (temp>endpts(j) && temp<=endpts(j+1)) {
      return j + start;
      break;
    }
  }
  return n-1+start;
}


NumericVector rnormTMNA(int n, double mu, double sigma) {
  NumericVector sample(n);
  for (int i=0; i<n; i++) {
    sample(i) = ::Rf_rnorm(mu, sigma);
  }
  return sample;
}


arma::mat testcholMNA(arma::mat X) {
  return arma::chol(X);
}


arma::mat testMulMNA(arma::mat X, double a) {
  return a*X;
}


arma::mat testDivMNA(arma::mat X, double a) {
  return 1./X;
}


arma::colvec testNormalMNA(int n) {
  return arma::randn(n);
}


NumericVector testNormal1MNA(int n) {
  return rnorm(n);
}

//' BQR based on DPM of Mixure of Normal Distributions (what is 'A'?)
//'
//' Bayesian quantile regression based on Dirichlet process mixture of mixture of two normal distributions.
//'
//' @param Xr Matrix of covariate.
//' @param Yr Vector of response.
//' @param tau Quantile of interest.
//' @param nsim Number of simulation.
//' @param initCoef Initial value (vector) of the model coefficients.
//' @param burn Number of burn-in.
//' @param plotDensity A boolean variable which decides whether to plot the density estimation or not.
//' @param nF The number of terms in the finite approximation.
//' @param aD,bD V_k~Beta(1, D) and D~dgamma(aD, bD).
//' @param maxSigma The maximum possible value for sigma.
//' @param precBeta beta[j] ~ norm(0, 1/sqrt(prec.beta)).
//' @return List of model parameters including the coefficients.
//' @export
//'
//[[Rcpp::export]]
List bqrDPMMNA(NumericMatrix Xr, NumericVector Yr, double tau, int nsim, NumericVector initCoef, int burn = 5000, bool plotDensity = false, int nF=10, double aD=100, double bD=100, double maxSigma=1.0/0.0, double precBeta = 0.000001) {
  //initialization
  //nsim is the number of simulation
  //nF is the number of terms in the finite approximation
  //tau is the quantile of interest
  //Xr is the design matrix with the first column being ones
  //Yr is the response
  //initCoef is the initial guess for the regression coefficient
  //aD, bD      V_k~Beta(1, D)    D~dgamma(aD, bD)
  //maxSigma is the maximum possible value for sigma
  //precBeta   beta[j] ~ norm(0, 1/sqrt(prec.beta))
  RNGScope scope;
  int n = Yr.size();
  int p = Xr.ncol();
  arma::mat X(Xr.begin(), n, p, false);
  arma::colvec Y(Yr.begin(), n, false);


  //Generate initial valuss
  double D = 1.0;
  double minSigma = 0.2;
  if (std::isinf(maxSigma)) {
    double sdy = 0;
    double meany = 0;
    for (int i=0; i<n; i++) {
      meany += Yr(i);
    }
    meany /= double(n);
    for (int i=0; i<n; i++) {
      sdy += pow(Yr(i)-meany, 2.0);
    }
    sdy /= double(n-1);
    sdy = sqrt(sdy);
    maxSigma = 2 * sdy;
    //Rcout<<sdy<<"\n";
  }
  arma::colvec beta = initCoef;
  double lambda = 10.0;
  NumericVector q(nF, 2.0);
  NumericVector sigma1(nF);
  NumericVector sigma2(nF);
  NumericVector mu1(nF);
  NumericVector mu2(nF);
  for (int i=0; i<nF; i++) {
    while(q(i)<0 || q(i)>1) {
      mu1(i) = rASLsMNA(lambda, tau);
      //Rcout<<mu1(i)<<"\n";
      mu2(i) = rASLsMNA(lambda, tau);
      sigma1(i) = ::Rf_runif(minSigma, maxSigma);
      sigma2(i) = ::Rf_runif(minSigma, maxSigma);
      q(i) = makeQMNA(mu1(i),mu2(i),sigma1(i),sigma2(i),tau);
      //Rcout<<mu1(i)<<"\n";
    }
  }
  NumericVector v = rbeta(nF, 1.0, D);
  v(nF-1) = 1.0;
  NumericVector probs = makeProbsMNA(v);
  IntegerVector g = sampleWRMNA(0,nF-1, n, probs);
  //test
  /*
  IntegerVector clusterSize1(nF, 0);
  IntegerVector largerCount1(nF, 0);
  for (int i=0; i<n; i++) {
    clusterSize1(g(i))++;
    for (int j=0; j<g(i); j++) {
      largerCount1(j)++;
    }
  }
  for (int i=0; i<nF; i++) {
    Rcout<<clusterSize1(i)<<"\t";
  }
  Rcout<<"\n";
  for (int i=0; i<nF; i++) {
    Rcout<<largerCount1(i)<<"\t";
  }
  Rcout<<"\n";*/
  //test
  NumericVector h(n);
  for (int i=0; i<n; i++) {
    h(i) = ::Rf_rbinom(1,q(g(i)));
  }

  //keep track of stuff
  NumericMatrix keepBeta(nsim, p);
  NumericMatrix keepMu1(nsim, nF);
  NumericMatrix keepMu2(nsim, nF);
  NumericMatrix keepSigma1(nsim, nF);
  NumericMatrix keepSigma2(nsim, nF);
  NumericVector keepNCluster(nsim, 0.0);
  NumericVector keepNClusterT(nsim, 0.0);

  double rejRateMu1 = 0.0;
  double rejRateMu2 = 0.0;
  double rejRateSigma1 = 0.0;
  double rejRateSigma2 = 0.0;

  NumericVector truncProb(nsim);
  NumericVector xGrid(4001);
  xGrid(0) = -20.0;
  for (int i=1; i<4001; i++) {
    xGrid(i) = xGrid(i-1) + 0.01;
  }
  NumericVector ddd(4001);
  NumericVector sumdense(4001, 0.0);
  NumericVector sumdense2(4001, 0.0);
  NumericVector acc(6, 0.0);
  NumericVector att(6, 0.0);
  NumericVector can(6, 0.25);
  arma::colvec normTemp(n);
  arma::colvec sdTemp(n);
  arma::mat covM(p,p);
  NumericVector resids(n);
  arma::colvec lll(nF);
  double canmu1;
  double canmu2;
  double cansigma1;
  double cansigma2;
  double canq;
  double p0;
  double p1;
  double mhRate;
  double sss;
  double temp;
  IntegerVector indSelect(n);



  NumericVector r(n);
  int numSelect;
  //std::clock_t start;
  //start the mcmc sampling
  for (int sim=0; sim<nsim; sim++) {
    //Rcout<<"update beta\n";
    //start = std::clock();
    //for (int timer=0; timer<25000; timer++){
    for (int i=0; i<n; i++) {
    //update beta
      sdTemp(i) = h(i) * sigma1(g(i)) + (1.0 - h(i)) * sigma2(g(i));
      normTemp(i) = (Yr(i) - (h(i) * mu1(g(i)) + (1.0 - h(i)) * mu2(g(i)))) / (pow(sdTemp(i), 2.0));
    }
    covM = X.t() * arma::diagmat(1/(sdTemp%sdTemp)) * X + precBeta * arma::diagmat(arma::ones(p));
    covM = inv(covM);
    NumericVector tempV = rnorm(p);
    arma::colvec tempArmaV(tempV.begin(), p, false);
    beta = covM*X.t()*normTemp + arma::chol(covM).t()*tempArmaV;
    resids = Y - X*beta;
    //}Rcout<<"Time for updating beta is "<<(std::clock() - start) <<"/"<< CLOCKS_PER_SEC;

    //Rcout<<"update g,h\n";
    //update g, h
    //start = std::clock();
    //for (int timer=0; timer<25000; timer++){
    for (int i=0; i<n; i++) {
      if (h(i) == 0) {
        for (int j=0; j<nF; j++) {
          lll(j) = log(1.0-q(j)) + ::Rf_dnorm4(resids(i), mu2(j), sigma2(j), 1) + log(probs(j));
        }
      }
      if (h(i) == 1) {
        for (int j=0; j<nF; j++) {
          lll(j) = log(q(j)) + ::Rf_dnorm4(resids(i), mu1(j), sigma1(j), 1) + log(probs(j));
        }
      }
      g(i) = sampleOneMNA(0,nF-1,arma::exp(lll-arma::max(lll)));
    }
    //Rcout<<"update g,h2\n";
    for (int i=0; i<n; i++) {
      p1 = q(g(i)) * ::Rf_dnorm4(resids(i), mu1(g(i)), sigma1(g(i)), 0);
      p0 = (1.0 - q(g(i))) * ::Rf_dnorm4(resids(i), mu2(g(i)), sigma2(g(i)), 0);
      h(i) = ::Rf_rbinom(1, p1/(p0+p1));
    }
    //}Rcout<<"Time for updating g,h is "<<(std::clock() - start) <<"/"<< CLOCKS_PER_SEC;

    //start = std::clock();
    //Rcout<<"update 4\n";
    //for (int timer=0; timer<25000; timer++){
    for (int l=0; l<nF; l++) {
      //update mu1
      if (probs(l) > 0.2) {
        att(0)++;
      }
      canmu1 = ::Rf_rnorm(mu1(l), can(0));

      canq = makeQMNA(canmu1, mu2(l), sigma1(l), sigma2(l), tau);
      if (canq>0 && canq<1) {
        mhRate = fASLsMNA(canmu1, 0, lambda, tau) - fASLsMNA(mu1(l), 0, lambda, tau);
        numSelect = 0;
        for (int i=0; i<n; i++) {
          if (g(i) == l) {
            if (h(i) == 1) {
              indSelect(numSelect) = i;
              numSelect++;
            }
            mhRate += h(i) * log(canq) + (1.0-h(i)) * log(1.0-canq) - h(i) * log(q(l)) - (1.0-h(i)) * log(1.0-q(l));
          }
        }
        //test
        //Rcout<<canmu1<<"\t"<<mhRate<<"\t"<<canq<<"\n";
        if (numSelect > 0) {
          for (int j=0; j<numSelect; j++) {
            mhRate += ::Rf_dnorm4(resids(indSelect(j)), canmu1, sigma1(l), 1) - ::Rf_dnorm4(resids(indSelect(j)), mu1(l), sigma1(l), 1);
          }
        }
        if (::Rf_runif(0,1) < exp(mhRate)) {
          if (probs(l)>0.2) {
            acc(0)++;
          }
          q(l) = canq;
          mu1(l) = canmu1;
        }
        else {
          rejRateMu1++;
        }
      }

      //update mu2
      if (probs(l) > 0.2) {
        att(1)++;
      }
      canmu2 = ::Rf_rnorm(mu2(l), can(1));
      canq = makeQMNA(mu1(l), canmu2, sigma1(l), sigma2(l), tau);
      if (canq>0 && canq<1) {
        mhRate = fASLsMNA(canmu2, 0, lambda, tau) - fASLsMNA(mu2(l), 0, lambda, tau);
        numSelect = 0;
        for (int i=0; i<n; i++) {
          if (g(i) == l) {
            if (h(i) == 0) {
              indSelect(numSelect) = i;
              numSelect++;
            }
            mhRate += h(i) * log(canq) + (1.0-h(i)) * log(1.0-canq) - h(i) * log(q(l)) - (1.0-h(i)) * log(1.0-q(l));
          }
        }
        if (numSelect > 0) {
          for (int j=0; j<numSelect; j++) {
            mhRate += ::Rf_dnorm4(resids(indSelect(j)), canmu2, sigma2(l), 1) - ::Rf_dnorm4(resids(indSelect(j)), mu2(l), sigma2(l), 1);
          }
        }
        if (::Rf_runif(0,1) < exp(mhRate)) {
          if (probs(l)>0.2) {
            acc(1)++;
          }
          q(l) = canq;
          mu2(l) = canmu2;
        }
        else {
          rejRateMu2++;
        }
      }

      for (int i=0; i<n; i++) {
        r(i) = resids(i) - h(i) * mu1(g(i)) - (1.0-h(i))*mu2(g(i));
      }

      //update sigma1
      if (probs(l) > 0.2) {
        att(2)++;
      }
      cansigma1 = ::Rf_rnorm(sigma1(l), can(2));
      if (cansigma1>minSigma && cansigma1<maxSigma) {
        canq = makeQMNA(mu1(l), mu2(l), cansigma1, sigma2(l), tau);
        if (canq>0 && canq<1) {
          mhRate = 0.0;
          numSelect = 0;
          for (int i=0; i<n; i++) {
            if (g(i) == l) {
              if (h(i) == 1) {
                indSelect(numSelect) = i;
                numSelect++;
              }
              mhRate += h(i) * log(canq) + (1.0-h(i)) * log(1.0-canq) - h(i) * log(q(l)) - (1.0-h(i)) * log(1.0-q(l));
            }
          }
          if (numSelect > 0) {
            for (int j=0; j<numSelect; j++) {
              mhRate += ::Rf_dnorm4(r(indSelect(j)), 0, cansigma1, 1) - ::Rf_dnorm4(r(indSelect(j)), 0, sigma1(l), 1);
            }
          }
          if (::Rf_runif(0,1) < exp(mhRate)) {
            if (probs(l) > 0.2) {
              acc(2)++;
            }
            q(l) = canq;
            sigma1(l) = cansigma1;
          }
          else {
            rejRateSigma1++;
          }
        }
      }

      //update sigma2
      if (probs(l) > 0.2) {
        att(2)++;
      }
      cansigma2 = ::Rf_rnorm(sigma2(l), can(2));
      if (cansigma2>minSigma && cansigma2<maxSigma) {
        canq = makeQMNA(mu1(l), mu2(l), sigma1(l), cansigma2, tau);
        if (canq>0 && canq<1) {
          mhRate = 0.0;
          numSelect = 0;
          for (int i=0; i<n; i++) {
            if (g(i) == l) {
              if (h(i) == 0) {
                indSelect(numSelect) = i;
                numSelect++;
              }
              mhRate += h(i) * log(canq) + (1.0-h(i)) * log(1.0-canq) - h(i) * log(q(l)) - (1.0-h(i)) * log(1.0-q(l));
            }
          }
          if (numSelect > 0) {
            for (int j=0; j<numSelect; j++) {
              mhRate += ::Rf_dnorm4(r(indSelect(j)), 0, cansigma2, 1) - ::Rf_dnorm4(r(indSelect(j)), 0, sigma2(l), 1);
            }
          }
          if (::Rf_runif(0,1) < exp(mhRate)) {
            if (probs(l) > 0.2) {
              acc(2)++;
            }
            q(l) = canq;
            sigma2(l) = cansigma2;
          }
          else {
            rejRateSigma2++;
          }
        }
      }
    }
    //}Rcout<<"Time for updating mu1, mu2, sigma1 and sigma2 is "<<(std::clock() - start) <<"/"<< CLOCKS_PER_SEC;

    //update lambda
    sss = 0.0;
    for (int i=0; i<nF; i++) {
      sss += std::abs(mu1(i)) * ((mu1(i)<0)? 1.0-tau:tau) + std::abs(mu2(i)) * ((mu2(i)<0)? 1.0-tau:tau);
    }
    lambda = ::Rf_rgamma(2.0*nF+0.01, 1.0/(sss+0.01));

    //start = std::clock();
    //for (int timer=0; timer<25000; timer++){
    //update v
    IntegerVector clusterSize(nF, 0);
    IntegerVector largerCount(nF, 0);
    for (int i=0; i<n; i++) {
      clusterSize(g(i))++;
      for (int j=0; j<g(i); j++) {
        largerCount(j)++;
      }
    }

    for (int i=0; i<nF; i++) {
      if (clusterSize(i) > 0) {
        keepNCluster(sim)++;
      }
      if (clusterSize(i) > 0.1 * n) {
        keepNClusterT(sim)++;
      }
    }
    //test
    /*
    for (int i=0; i<nF; i++) {
      Rcout<<clusterSize(i)<<"\t";
    }
    Rcout<<"\n";
    for (int i=0; i<nF; i++) {
      Rcout<<largerCount(i)<<"\t";
    }
    Rcout<<"\n";*/
    //test
    temp = 0.0;
    for (int j=0; j<nF-1; j++) {
      v(j) = ::Rf_rbeta(clusterSize(j)+1.0, largerCount(j)+D);
      temp += log(1.0 - v(j));
    }
    v(nF-1) = 1.0;
    probs = makeProbsMNA(v);



    //update D
    //Rcout<<"update D\n";
    D = ::Rf_rgamma(nF-1.0+aD, 1.0 / (bD-temp));
    //Rcout<<D<<"\n";
    if (D != D) {
      D = 1.0;
    }
    if (D < 0.001) {
      D = 0.001;
    }
    //}Rcout<<"Time for updating v and D is "<<(std::clock() - start) <<"/"<< CLOCKS_PER_SEC;


    //test
    /*
    Rcout<<"v\n";
    for (int i=0; i<nF; i++) {
      Rcout<<v(i)<<"\t";
    }
    Rcout<<"\n";
    Rcout<<"probs\n";
    for (int i=0; i<nF; i++) {
      Rcout<<probs(i)<<"\t";
    }
    Rcout<<"\n";
    Rcout<<"g\n";
    for (int i=0; i<n; i++) {
      Rcout<<g(i)<<"\t";
    }
    Rcout<<"\n";
    Rcout<<"h\n";
    for (int i=0; i<n; i++) {
      Rcout<<h(i)<<"\t";
    }
    Rcout<<"\n";
    Rcout<<"q\n";
    for (int i=0; i<nF; i++) {
      Rcout<<q(i)<<"\t";
    }
    Rcout<<"\n";
    Rcout<<"mu1\n";
    for (int i=0; i<nF; i++) {
      Rcout<<mu1(i)<<"\t";
    }
    Rcout<<"\n";
    Rcout<<"mu2\n";
    for (int i=0; i<nF; i++) {
      Rcout<<mu2(i)<<"\t";
    }
    Rcout<<"\n";
    Rcout<<"sig1\n";
    for (int i=0; i<nF; i++) {
      Rcout<<sigma1(i)<<"\t";
    }
    Rcout<<"\n";
    Rcout<<"sig2\n";
    for (int i=0; i<nF; i++) {
      Rcout<<sigma2(i)<<"\t";
    }
    Rcout<<"\n";*/
    //test

    //keep track of stuff
    //Rcout<<"keep track of stuff\n";
    //Rcout<<sim<<"\n";
    keepBeta.row(sim) = as<NumericVector>(wrap(beta));
    keepMu1.row(sim) = mu1;
    keepMu2.row(sim) = mu2;
    keepSigma1.row(sim) = sigma1;
    keepSigma2.row(sim) = sigma2;
    //Rcout<<"keep track of stuff0\n";
    truncProb(sim) = probs(nF-1);
    //Rcout<<"keep track of stuff1\n";
    for (int j=0; j<6; j++) {
      if (att(j) > 200 && 2*sim < burn) {
        can(j) = can(j) * ((acc(j)/att(j)<0.2)?0.5:1.0) * ((acc(j)/att(j)>0.7)?1.5:1.0);
        acc(j) = 1.0;
        att(j) = 1.0;
      }
    }
    //Rcout<<"keep track of stuff2\n";
    //start = std::clock();
    //Rcout<<"update 4\n";
    //for (int timer=0; timer<25000; timer++){
    if (plotDensity) {
      if (sim > burn) {
        //for (int i=0; i<4001; i++) {
        //  ddd(i) = 0.0;
        //}
        for (int j=0; j<nF; j++) {
          if (probs(j) > 0) {
            for (int i=0; i<4001; i++) {
            //  ddd(i) += q(j)*probs(j)*::Rf_dnorm4(xGrid(i), mu1(j), sigma1(j), 0) + (1.0-q(j))*probs(j)*::Rf_dnorm4(xGrid(i), mu2(j), sigma2(j), 0);
            sumdense(i) += (q(j)*probs(j)*::Rf_dnorm4(xGrid(i), mu1(j), sigma1(j), 0) + (1.0-q(j))*probs(j)*::Rf_dnorm4(xGrid(i), mu2(j), sigma2(j), 0))/(nsim-burn);  //ddd(i)/(nsim-burn);
            //sumdense(i) += (q(j)*probs(j)*dNorm(xGrid(i), mu1(j), sigma1(j)) + (1.0-q(j))*probs(j)*dNorm(xGrid(i), mu2(j), sigma2(j)))/(nsim-burn);  //ddd(i)/(nsim-burn);
            //  sumdense2(i) += ddd(i)*ddd(i)/(nsim-burn) - ddd(i)*ddd(i)/pow(nsim-burn, 2.0);
            }
          }
        }
      }
    }
    //}Rcout<<"Time for updating sumdense is "<<(std::clock() - start) <<"/"<< CLOCKS_PER_SEC;

  }
  rejRateMu1 /= nsim*10;
  rejRateMu2 /= nsim*10;
  rejRateSigma1 /= nsim*10;
  rejRateSigma2 /= nsim*10;
  //Rcout<<"keep track of stuff3\n";
  return List::create(Rcpp::Named("nClusterT")=keepNClusterT, Rcpp::Named("nCluster")=keepNCluster, Rcpp::Named("rejRateSigma2") = rejRateSigma2, Rcpp::Named("rejRateSigma1") = rejRateSigma1, Rcpp::Named("rejRateMu2") = rejRateMu2, Rcpp::Named("rejRateMu1") = rejRateMu1, Rcpp::Named("beta") = keepBeta, Rcpp::Named("truncProb")=truncProb, Rcpp::Named("xGrid")=xGrid, Rcpp::Named("dense.mean")=sumdense, Rcpp::Named("dense.var")=sumdense2, Rcpp::Named("xGrid")=xGrid, Rcpp::Named("maxSigma")=maxSigma, Rcpp::Named("q")=q, Rcpp::Named("mu1")=keepMu1, Rcpp::Named("mu2")=keepMu2, Rcpp::Named("sigma1")=keepSigma1, Rcpp::Named("sigma2")=keepSigma2, Rcpp::Named("probs")=probs, Rcpp::Named("v")=v);
}
