// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <string>
#include <iostream>
#include <fstream>
//#include <Rcpp.h>

using namespace Rcpp;
using namespace arma;
using namespace std;

//' Sample Quantile Function
//'
//' Compute the specified sample quantile of a given sample.
//'
//' @param array Vector of sample points.
//' @param p Quantile of interest.
//' @return the specified sample quantile of the given sample.
//' @export
//'
//[[Rcpp::export]]
double sampleQuantile(NumericVector array, double p) {
  NumericVector data = clone(array);
  int n = (int)(data.size()*p);
  std::nth_element(data.begin(), data.begin()+n, data.end());
  double result = data(n);
  std::nth_element(data.begin(), data.begin()+n-1, data.end());
  result = (data(n) + data(n-1))/2;
  return result;
}



