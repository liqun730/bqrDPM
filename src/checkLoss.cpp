#include <Rcpp.h>
using namespace Rcpp;

//' The Check Loss Function
//'
//' Check loss function for frequentist quantile regression.
//'
//' @param y Vector of observed response.
//' @param yHat Vector of fitted response.
//' @param tau Quantile of interest.
//' @return check loss between fitted and observed responses.
//' @export
//'
// [[Rcpp::export]]
double checkLoss(NumericVector y, NumericVector yHat, double tau) {
   double loss = 0.0;
   for (int i=0; i<y.size(); i++) {
     if (y(i) < yHat(i)) {
       loss += (tau - 1.0) * (y(i) - yHat(i));
     }
     else {
       loss += tau * (y(i) - yHat(i));
     }
   }
   return loss;
}
