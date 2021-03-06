% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{bqrDPMMNA}
\alias{bqrDPMMNA}
\title{BQR based on DPM of Mixure of Normal Distributions (what is 'A'?)}
\usage{
bqrDPMMNA(Xr, Yr, tau, nsim, initCoef, burn = 5000L, plotDensity = FALSE,
  nF = 10L, aD = 100, bD = 100, maxSigma = 1/0, precBeta = 1e-06)
}
\arguments{
\item{Xr}{Matrix of covariate.}

\item{Yr}{Vector of response.}

\item{tau}{Quantile of interest.}

\item{nsim}{Number of simulation.}

\item{initCoef}{Initial value (vector) of the model coefficients.}

\item{burn}{Number of burn-in.}

\item{plotDensity}{A boolean variable which decides whether to plot the density estimation or not.}

\item{nF}{The number of terms in the finite approximation.}

\item{aD, bD}{V_k~Beta(1, D) and D~dgamma(aD, bD).}

\item{maxSigma}{The maximum possible value for sigma.}

\item{precBeta}{beta[j] ~ norm(0, 1/sqrt(prec.beta)).}
}
\value{
List of model parameters including the coefficients.
}
\description{
Bayesian quantile regression based on Dirichlet process mixture of mixture of two normal distributions.
}
