/* Hierarchy model */
data {
  int<lower=1> C;       // number of counties
  int<lower=1> A;       // number of ages
  int<lower=1> R;       // number of races
  int<lower=1> N;       // number of observations
  vector[N] y;    // dependent variable
  int<lower=1,upper=C> county[N]; //the county where the respondents worked
  int<lower=1,upper=A> age[N]; //the county where the respondents worked
  int<lower=1,upper=R> race[N]; //the county where the respondents worked
  int<lower=0,upper=1> gender[N];    // the gender of the respondents
  vector[N] education;  // the education level of the respondents.
  vector[N] uhrswork;  //usual hours the respondent worked per week
  vector[N] occscore;  // occupation income score
}
parameters {
  real beta0;
  real beta1;
  real beta2;
  real beta3;
  real beta4;
  vector[C] eta_c;         // county mean
  vector[A] eta_a;         // age mean
  vector[R] eta_r;         // race mean
  real<lower=0> sigma_y;       //sd of y
  real<lower=0> sigma_c;          // sd of countries
  real<lower=0> sigma_a;          // sd of ages
  real<lower=0> sigma_r;          // sd of races
}
model {
  vector[N] y_hat;
  
  for (i in 1:N){
    y_hat[i] = eta_c[county[i]]+ eta_a[age[i]]+ eta_r[race[i]]+beta0+beta1*gender[i]+beta2*education[i]+beta3*uhrswork[i]+beta4*occscore[i];
  }
  
  beta0 ~ normal(0,1);
  beta1 ~ normal(0,1);
  beta2 ~ normal(0,1);
  beta3 ~ normal(0,1);
  beta4 ~ normal(0,1);
  eta_c ~ normal(0,sigma_c);
  eta_a ~ normal(0,sigma_a);
  eta_r ~ normal(0,sigma_r);
  sigma_y ~ normal(0,1);
  sigma_c ~ normal(0,1);
  sigma_a ~ normal(0,1);
  sigma_r ~ normal(0,1);

  
  // likelihood
  y ~normal(y_hat,sigma_y);
}
generated quantities {
  vector[N] log_lik;    // pointwise log-likelihood for LOO
  vector[N] y_rep; // replications from posterior predictive dist

  for (i in 1:N) {
    real y_hat_i =  eta_c[county[i]]+ eta_a[age[i]]+ eta_r[race[i]]+beta0+beta1*gender[i]+beta2*education[i]+beta3*uhrswork[i]+beta4*occscore[i];
    log_lik[i] = normal_lpdf( y[i] | y_hat_i,sigma_y);
    y_rep[i] = normal_rng(y_hat_i,sigma_y);
  }
}
