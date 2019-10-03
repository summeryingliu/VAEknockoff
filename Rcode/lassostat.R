max_lambda<- function(XX, X_k, y, intercept=F,standardize=T, family="gaussian",verbose1=0,lambda=c(2^(-15:-5),1:500/500,55:750/50,16:1000),...) {
  
 # Standardize the variables
  X=cbind(XX,X_k)
  n = nrow(X); p = ncol(XX)
  X=as.matrix(X)
  if( standardize ){ X = scale(X);lambda=lambda/n }
  
 # if (!methods::hasArg(lambda) ) {
      
      # Unless a lambda sequence is provided by the user, generate it
      
  #    lambda_max = max(abs(t(X) %*% y)) / n
      
  #    lambda_min = lambda_max / 3e3
      
  #    k = (0:(nlambda-1)) / nlambda
      
  #    lambda = lambda_max * (lambda_min/lambda_max)^k
      
  #  }

  
  fit <- glmnet::glmnet(X, y, lambda=lambda, intercept=intercept, 
                        
                        standardize=F, standardize.response=F,family=family,...)
  
  first_nonzero <- function(x) match(T, abs(x) > 0) # NA if all(x==0)
  indices <- apply(fit$beta, 1, first_nonzero)
  names(indices)=NULL
  Z= ifelse(is.na(indices), 0, fit$lambda[indices] * n)
  orig = 1:p
  s=sign(Z[orig] - Z[orig+p])
  #print(sum(s==0))
  #if (verbose1==1){
  #print(which(s==0))
  #print(c('Wrong',Z[orig][which(s==0)]))}
  s[s==0]=2*rbinom(sum(s==0),1,0.5)-1
  pmax(Z[orig], Z[orig+p]) * s
  
}


threshold<-function (W, fdr = 0.1, offset = 1) 
{
  if (offset != 1 && offset != 0) {
    stop("Input offset must be either 0 or 1")
  }
  ts1 = sort(c(0, abs(W)))
  ratio = sapply(ts1, function(t) (offset + sum(W <= -t))/max(1, 
                                                             sum(W >= t)))
  ok = which(ratio <= fdr)
  ifelse(length(ok) > 0, ts1[ok[1]], Inf)
}
