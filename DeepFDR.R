library(knockoff)
#source('C:/Users/Ying Liu/PycharmProjects/BNN/HIV/lassostat.R')
DeepFDR<-function(drugname='PI',rho,ts,fam='gaussian',fdr=0.2,standardize=1,verbose=0,eps=0,lambda=c(2^(-15:-5),1:500/500,55:750/50,16:1000)){
  FDR=numeric(100)
  FDRrina=numeric(100)
  TP=numeric(100)
  TPr=numeric(100)
  TPg=numeric(100)
  FDRg=numeric(100)
  Xknock<-read.csv('PIXknock.csv',header=FALSE)
  X<-read.csv('PIX.csv')
  X=as.matrix(X)
  Xknock=as.matrix(Xknock)
  
  for (i in 1:99){
    #load data
    set.seed(2*i+1) 
    truth=sample(dim(X)[2],size=ts)
    coef=rep(c(1,-1),floor(ts/2))
    if (ts%%2==1){coef=c(coef,1)} 
    m=dim(X)[1]
    #generate data
    if(fam=='gaussian'){y=as.matrix(X[,truth])%*%coef*rho+rnorm(m)}
    if(fam=='binomial'){eta=as.matrix(X[,truth])%*%coef*rho
    y=rbinom(m,1,1/(1+exp(-eta)))}
    if(fam=='cox'){ eta = as.matrix(X[,truth])%*%coef*rho
    T=rexp(m,rate=1*exp(eta))
    C = 1000
    time = pmin(T,C)  
    status = (T<C) 
    y=cbind(time,status)}
    
    #standardize 
    if(standardize==1){
      v=diag(sqrt(1/colSums(X^2)))
      X=X%*%v
      Xknock=Xknock%*%v}
    
    #generate y
    
    
    #variable selection
    #W=max_lambda(X, Xknock,y,family=fam)
    W=max_lambda(X, Xknock,y,family=fam,lambda=lambda)
    thred=threshold(W,fdr=fdr,offset=1)
    select=which(W>=thred+eps)
    
    Xr=create.fixed(X)$Xk
    Wr=max_lambda(X, Xr, y,family=fam,lambda=lambda)
    thredr=threshold(Wr,fdr=fdr,offset=1)
    selectr=which(Wr>=thredr+eps)
    
    Xg=create.second_order(X)
    Wg=max_lambda(X,Xg,y,family=fam,lambda=lambda)
    thredg=threshold(Wg,fdr=fdr,offset=1)
    selectg=which(Wg>=thredg+eps)
    
    if(verbose==1){
      print(c('DL',select))
      print(c('Rina',selectr))
      print(c('Second Order',selectg))}
    
    right=truth%in%select
    rightr=truth%in%selectr
    rightg=truth%in%selectg
    
    FDR[i+1]=(length(select)-sum(right))/max(length(select),1)
    FDRrina[i+1]=(length(selectr)-sum(rightr))/max(length(selectr),1)
    FDRg[i+1]=(length(selectg)-sum(rightg))/max(length(selectg),1)
    
    TP[i+1]=sum(right)/ts
    TPr[i+1]=sum(rightr)/ts
    TPg[i+1]=sum(rightg)/ts}
  
  coln=c('Deep','Fixed','Second-Order')
  Pow=cbind(TP,TPr,TPg)
  colnames(Pow)=coln
  
  FDRrate=cbind(FDR,FDRrina,FDRg)
  colnames(FDRrate)=coln
  
  M=cbind(FDRrate,Pow)
  name=paste(fam,rho,'ts',ts,'fdr',fdr,'.csv',sep='')
  write.csv(M,name)
  return(M)
}
