library(knockoff)

DeepFDRsep<-function(D,rho,ts,namek=c('Xknock'),fam='gaussian',fdr=0.1,rep=50,standardize=1,verbose=0,eps=0,lambda=c(2^(-15:-5),1:500/500,55:750/50,16:1000)){
  setwd(D) 
  knocknum=length(namek)
  name=paste('FDR',fam,rho,'ts',ts,'fdr',fdr,'.csv',sep='')
  name2=paste('TP',fam,rho,'ts',ts,'fdr',fdr,'.csv',sep='')
  fixed=!file.exists(name)
  FDR=matrix(0,rep,knocknum+2*(fixed==1))
  TP=matrix(0,rep,knocknum+2*(fixed==1))

  for (i in 0:(rep-1)){
    #load data
    X<-as.matrix(read.csv(paste('X',i,'.csv',sep=''),header=FALSE))

    set.seed(2*i+1) 
    coef=rep(c(1,-1),floor(ts/2))
    if (ts%%2==1){coef=c(coef,1)} 
    m=dim(X)[1]
    p=dim(X)[2]
    #generate data
    if(fam=='gaussian'){y=as.matrix(X[,1:ts])%*%coef*rho+rnorm(m)}
    if(fam=='binomial'){eta=as.matrix(X[,1:ts])%*%coef*rho
    y=rbinom(m,1,1/(1+exp(-eta)))}
    if(fam=='cox'){ eta = as.matrix(X[,1:ts])%*%coef*rho
    T=rexp(m,rate=1*exp(eta))
    C = 1000
    time = pmin(T,C)  
    status = (T<C) 
    y=cbind(time,status)}
    
    
    for (j in 1:knocknum){
      Xknock<-as.matrix(read.csv(paste(namek[j],i,'.csv',sep=''),header=FALSE))
    #standardize 
      if(standardize==1){
        v=diag(sqrt(1/colSums(X^2)))
        X=X%*%v
        print(dim(Xknock))
        Xknock=Xknock%*%v}
      #variable selection
      #W=max_lambda(X, Xknock,y,family=fam)
      W=max_lambda(X, Xknock,y,family=fam,lambda=lambda,standardize =1)
      thred=threshold(W,fdr=fdr,offset=1)
      select=which(W>=thred+eps)
      right=c(1:ts)%in%select
      
      FDR[i+1,j]=(length(select)-sum(right))/max(length(select),1)
      TP[i+1,j]=sum(right)/ts
    }
    
    if (fixed){
    set.seed(i)  
    if(m>=2*p){
    Xr=create.fixed(X)$Xk}
    if (m<2*p){
      Xr=create.fixed(X,y=y)$Xk[1:m,]
    }
    Wr=max_lambda(X, Xr, y,family=fam,lambda=lambda)
    thredr=threshold(Wr,fdr=fdr,offset=1)
    selectr=which(Wr>=thredr+eps)
    
    set.seed(i) 
    Xg=create.second_order(X)
    Wg=max_lambda(X,Xg,y,family=fam,lambda=lambda)
    thredg=threshold(Wg,fdr=fdr,offset=1)
    selectg=which(Wg>=thredg+eps)
    
    rightr=c(1:ts)%in%selectr
    rightg=c(1:ts)%in%selectg
    
    FDR[i+1,j+1]=(length(selectr)-sum(rightr))/max(length(selectr),1)
    FDR[i+1,j+2]=(length(selectg)-sum(rightg))/max(length(selectg),1)
    
    TP[i+1,j+1]=sum(rightr)/ts
    TP[i+1,j+2]=sum(rightg)/ts
    }
  }
  
  
  rown=namek
  if(fixed){rown=c(namek,'Fixed','Second-Order')}
  colnames(TP)=rown
  colnames(FDR)=rown
  
  if(fixed){
  write.table(t(FDR),name,col.names =FALSE,sep=',')
  write.table(t(TP),name2,col.names =FALSE,sep=',')}
  
  if(!fixed){
    write.table(t(FDR),name,append=TRUE,col.names = FALSE,sep=',')
    write.table(t(TP),name2,append=TRUE,col.names = FALSE,sep=',')
  }
}