
ts=50
rhoc=c(3,5,10)
source('C:/Users/Ying Liu/PycharmProjects/latentoutput/HIV/DeepFDR.R')
setwd('C:/Users/Ying Liu/PycharmProjects/latentoutput/HIV/')
for (i in 1:length(rhoc)){
  lambda=c(2^(-15:-5),1:500/500,55:750/50,31:400/2,201:1000)
  rho=rhoc[i]
  print(rho)
  #name=paste('resultrho',rho,'ts',ts,'.csv',sep='')
  #if (file.exists(name)){A <- read.csv(name)[,-1]}
  #if (!file.exists(name)){A<-DeepFDR(D1,rho,ts=20,verbose=1)}
  A<-DeepFDR('PI',rho,ts=ts,verbose=1,lambda=lambda,fdr=0.2)
  print(colMeans(A))
}

rhoc=c(10,20,30)
source('C:/Users/Ying Liu/PycharmProjects/latentoutput/HIV/DeepFDR.R')
setwd('C:/Users/Ying Liu/PycharmProjects/latentoutput/HIV/')
for (i in 1:length(rhoc)){
  lambda=c(2^(-15:-5),1:500/500,55:750/50,31:400/2,201:1000)
  rho=rhoc[i]
  print(rho)
  A<-DeepFDR('PI',rho,ts=ts,verbose=0,fam='binomial',lambda=lambda,fdr=0.2)
  print(colMeans(A))
}

source('summary.R')
