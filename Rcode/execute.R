setwd('C:/Users/Ying Liu/PycharmProjects/latent/Rcode/')
source('DeepFDR.R')
source('lassostat.R')
source('FDRplot_code.r')

execute<-function(D1,ts,rhoc,fam='gaussian',lambda=c(2^(-15:-5),1:500/500,55:750/50,31:400/2,201:1000),verbose=0,fdr=0.1,plotonly=0){
  if(plotonly==0)
  {  for (i in 1:length(rhoc)){
    rho=rhoc[i]
    #if (file.exists(name)){A <- read.csv(name)[,-1]}
    #if (!file.exists(name)){A<-DeepFDR(D1,rho,ts=20,verbose=1)}
    A<-DeepFDR(D1,rho,ts=ts,verbose=verbose,fam=fam,lambda=lambda,fdr=fdr)
    print(colMeans(A))
  } }
  FF=matrix(0,length(rhoc),3)
  PP=matrix(0,length(rhoc),3)
  for (i in 1:length(rhoc)){
    rho=rhoc[i]
    name=paste(fam,'lastone',rho,'ts',ts,'fdr',fdr,'.csv',sep='')
    A <- read.csv(name)[,-1]
    FF[i,]=colMeans(A)[1:3]
    PP[i,]=colMeans(A)[4:6]
  }
  name1=paste(fam,'ts',ts,'fdr',fdr,'.png',sep='')
  png(name1)
  FDR_and_power_plot(FF,PP,c('Deep','Fixed','Second-Order'),cols=1:length(rhoc), fdr=fdr,pchs=1:length(rhoc), alpha=rhoc )
  dev.off()
  
}