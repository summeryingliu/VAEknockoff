setwd('C:/Users/Ying Liu/PycharmProjects/latent/Rcode/')
source('DeepFDRsep.R')
source('lassostat.R')
source('FDRplot_code.r')

execute<-function(D1,ts,rhoc,fam='gaussian',namek,rep,lambda=c(2^(-15:-5),1:500/500,55:750/50,31:400/2,201:1000),verbose=0,fdr=0.1,plotonly=0){
  if(plotonly==0)
  {  for (i in 1:length(rhoc)){
    rho=rhoc[i]
    #if (file.exists(name)){A <- read.csv(name)[,-1]}
    #if (!file.exists(name)){A<-DeepFDR(D1,rho,ts=20,verbose=1)}
    DeepFDRsep(D1,rho,ts=ts,verbose=verbose,rep=rep,fam=fam,lambda=lambda,fdr=fdr,namek=namek)
    #print(colMeans(A))
  } }
  
}