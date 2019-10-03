batchplot<-function(rhoc,fdrc=c(0.1,0.2),famc=c('gaussian','binomial'),tsc=c(10,20),m=6,legend=1){
FF=matrix(0,length(rhoc),m)
FSE=matrix(0,length(rhoc),m)
PP=matrix(0,length(rhoc),m)
PSE=matrix(0,length(rhoc),m)
for (s in 1:2){
  fam=famc[s]
  fdr=fdrc[s]
  for (ts in tsc){
    for (i in 1:length(rhoc)){
      rho=rhoc[i]
      name=paste('FDR',fam,rho,'ts',ts,'fdr',fdr,'.csv',sep='')
      name2=paste('TP',fam,rho,'ts',ts,'fdr',fdr,'.csv',sep='')
      FDR <- t(as.matrix(read.table(name,sep=',',row.names=1)))
      TP<-t(as.matrix(read.table(name2,sep=',',row.names=1)))
      n=dim(FDR)[1]
      FF[i,]=colMeans(FDR)[1:m]
      PP[i,]=colMeans(TP)[1:m]
      FSE[i,]=sqrt(apply(FDR[,1:m],2,var))/sqrt(n)
      PSE[i,]=sqrt(apply(TP[,1:m],2,var))/sqrt(n)[1:m]
      }
    namek=colnames(FDR)

    namep=paste(fam,'ts',ts,'fdr',fdr,'.png',sep='')
    namet=paste(fam,'ts',ts,'fdr',fdr,'.csv',sep='')
    png(namep)
    FDR_and_power_plot(FF,PP,namek,cols=1:length(namek), fdr=fdr,pchs=1:length(namek), alpha=rhoc,show_legend = legend )
    dev.off()
    M1=cbind(FF,PP)
    M2=cbind(FSE,PSE)
    digit=3
    M=matrix(paste(round(M1,digit),'(',round(M2,digit-1),')',sep=''),dim(M1)[1],dim(M1)[2])
    rownames(M)=rhoc
    colnames(M)=c(namek,namek)
    write.csv(M,namet)
}
}
}
source('C:/Users/Ying Liu/PycharmProjects/latent/Rcode/FDRplot_code.R')
setwd('C:/Users/Ying Liu/PycharmProjects/latentoutput/setting1/')
batchplot(c(1,1.5,2,3,5),m=8)
setwd('C:/Users/Ying Liu/PycharmProjects/latentoutput/setting2/')
batchplot(c(0.5,1,2,5,10),m=4)
batchplot(c(0.5,1,2,5,10),tsc=c(30,40),m=4)
setwd('C:/Users/Ying Liu/PycharmProjects/latentoutput/setting3/')
batchplot(c(1,2,5,10),tsc=c(10,40),fdrc=c(0.1,0.1),m=3,legend=1)
batchplot(c(0.5,1,2,5,10),tsc=c(10),fdrc=c(0.1,0.1),m=3,legend=0)
setwd('C:/Users/Ying Liu/PycharmProjects/latentoutput/setting4/')
batchplot(c(1,2,5,10,20),ts=30,m=3)
setwd('C:/Users/Ying Liu/PycharmProjects/latentoutput/setting5/')
batchplot(c(1,1.5,2,3,5))
setwd('C:/Users/Ying Liu/PycharmProjects/latentoutput/setting6/')
batchplot(c(1,2,5,10),tsc=c(10,20),m=4,legend=0)
batchplot(c(1,2,5,10),tsc=c(30,40),m=4,legend=0)
