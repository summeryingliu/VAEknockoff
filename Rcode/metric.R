#compute diagnostic metrics
cormet<-function(X,Xk){
  R1<-cor(X)
  R2<-cor(X,Xk)
  S<-abs(R1-R2)
  offdiag<-mean(S[row(S)!=col(S)])
  diag<-mean(S[row(S)==col(S)])
  return (list(offdiag,diag))
}
D1="C:/Users/Ying Liu/PycharmProjects/latentout/setting3"
plot<-function(D1,namek=c('X','Xka','Xkb','Xkc')){
  setwed(D1)
  m=length(namek)
  Off<-Diag<-matrix(0,100,m-1)
  for(i in 0:99){
    X<-as.matrix(read.csv(paste('X',i,'.csv',sep=''),header=FALSE))
    for (j in 2:m){ 
      Xk<-as.matrix(read.csv(paste(namek[j],i,'.csv',sep=''),header=FALSE))
      s<-cormet(X,Xk)
      Off[i+1,j-1]<-s[[1]]
      Diag[i+1,j-1]<-s[[2]]
    }
  }
  png(metric.png)
  boxplot(Off)
  boxplot(Diag)
  dev.off()
  
}