#this file excute the simulation and summarize 
args=commandArgs(TRUE)
if (length(args)==0){
  stop("Enter the destination folder for Summarize.n",call.=FALSE)
}  else {D1=args[1]
        for (i in 2:length(args)){
          eval(parse(text=args[[i]]))
        }
}

source('C:/Users/Ying Liu/PycharmProjects/latent/Rcode/executesep.R')
if (!exists('ts')){ts=20}
if (!exists('fam')){fam='gaussian'}
if (!exists('rhoc')){rhoc=c(0.5,1,2,5)}
if (!exists('lambda')){
  if (fam=='gaussian') lambda=c(2^(-15:-5),1:500/500,55:750/50,31:400/2,201:1000)
  if (fam=='binomial') lambda=c(2^(-30:-10),1:2000/2000,21:1000/10,11:50)}
if (!exists('fdr')){
  if (fam=='gaussian') fdr=0.1
  if (fam=='binomial') fdr=0.2
}
if (!exists('rep')){rep=500}
execute(D1,ts,rhoc,fam=fam,lambda=lambda,fdr=fdr,namek=namek,rep=rep)



# execute("C:/Users/Ying\ Liu/PycharmProjects/latentoutput/cont",20,c(1,2,5,10),plotonly=1)
# execute("C:/Users/Ying\ Liu/PycharmProjects/latentoutput/cont0",20,c(1,2,5,10))
# execute("C:/Users/Ying\ Liu/PycharmProjects/latentoutput/cont0h",20,c(1,2,5,10))
# execute("C:/Users/Ying\ Liu/PycharmProjects/latentoutput/cont050",20,c(1,2,5,10))
# execute("C:/Users/Ying\ Liu/PycharmProjects/latentoutput/cont050_10",20,c(1,2,5,10))
# execute("C:/Users/Ying\ Liu/PycharmProjects/latentoutput/cont050_10h",20,c(1,2,5,10))
# 
# 
# execute("C:/Users/Ying\ Liu/PycharmProjects/latentout1000/cont0",20,c(1,2,5,10))
# execute("C:/Users/Ying\ Liu/PycharmProjects/latentout1000/cont50_10",20,c(0.2,0.5,1,2))
# execute("C:/Users/Ying\ Liu/PycharmProjects/latentout1000/cont50_50epoch",20,c(0.2,0.5,1,2))
# 
# execute("C:/Users/Ying\ Liu/PycharmProjects/latentout1000/setting2",20,c(0.2,0.5,1,2))
# execute("C:/Users/Ying\ Liu/PycharmProjects/latentout1000/setting2_50_10",20,c(0.5,1,2,5))
# execute("C:/Users/Ying\ Liu/PycharmProjects/latentoutput/setting2_150_200",20,c(0.5,1,2,5))
# execute("C:/Users/Ying\ Liu/PycharmProjects/latentoutput/setting2_200",20,c(0.5,1,2,5))
# 
# 
# execute("C:/Users/Ying\ Liu/PycharmProjects/latentoutput/cont",20,c(2,5,10),plotonly=1,lambda=c(2^(-30:-10),1:2000/2000,21:1000/10,11:50),fdr=0.2,fam='binomial')
# execute("C:/Users/Ying\ Liu/PycharmProjects/latentoutput/cont0h",20,c(2,5,10),lambda=c(2^(-30:-10),1:2000/2000,21:1000/10,11:50),fdr=0.2,fam='binomial')
# execute("C:/Users/Ying\ Liu/PycharmProjects/latentoutput/cont050_10h",20,c(2,5,10),lambda=c(2^(-30:-10),1:2000/2000,21:1000/10,11:50),fdr=0.2,fam='binomial')
# 




