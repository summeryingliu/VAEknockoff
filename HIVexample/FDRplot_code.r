FDR_and_power_plot <- function(FDRmat, powermat, names, xname='', mycex=1.5,cols, pchs, alpha,fdr=0.1, alpha_display_div=1, title='',leg_coords=integer(0),show_legend=FALSE){

  alpha_display=alpha[seq(1,length(alpha),by=alpha_display_div)]

  if(length(leg_coords)==0){leg_x=min(alpha);leg_y=1}else{leg_x=leg_coords[1];leg_y=leg_coords[2]}

  xzero=min(alpha)-.05*(max(alpha)-min(alpha))

  xzero1=min(alpha)-.17*(max(alpha)-min(alpha))

  xzero2=min(alpha)-.07*(max(alpha)-min(alpha))

  xzero3=min(alpha)-.1*(max(alpha)-min(alpha))

  plot(0:1,0:1,type='n',xlab=xname,ylab='',xlim=c(xzero1,max(alpha)),ylim=c(-.55,1.05),axes=FALSE,main=title,cex.main=1.2*mycex,cex.lab=mycex)

  segments(xzero2,0,max(alpha),0)

  axis(side=1,at=alpha_display,cex.axis=mycex)

  segments(xzero,-.55,xzero,-.05)

  segments(xzero,.05,xzero,1.05)

  for(i in 0:5){

  segments(xzero,-.55+i/10,xzero2,-.55+i/10)

  text(xzero3,-.55+i/10,i/10,cex=mycex)

  }

  for(i in 0:10){

  segments(xzero,.05+i/10,xzero2,0.05+i/10)

  text(xzero3,.05+i/10,i/10,cex=mycex)

  }

  text(xzero1,-.3,'FDR',srt=90,cex=mycex)

  text(xzero1,.55,'Power',srt=90,cex=mycex)

  for(i in 1:length(names)){

  points(alpha,FDRmat[,i]-.55,type='l',col=cols[i])

  points(alpha,powermat[,i]+.05,type='l',col=cols[i])

  points(alpha,FDRmat[,i]-.55,pch=pchs[i],col=cols[i],cex=mycex)

  points(alpha,powermat[,i]+.05,pch=pchs[i],col=cols[i],cex=mycex)

  }

  points(alpha,rep(-.35+fdr-0.2,length(alpha)),type='l',lty='dotted',col='gray50',lwd=2)

  if(show_legend){

  legend(leg_x,leg_y,legend=names,col=cols,pch=pchs,lty='solid',cex=mycex,bty="n")

  }

  }
