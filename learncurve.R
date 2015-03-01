setwd("/home/srimugunthan/TookitakiEx")
res <-  read.csv("multidrawc.txt",   header=TRUE, sep="", as.is=TRUE)

# get the range for the x and y axis
xrange <- range(1000:30000)
yrange <- range(0:1)
colors <- rainbow(2)
linetype <- c(1:2)
plotchar <- seq(18,19,1)

pdf(file="multilcuve.pdf")
#Learning curve
plot(xrange,yrange,xlab="Number of training example",ylab="Accuracy")
lines(res$Datasetsize, 1-res$TrainingAccuracy, type="b", lwd=1.5,
      lty=linetype[1], col=colors[1], pch=plotchar[1]) 
lines(res$Datasetsize, 1-res$TestsetAccuracy, type="b", lwd=1.5,
      lty=linetype[2], col=colors[2], pch=plotchar[2]) 
# lines(2:nrow(X), error$error_val[2:nrow(X)], type="b", lwd=1.5,
#       lty=linetype[2], col=colors[2], pch=plotchar[2]) 
# legend(xrange[1], yrange[2], 1:2, cex=0.8, col=colors,
#        pch=plotchar, lty=linetype, title="Linear Regression learing curve")

dev.off()

pdf(file="multiAUCcuve.pdf")
#Learning curve
plot(xrange,yrange,xlab="Number of training example",ylab="AUC")
lines(res$Datasetsize, 1- res$TrainingAUC, type="b", lwd=1.5,
      lty=linetype[1], col=colors[1], pch=plotchar[1]) 
lines(res$Datasetsize, 1- res$TestAUC, type="b", lwd=1.5,
      lty=linetype[2], col=colors[2], pch=plotchar[2]) 
# lines(2:nrow(X), error$error_val[2:nrow(X)], type="b", lwd=1.5,
#       lty=linetype[2], col=colors[2], pch=plotchar[2]) 
# legend(xrange[1], yrange[2], 1:2, cex=0.8, col=colors,
#        pch=plotchar, lty=linetype, title="Linear Regression learing curve")

dev.off()

nnres <-  read.csv("nndrawc.txt",   header=TRUE, sep="", as.is=TRUE)

pdf(file="NNlcuve.pdf")
#Learning curve
plot(xrange,yrange,xlab="Number of training example",ylab="Accuracy")
lines(nnres$DatasetSize, 1-res$TrainingAccuracy, type="b", lwd=1.5,
      lty=linetype[1], col=colors[1], pch=plotchar[1]) 
lines(nnres$DatasetSize, 1-res$TestsetAccuracy, type="b", lwd=1.5,
      lty=linetype[2], col=colors[2], pch=plotchar[2]) 
# lines(2:nrow(X), error$error_val[2:nrow(X)], type="b", lwd=1.5,
#       lty=linetype[2], col=colors[2], pch=plotchar[2]) 
# legend(xrange[1], yrange[2], 1:2, cex=0.8, col=colors,
#        pch=plotchar, lty=linetype, title="Linear Regression learing curve")

dev.off()

pdf(file="NNAUCcuve.pdf")
#Learning curve
plot(xrange,yrange,xlab="Number of training example",ylab="AUC")
lines(nnres$DatasetSize, 1- nnres$TrainingAUC, type="b", lwd=1.5,
      lty=linetype[1], col=colors[1], pch=plotchar[1]) 
lines(nnres$DatasetSize, 1- nnres$TestAUC, type="b", lwd=1.5,
      lty=linetype[2], col=colors[2], pch=plotchar[2]) 
# lines(2:nrow(X), error$error_val[2:nrow(X)], type="b", lwd=1.5,
#       lty=linetype[2], col=colors[2], pch=plotchar[2]) 
# legend(xrange[1], yrange[2], 1:2, cex=0.8, col=colors,
#        pch=plotchar, lty=linetype, title="Linear Regression learing curve")

dev.off()