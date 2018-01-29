
### Perform a PCA
library(ggfortify)
library(colonCA)
data(colonCA)
mat <- exprs(colonCA) #Extract the expression matrix of sample information using exprs
log_colonCA<-t(log(mat)) #log-transformation and then transpose
pca = prcomp(log_colonCA,  scale = TRUE) #Same as pr = prcomp(log_colonCA,center=TRUE, scale = TRUE)
summary(pca)

### 2D PCA plot
#biplot(pr,var.axes = FALSE,ylabs = NULL)
#Reference: https://cran.r-project.org/web/packages/ggfortify/vignettes/plot_pca.html
df_class=data.frame(log_colonCA)
df_class=cbind(pData(colonCA)$class,df_class) #add the class information to each patient row
colnames(df_class)[1]="class"
autoplot(pca,data=df_class,colour="class",shape = FALSE)+scale_colour_discrete(labels=c("Normal", "Tumor"))


#About the diff between R-mode and Q-mode PCA (princomp vs prcomp):
#https://stats.stackexchange.com/questions/20101/what-is-the-difference-between-r-functions-prcomp-and-princomp
#https://stat.ethz.ch/pipermail/r-help/2011-September/289101.html

#diff between PCA(princomp) and SVD(prcomp) 
#https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca


### Scree plot of the eigenvalues
screeplot(pca, npcs=62,type="lines",main="screeplot",cex=0.5)
#axis(1,at = c(1:62), cex.axis = 0.5)

#d
PoV <- pca$sdev^2/sum(pca$sdev^2)
cum_V <- cumsum(PoV)
plot(cum_V,xlab="n",ylab="Percent variance",main="Fraction of variance explained by the first n PCs",cex.lab=1.2,cex=0.3,xaxp=c(1,62,61))
abline(h=0.95,col=2)
abline(v=35,col=2)




#What happens when data matrix has more columns (variables) than rows (observations) when doing PCA
#https://stats.stackexchange.com/questions/28909/pca-when-the-dimensionality-is-greater-than-the-number-of-samples
# If n is the number of points and p is the number of dimensions and n<=p 
# then the number of principal components with non-zero variance cannot exceed n 
# (when doing PCA on raw data) or n???1 (when doing PCA on centered data - as usual).
# **More intuitive explanation: https://stats.stackexchange.com/questions/123318/why-are-there-only-n-1-principal-components-for-n-data-points-if-the-number

# # Information about ExpressionSet object (details see the pdf in the same file)
# featureNames(colonCA)[1:5] #retrieve the names of the features
# colonCA$class[1:100]
# sampleNames(colonCA)[1:100] #unique identifiers of the samples in the data set
# varLabels(colonCA) #lists the column names of the phenotype data (phenoData)
# mat <- exprs(colonCA) #Extract the expression matrix of sample information using exprs
# vv <- colonCA[1:5, 1:3] #Subsetting: Create a new ExpressionSet consisting of the 5 features and the first 3 samples
# males <- colonCA[ , colonCA$class == "t"] #Subsetting: Create a subset consisting of only the male samples