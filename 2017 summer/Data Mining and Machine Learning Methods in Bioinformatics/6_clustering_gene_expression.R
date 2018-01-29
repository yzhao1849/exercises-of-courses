# Load data from the microarray experiment done by Alon et al. (1999)
library(gplots)
library(colonCA)
data(colonCA)

### Part 1: Hierarchical clustering

## For each gene, perform an unpaired t-test to see whether it is differentially expressed between normal and cancer patients
## Take the top 50 genes with lowest p-value
Welch_test_pvalue <- function(row) {
  x<-row[colnames(log_colonCA)=="n"] #normal
  y<-row[colnames(log_colonCA)=="t"] #tumor
  t.test(x,y)$p.value #return only p value of Welch's test (unequal 2 sample t test)
}

mat <- exprs(colonCA) #Extract the expression matrix of sample information using exprs
log_colonCA <- log(mat) #log-transformation
colnames(log_colonCA) <- as.character(colonCA$class)
log_colonCA_t_pvalue<-apply(log_colonCA,1,Welch_test_pvalue)

new_colonCA<-log_colonCA[order(log_colonCA_t_pvalue)[1:50],]
sort(log_colonCA_t_pvalue)[1:50] #show the p-values of the top 50 genes



## Conduct hierarchical clustering, and plot dendrograms
par(mfrow=c(2,2))
dist_mat=dist(t(new_colonCA),method = "euclidean") #distances between the rows (samples:n/t) of a data matrix
hc1 <- hclust(dist_mat,method = "single")
hc2 <- hclust(dist_mat,method = "complete")
hc3 <- hclust(dist_mat,method = "average")
hc4 <- hclust(dist_mat,method = "ward.D")
par(cex=0.6)
plot(hc1,hang=-1) # align the leaves
plot(hc2,hang=-1)
plot(hc3,hang=-1)
plot(hc4,hang=-1)


## Generate a heatmap for the above 50 genes: rows are different genes, columns are different 
#change the column names:
colnames(new_colonCA)[which(colnames(new_colonCA) == "n")] <- "healthy"
colnames(new_colonCA)[which(colnames(new_colonCA) == "t")] <- "cancer"

cc <- colnames(new_colonCA)
cc[cc == "cancer"] <- "#FF7F24" #brown1
cc[cc == "healthy"] <- "#7FFF00" #acquamarine
my_palette <- colorRampPalette(c("blue", "white", "red"))(n = 100) #or simply: bluered(100)
heatmap.2(x=new_colonCA,distfun=function(c) dist(c,method = "euclidean"),
          hclustfun=function(c) hclust(c,method = "ward.D"),scale="row",col=my_palette,
          main = "Heatmap with clustering \nAlon (1999) Data Set", labCol= "", 
          ColSideColors=cc,density.info="density",key.xlab="RowZscores",
          key.ylab="counts",key.title="Color Key",trace="none") #default distfun is already euclidean
#xpd=TRUE to enable things to be drawn outside the plot region
legend("topright",xpd=TRUE,cex = 0.5, c("Cancer", "Normal"),fill=c("#FF7F24","#7FFF00"))

### Part 2: K-means clustering
library(colonCA)
data(colonCA)
mat <- exprs(colonCA) #Extract the expression matrix of sample information using exprs
log_colonCA <- log(mat) #log-transformation
colnames(log_colonCA) <- as.character(colonCA$class)
kmeans_2<-kmeans(t(log_colonCA),2,nstart = 20) #2 clusters. 20 random starts
print(kmeans_2$totss)
print(kmeans_2$tot.withinss)
print(kmeans_2$betweenss)
print(kmeans_2$cluster)
cat("fraction of variation within clusters (tot.withinss) in comparison total variation (totss):",kmeans_2$tot.withinss/kmeans_2$totss)


kmeans_3<-kmeans(t(log_colonCA),3,nstart = 20)
print(kmeans_3$totss) #The total sum of squares
print(kmeans_3$tot.withinss)
print(kmeans_3$betweenss)

kmeans_4<-kmeans(t(log_colonCA),4,nstart = 20)
print(kmeans_4$totss) #The total sum of squares
print(kmeans_4$tot.withinss)
print(kmeans_4$betweenss)

kmeans_5<-kmeans(t(log_colonCA),5,nstart = 20)
print(kmeans_5$totss) #The total sum of squares
print(kmeans_5$tot.withinss)
print(kmeans_5$betweenss)

# https://stats.stackexchange.com/questions/48520/interpreting-result-of-k-means-clustering-in-r
# If you compute the sum of squared distances of each data point to the global sample mean, you get  
# total_SS. If, instead of computing a global sample mean (or 'centroid'), you compute one per group 
# (here, there are three groups) and then compute the sum of squared distances of these three means to 
# the global mean, you get between_SS. (When computing this, you multiply the squared distance of each 
# mean to the global mean by the number of data points it represents.)
##-->so, totss=tot.withinss+betweenss. And the more clusters, the larger betweenss and smaller tot.withinss. But totss is always the same

Welch_test_pvalue <- function(row) {
  x<-row[colnames(log_colonCA)=="n"] #normal
  y<-row[colnames(log_colonCA)=="t"] #tumor
  t.test(x,y)$p.value #return only p value of Welch's test (unequal 2 sample t test)
}

mat <- exprs(colonCA) #Extract the expression matrix of sample information using exprs
log_colonCA <- log(mat) #log-transformation
colnames(log_colonCA) <- as.character(colonCA$class)
log_colonCA_t_pvalue<-apply(log_colonCA,1,Welch_test_pvalue)

new_colonCA<-log_colonCA[order(log_colonCA_t_pvalue)[1:50],]
kmeans_2_new<-kmeans(t(new_colonCA),2,nstart = 20) #2 clusters. 20 random starts
print(kmeans_2_new$totss)
print(kmeans_2_new$tot.withinss)
print(kmeans_2_new$betweenss)
print(kmeans_2_new$cluster)
cat("fraction of variation within clusters (tot.withinss) in comparison total variation (totss):",kmeans_2_new$tot.withinss/kmeans_2_new$totss)

kmeans_3_new<-kmeans(t(new_colonCA),3,nstart = 20) #2 clusters. 20 random starts
print(kmeans_3_new$totss)
print(kmeans_3_new$tot.withinss)
print(kmeans_3_new$betweenss)
print(kmeans_3_new$cluster)
cat("fraction of variation within clusters (tot.withinss) in comparison total variation (totss):",kmeans_3_new$tot.withinss/kmeans_3_new$totss)

kmeans_4_new<-kmeans(t(new_colonCA),4,nstart = 20) #2 clusters. 20 random starts
print(kmeans_4_new$totss)
print(kmeans_4_new$tot.withinss)
print(kmeans_4_new$betweenss)
print(kmeans_4_new$cluster)
cat("fraction of variation within clusters (tot.withinss) in comparison total variation (totss):",kmeans_4_new$tot.withinss/kmeans_4_new$totss)

kmeans_5_new<-kmeans(t(new_colonCA),5,nstart = 20) #2 clusters. 20 random starts
print(kmeans_5_new$totss)
print(kmeans_5_new$tot.withinss)
print(kmeans_5_new$betweenss)
print(kmeans_5_new$cluster)
cat("fraction of variation within clusters (tot.withinss) in comparison total variation (totss):",kmeans_5_new$tot.withinss/kmeans_5_new$totss)
