### Loading gene expression data of the Golub data set
library(multtest)
data ( golub )

### Perform two sampled student t-tests for all genes comparing the distributions for ALL and AML
two_sample_t_test <- function(row) {
  x<-row[golub.cl==0]
  y<-row[golub.cl==1]
  t.test(x,y) #Welch's test
  #t.test(x,y)$p.value #return only p value
}
golub_t<-apply(golub,1,two_sample_t_test)

### Obtain the top 10 genes with the lowest p-values
two_sample_t_test_pvalue <- function(row) {
  x<-row[golub.cl==0]
  y<-row[golub.cl==1]
  t.test(x,y)$p.value #return only p value
}
golub_t_pvalue<-apply(golub,1,two_sample_t_test_pvalue)
print(golub.gnames[order(golub_t_pvalue)[1:10],2])
# These 10 genes have lowest p-value in t-test, which means that it is highly unlikely that they have the same expression level in AML and ALL groups

### Use Q-Q plot to determine which of the 10 genes obtained above follow normal distribution
# par(mfrow=c(4,3))
# par(mar = c(3,3,3,3))
# https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/par.html
par(cex.lab=0.8)

for (i in order(golub_t_pvalue)[1:10]){
  y<-golub[i,]
  qqnorm(y,main = golub.gnames[i,2],cex.main=0.8,mgp = c(1.5, 0.5, 0)); qqline(y, col = 2)
}
# all of the 10 genes except "CST3 Cystatin C (amyloid angiopathy and cerebral hemorrhage)" more of less follow the assumption of normally distributed

### Now, relax the assumption of equal variance for ALL and AML classes, then determine the top 10 genes with the lowest p-values
equal_var_two_s_ttest_pvalue <- function(row) {
  x<-row[golub.cl==0]
  y<-row[golub.cl==1]
  t.test(x,y,var.equal = TRUE)$p.value #assume the variances are equal
}
golub_t_pvalue_equal_var<-apply(golub,1,equal_var_two_s_ttest_pvalue)
print(golub.gnames[order(golub_t_pvalue_equal_var)[1:10],2])

### Using Shapiro-Wilk test to identify the top 100 genes which deviate significantly from normal distribution
shapiro_test_pvalue<- function(x) {
  shapiro.test(x)$p.value #return only p value
}
golub_sha_pvalue<-apply(golub,1,shapiro_test_pvalue)
print(golub.gnames[order(golub_sha_pvalue)[1:100],2])

### Out of the top 100 genes, obtain 10 most differentiating genes between ALL and AML classes
indices_100_norm<-order(golub_sha_pvalue)[1:100] #the indices of the top 100 genes which deviate significantly from normal
wilcox_pvalue <- function(row) {
  x<-row[golub.cl==0]
  y<-row[golub.cl==1]
  wilcox.test(x,y)$p.value # independent 2-group Mann-Whitney U Test: nonparametric
}
nonnorm_wilcox_pvalue<-apply(golub[indices_100_norm,],1,wilcox_pvalue)
nonnorm_wilcox_pvalue_names<-golub.gnames[indices_100_norm,]
nonnorm_wilcox_pvalue_names[order(nonnorm_wilcox_pvalue)[1:10],2]

