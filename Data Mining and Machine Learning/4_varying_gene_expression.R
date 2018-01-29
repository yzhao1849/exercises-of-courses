# Loading gene expression data of the Golub data set
library(multtest)
data ( golub )

### Calculate the sample mean of each gene
gene_mean_expressions<-rowMeans(golub,na.rm=TRUE) #na.rm=TRUE omits NaN
global_mean<-mean(gene_mean_expressions)
print("global_mean:")
print(global_mean)

### Obtain the 100 most varying genes according to t-statistic
se<-function(x) sd(x)/sqrt(length(x)) #define the function of standard error
SE<-apply(golub,1,se) #standard error
#print(SE)
t_statistic<-(gene_mean_expressions-global_mean)/SE
abs_t_statistic<-abs(t_statistic) #get the absolute values

print("The gene names for the 100 most varying genes:")
print(golub.gnames[order(-abs_t_statistic)[1:100],2])
#get the gene names for the 100 most varying genes

### Get the genes which varys the most between ALL and AML groups (according to Fisher's statistic), out of the 100 genes selected 
# > sum(golub.cl==0)
# [1] 27
# > sum(golub.cl==1)
# [1] 11
indices_t_100<-order(-abs_t_statistic)[1:100] #variable to store the indices of the top 100 most varying genes
golub_100_genes=golub[indices_t_100,] #get the expression values only for the top 100 most varying genes
ALL_cases <- golub_100_genes[,golub.cl==0]
AML_cases <- golub_100_genes[,golub.cl==1]
# > data.class(golub)
# [1] "matrix"
gene_mean_ALL<-rowMeans(ALL_cases,na.rm=TRUE) #na.rm=TRUE: remove the NaN values
gene_mean_AML<-rowMeans(AML_cases,na.rm=TRUE)
SE_100<-SE[indices_t_100]
F_stat<-(gene_mean_ALL-gene_mean_AML)/SE_100
#print("F statistics:")
#print(F_stat)
golub_100_names=golub.gnames[indices_t_100,]
abs_F_stat<-abs(F_stat) #get the absolute values
print("The gene names for the top 10 genes which show the most variation between ALL and AML:")
print(golub_100_names[order(-abs_F_stat)[1:10],2]) 
print("The F scores of these genes:")
print(abs_F_stat[order(-abs_F_stat)[1:10]])
#get the gene names for the top 10 genes which show the most variation between ALL and AML (absolute value)

# get the corresponding index in the orginal matrix
# > match(c("5642","532","1239", "3183", "2300", "5708", "2597", "6388", "1674", "5579"),golub.gnames[,1])
# [1] 2438  253  560 1389 1014 2466 1145 2753  735 2416