### Consensus clustering on gene expression data: 80% item re-sampling, 80% gene re-sampling, a maximum number of 6 clusters with a total of 100 re-samplings
library(ConsensusClusterPlus)
library(GSVAdata)
data(gbm_VerhaakEtAl)
mads<-apply(exprs(gbm_eset),1,mad)
top2000genes<-exprs(gbm_eset)[order(-mads)[1:2000],]
#results = ConsensusClusterPlus(top2000genes,maxK=6,reps=40,pItem=0.8,pFeature=0.8,clusterAlg="km",distance="euclidean")
results = ConsensusClusterPlus(top2000genes,maxK=6,reps=100,pItem=0.8,pFeature=0.8,clusterAlg="hc",distance="pearson")
icl = calcICL(results)
# The Item-Consensus Plot:
# Bars??? rectangles are ordered by increasing value from bottom to top.
# The asterisks at the top indicate the consensus cluster for each item.

# NMF clustering on gene expression data
library(NMF)
library(GSVAdata)
data(leukemia)
madl<-apply(exprs(leukemia_eset),1,mad)
top2000genes_l<-exprs(leukemia_eset)[order(-madl)[1:2000],]
result_nmf2<-nmf(top2000genes_l,rank=2,method='brunet',seed='random',nrun=50)
coefmap(result_nmf2, Colv="consensus") # Plots a heatmap of the coefficient matrix of the best fit in object.
consensusmap(result_nmf2)
result_nmf3<-nmf(top2000genes_l,rank=3,method='brunet',seed='random',nrun=50)
coefmap(result_nmf3, Colv="consensus") # Plots a heatmap of the coefficient matrix of the best fit in object.
consensusmap(result_nmf3)