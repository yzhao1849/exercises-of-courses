"""Principal Component Analysis"""

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

df1=pd.read_excel("breast-cancer-wisconsin.xlsx")
df1=df1.fillna(df1.mean()) #replace the NaN with the mean of the column
pca_var_list=[]

pca = PCA()
pca.fit(df1.drop(["code","class"],axis=1))  #remove the 'code' and 'class' columns when fitting
for i in range(len(pca.explained_variance_ratio_)):
    if i==0:
        pca_var_list.append(pca.explained_variance_ratio_[0])
    else: pca_var_list.append(pca_var_list[i-1]+pca.explained_variance_ratio_[i])

plt.plot([1,2,3,4,5,6,7,8,9], pca_var_list, '-o',color='r',alpha=0.7,markeredgewidth=0.0)
plt.ylabel("Fraction of variance explained")
plt.xlabel("First n components")
plt.axhline(y=0.9, color='c', linestyle='--',alpha=0.8)
ax=plt.gca()
ax.set_ylim(0.65,1)
plt.savefig("Fraction_of_variance.pdf",dpi=100) #save the figure

reduced=pca.transform(df1.drop(["code","class"],axis=1))  #the result is a 699*9 numpy.ndarray
reduced_df1=pd.DataFrame(reduced) #transform the numpy.ndarray to a pandas dataframe
#print(reduced_df1)
reduced_df1["class"]=df1["class"] #add the class information to each sample of the transformed matrix
grouped = reduced_df1.groupby('class') #group by class

plt.figure() #open a new figure
#plot the diagonal cells of the matrix
for i in range(5):
    ax=plt.subplot(5,5,6*i+1) #for matrix diagonal
    x = grouped.get_group(2)[i]  # 2-benign
    y = grouped.get_group(4)[i]  # 4-malignant
    x.plot.kde(label="benign",c='r') #density plot
    y.plot.kde(label="malignant",c='c')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8) #make the y axis fontsize smaller
    plt.subplots_adjust(hspace=0.2)
    plt.subplots_adjust(wspace=0.2)
    if i==0:
        ax.set_ylabel('Density')
    else:
        ax.set_ylabel('')
    ax.set_title("No.{} component".format(i+1),fontsize=10)
    ttl = ax.title
    ttl.set_position([0.5, 1.02])  # set position of the title
    plt.legend(bbox_to_anchor=(1, 1),fontsize=6) #adjust the location of the legend box

for i in range(5):
    for j in range(5):
        if i!=j: #off-diagonal cells
            plt.subplot(5, 5, i*5+j+1)
            x_2 = grouped.get_group(2)[i]
            y_2 = grouped.get_group(2)[j]
            plt.scatter(y_2, x_2, alpha=0.5, c="r",s=5, edgecolors='none') #use square root to make the contrast between the size of the biggest points and smallest points not so big

            x_4 = grouped.get_group(4)[i]
            y_4 = grouped.get_group(4)[j]
            plt.scatter(y_4,x_4, alpha=0.5, c="c",s=5,edgecolors='none')  # use square root to make the contrast between the size of the biggest points and smallest points not so big
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)

            plt.subplots_adjust(hspace=0.2)
            plt.subplots_adjust(wspace=0.2)
            ax=plt.gca()

fig = plt.gcf() #get current figure
fig.set_size_inches(15, 15)
plt.savefig("Assignment4.pdf",dpi=200) #save the figure

id_outlier=abs(reduced_df1[3]-reduced_df1[3].mean()).idxmax() #return the index of the furthest point from the mean
reduced_df1=reduced_df1.drop(reduced_df1.index[id_outlier]) #remove this outlier

def f_score_class(df,attri): #group the dataframe by the benign and malignant class labels and then calculates F for any given attribute (attri)
    grouped = df.groupby('class')
    grand_mean=df[attri].mean()
    mean_2=grouped[attri].mean()[2]
    mean_4=grouped[attri].mean()[4]
    var_2 = (grouped[attri].std()[2]) ** 2
    var_4 = (grouped[attri].std()[4]) ** 2
    fscore = ((mean_2 - grand_mean) ** 2 + (mean_4 - grand_mean) ** 2) / (var_2 + var_4)
    return fscore

plt.figure() #open a new figure

plt.subplot(121) #plot the the original PCA result
plt.plot([1,2,3,4,5,6,7,8,9], pca_var_list, '-o',color='r',alpha=0.7,markeredgewidth=0.0)
plt.ylabel("Fraction of variance explained")
plt.xlabel("First n components")
plt.axhline(y=0.9, color='c', linestyle='--',alpha=0.8)
ax=plt.gca()
ax.set_ylim(0.65,1)

df2=pd.DataFrame() #df2 is df1 adjusted with F scores
for i in df1.columns[1:10]: #skip the 'code' and 'class' columns
    f_score=f_score_class(df1,i)
    df2[i]=df1[i]*f_score
pca.fit(df2)
pca_var_list_2=[]

for i in range(len(pca.explained_variance_ratio_)):
    if i==0:
        pca_var_list_2.append(pca.explained_variance_ratio_[0])
    else: pca_var_list_2.append(pca_var_list_2[i-1]+pca.explained_variance_ratio_[i])
plt.subplot(122) #plot the the F-score adjusted PCA result
plt.plot([1,2,3,4,5,6,7,8,9], pca_var_list_2, '-o',color='r',alpha=0.7,markeredgewidth=0.0)
plt.ylabel("Fraction of variance explained (F score adjusted)")
plt.xlabel("First n components")
plt.axhline(y=0.9, color='c', linestyle='--',alpha=0.8)
ax=plt.gca()
ax.set_ylim(0.65,1)

df3=pd.DataFrame() #df2 is the result of multiplying the 1st column of df1 by 20
for i in df1.columns[1:10]: #copy df1 to df3, skip the 'code' and 'class' columns
    df3[i]=df1[i]
df3[df3.columns[0]]*=20
pca.fit(df3)
pca_var_list_3=[]
for i in range(len(pca.explained_variance_ratio_)):
    if i==0:
        pca_var_list_3.append(pca.explained_variance_ratio_[0])
    else: pca_var_list_3.append(pca_var_list_3[i-1]+pca.explained_variance_ratio_[i])

plt.figure() #open a new figure

plt.subplot(121) #plot the the original PCA result
plt.plot([1,2,3,4,5,6,7,8,9], pca_var_list, '-o',color='r',alpha=0.7,markeredgewidth=0.0)
plt.ylabel("Fraction of variance explained")
plt.xlabel("First n components")
plt.axhline(y=0.9, color='c', linestyle='--',alpha=0.8)
ax=plt.gca()
ax.set_ylim(0.65,1)

plt.subplot(122) #plot the modified PCA result
plt.plot([1,2,3,4,5,6,7,8,9], pca_var_list_3, '-o',color='r',alpha=0.7,markeredgewidth=0.0)
plt.ylabel("Fraction of variance explained (1st column * 20)")
plt.xlabel("First n components")
plt.axhline(y=0.9, color='c', linestyle='--',alpha=0.8)
ax=plt.gca()
ymin=pca_var_list_3[0]
ax.set_ylim(0.65,1)

plt.show()
