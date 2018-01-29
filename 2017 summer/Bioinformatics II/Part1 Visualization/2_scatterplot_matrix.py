"""Producing a Scatterplot Matrix"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress

df=pd.read_excel("5_most_relevant_attributes.xls")
grouped=df.groupby('class')


bins=np.linspace(0,12,13)
#plot the diagonal cells of the matrix
for i in range(len(df.columns)-1):
    for j in range(len(df.columns)-1):
        if i==j:
            ax=plt.subplot(5,5,6*i+1) #for matrix diagonal
            x = grouped.get_group(2)[df.columns[i]]  # 2-benign
            y = grouped.get_group(4)[df.columns[i]]  # 4-malignant
            max_lim=max(x.max(),y.max())
            min_lim=min(x.min(),y.min())
            bins = np.linspace(min_lim-1,max_lim+1,max_lim-min_lim+3)
            plt.hist(x, bins, alpha=0.6, label="benign", color="r")
            plt.hist(y, bins, alpha=0.6, label="malignant", color="c")
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=8) #make the y axis fontsize smaller
            plt.subplots_adjust(hspace=0.2)
            plt.subplots_adjust(wspace=0.2) #0.3 for diagonal
            ax.set_title(df.columns[i],fontsize=10)
            ttl = ax.title
            ttl.set_position([0.5, 0.55])  # set position of the title
            plt.legend(bbox_to_anchor=(1, 1),fontsize=6) #adjust the location of the legend box

list_rvalue_2=pd.Series(name="benign") # A pandas serie to store the correlation coefficients
list_rvalue_4=pd.Series(name="malignant")
list_rvalue_2and4=pd.Series(name="all")

list_DSC=pd.Series(name="Distance consistency")
for i in range(5):
    for j in range(5):
        if i!=j: #off-diagonal cells
            plt.subplot(5, 5, i*5+j+1, aspect='equal')
            x_2 = grouped.get_group(2)[df.columns[i]]
            y_2 = grouped.get_group(2)[df.columns[j]]
            xy_2 = pd.concat([x_2, y_2], axis=1)  # Combine the 2 Series into a dataframe
            gxy_2_count = xy_2.groupby([x_2, y_2]).count()  # Group the dataframe by same tuple
            gxy_2_count = gxy_2_count.add_suffix('_Count').reset_index()  # flatten the hierarchical index of gxy_count
            plt.scatter(gxy_2_count[df.columns[j]], gxy_2_count[df.columns[i]], alpha=0.7,
                        s=(gxy_2_count[df.columns[i]+'_Count'])**(1/2)*10, c="r", edgecolors='none') #use square root to make the contrast between the size of the biggest points and smallest points not so big


            x_4 = grouped.get_group(4)[df.columns[i]]
            y_4 = grouped.get_group(4)[df.columns[j]]
            xy_4 = pd.concat([x_4, y_4], axis=1)  # Combine the 2 Series into a dataframe
            gxy_4_count = xy_4.groupby([x_4, y_4]).count()  # Group the dataframe by same tuple
            gxy_4_count = gxy_4_count.add_suffix('_Count').reset_index()  # flatten the hierarchical index of gxy_count
            plt.scatter(gxy_4_count[df.columns[j]], gxy_4_count[df.columns[i]], alpha=0.7,
                        s=(gxy_4_count[df.columns[i]+'_Count'])**(1/2)*10, c="c",edgecolors='none') #another option: np.log2 (logarithmic function of the number of overlapping points)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

            plt.subplots_adjust(hspace=0.2)
            plt.subplots_adjust(wspace=0.2)
            ax=plt.gca()

            ax.set_ylim(min(x_2.min(),x_4.min())-1,max(x_2.max(),x_4.max())+1)
            ax.set_xlim(min(y_2.min(),y_4.min())-1,max(y_2.max(),y_4.max())+1)

            if j>i:
                list_rvalue_2["{} vs {}".format(df.columns[i],df.columns[j])]=linregress(x_2, y_2)[2]
                list_rvalue_4["{} vs {}".format(df.columns[i], df.columns[j])] = linregress(x_4, y_4)[2]
                list_rvalue_2and4["{} vs {}".format(df.columns[i], df.columns[j])] = \
                linregress(pd.concat([x_2, x_4], axis=0), pd.concat([y_2, y_4], axis=0))[2]

                center2_x = x_2.mean()  # the x coordinate of the center of the benign cluster
                center2_y = y_2.mean()  # the y coordinate of the center of the benign cluster
                center4_x = x_4.mean()  # the x coordinate of the center of the malignant cluster
                center4_y = y_4.mean()  # the y coordinate of the center of the malignant cluster
                n_row_2=x_2.size # the number of points that belong to the benign class
                n_row_4=x_4.size # the number of points that belong to the benign class
                n_consistent=0 # the number of points that are closer to their own cluster center than to all others

                for index,row in xy_2.iterrows(): #for the points in the benign class
                    xi=row[df.columns[i]]
                    yi=row[df.columns[j]]
                    if (xi-center2_x)**2+(yi-center2_y)**2<=(xi-center4_x)**2+(yi-center4_y)**2: # if the current point is closer to their own cluster center than to all others
                        n_consistent+=1
                for index,row in xy_4.iterrows(): #for the points in the malignant class
                    xi=row[df.columns[i]]
                    yi=row[df.columns[j]]
                    #if (xi-center2_x)**2+(yi-center2_y)**2>=(xi-center4_x)**2+(yi-center4_y)**2:
                    if (xi-center4_x)**2+(yi-center4_y)**2<=(xi-center2_x)**2+(yi-center2_y)**2:
                        n_consistent+=1
                list_DSC["{} vs {}".format(df.columns[i],df.columns[j])]=n_consistent/(n_row_2+n_row_4) #the distance consistency



rvalue_table=pd.concat([list_rvalue_2, list_rvalue_4,list_rvalue_2and4], axis=1)
print("The correlation rvalues between all pairs of variables in the whole dataset:\n",list_rvalue_2and4.sort_values(ascending=False))
print("Summary of correlation rvalues between all pairs:\n",rvalue_table)
print()
print(list_DSC.sort_values(ascending=False))

fig = plt.gcf() #get current figure
fig.set_size_inches(15, 15)
plt.savefig("Assignment3.pdf",dpi=200) #save the figure

plt.show()




