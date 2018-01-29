"""Graph visualization of correlations between all pairs of the variables"""

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_excel("breast-cancer-wisconsin.xlsx")
corr=df.corr(method='pearson') # pairwise correlation of columns, returning a 9*9 dataframe
corr_mat=corr.as_matrix() #convert the dataframe to a np array
for i in range(len(corr_mat)):
    corr_mat[i][i]=0 #set the self correlations to 0
corr_mat=abs(corr_mat)
print(corr)
graph1=nx.from_numpy_matrix(corr_mat)
edgelist_draw=[edge for edge in graph1.edges_iter(data=True) if edge[2]['weight']>0.6] #return the list of edges with a weight larger than 0.6


labeldict={}
for i in range(len(df.columns)):
    labeldict[i]=df.columns[i]
node_size=[1500,2100,2000,3500,1900,2800,2000,2700,2100,2000,1800]
edge_size=[0]*len(edgelist_draw)
for i in range(len(edgelist_draw)):
    edge_size[i]=(edgelist_draw[i][2]['weight']-0.6)/(1-0.6)*8

nx.draw_circular(graph1, edgelist=edgelist_draw, labels=labeldict,node_shape='o',node_color="c",node_size=node_size,font_size=10,width=edge_size, edge_color='r', alpha=0.8)  # matplotlib plot of the graph


for node in graph1.nodes(data=True): #label all the nodes
    node[1]["highest_corr"]=0
for edge in graph1.edges(data=True): #go through all the edges
    curr_weight = edge[2]['weight']
    if graph1.node[edge[0]]["highest_corr"]<curr_weight: #update the highest correlation score of the 2 nodes connecting the current edge
        graph1.node[edge[0]]["highest_corr"] = curr_weight
    if graph1.node[edge[1]]["highest_corr"]<curr_weight:
        graph1.node[edge[1]]["highest_corr"] = curr_weight

node_color=[]
for node in graph1.nodes(data=True):
    curr_highest_corr=node[1]["highest_corr"]
    if curr_highest_corr>0.9:
        node_color.append((0.1, 0.3, 1, 1))
    elif curr_highest_corr>0.8 and curr_highest_corr<=0.9:
        node_color.append((0.5, 0.7, 1, 1))
    elif curr_highest_corr>0.6 and curr_highest_corr<=0.8:
        node_color.append((0.85, 1, 1, 1))
    else:
        node_color.append((1, 1, 1, 1))


plt.figure() #open a new figure
nx.draw_circular(graph1, edgelist=edgelist_draw, labels=labeldict,node_shape='o',node_color=node_color,node_size=node_size,font_size=10,width=edge_size, edge_color='r', alpha=0.8)  # matplotlib plot of the graph

fig = plt.gcf() #get current figure
fig.set_size_inches(10, 10)
plt.savefig("Assignment5.pdf",dpi=200) #save the figure

plt.show()
