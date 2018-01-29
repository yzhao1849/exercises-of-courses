"""Read, Write, and Filter Data"""

import pandas as pd
def readdata(filename):
    df=pd.read_excel(filename)
    print("Number of instances:",df.shape[0],"\nNumber of columns:",df.shape[1])
    print("Column names:")
    for i in df.columns:
        print(i,end=" ")
    print()
    return df

#group the dataframe by the benign and malignant class labels and then calculates F for any given attribute (attri)
def f_score_class(df,attri):
    grouped = df.groupby('class')
    grand_mean=df[attri].mean()
    mean_2=grouped[attri].mean()[2]
    mean_4=grouped[attri].mean()[4]
    var_2 = (grouped[attri].std()[2]) ** 2
    var_4 = (grouped[attri].std()[4]) ** 2
    fscore = ((mean_2 - grand_mean) ** 2 + (mean_4 - grand_mean) ** 2) / (var_2 + var_4)
    return fscore


df1=readdata("breast-cancer-wisconsin.xlsx")
df1=df1.fillna(df1.mean()) #replace the NaN with the mean of the column
#replace the NaN with the mean of the column. Because the adding of these numbers does not change the sample mean
print()
print("value counts of the class column:")
print(df1['class'].value_counts())
print()

grouped=df1.groupby('class')

print("The number of instances in class 2 subgroup:",len(grouped.get_group(2)))
print("The number of instances in class 4 subgroup:",len(grouped.get_group(4)))
print()

list_fscore=pd.Series()
for i in df1.columns:
    if i!=df1.columns[0] and i!=df1.columns[-1]: #skip the first and last columns, code and class
        list_fscore[i]=f_score_class(df1,i)
print(list_fscore.sort_values(ascending=False))
#What do you expect the value of F will be for the “class” attribute itself?
#-->Infinity. Because the denominator is the sum of variance in each of the 2 subgroups, for the “class” attribute both will be 0, since elements in each subgroups have the same class

# Write a reduced dataset to disk, which contains only the five most relevant attributes
list_top5_attri=["uniCelShape","bareNuc","uniCelS","blaChroma","thickness","class"]
df_new=df1.loc[:,list_top5_attri]
df_new.to_excel('5_most_relevant_attributes.xls', index=False)
