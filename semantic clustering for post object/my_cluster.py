
# libraries
import pandas as pd
import numpy as np
from string import digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from gensim.models import Word2Vec


# reading the data
def reading():
    flist = []  # list of words
    flist_first_element=[]
    whip = eval(open('1000_obj_list.txt', 'r').read())
    flist = [sent.split(", ") for sent in list(whip.values())]
    flist_first_element= [[sent.split(", ")[0]] for sent in list(whip.values())]
    flist_first_element_string= [sent.split(", ")[0] for sent in list(whip.values())]
    a=[]
    with open('categories_places365.txt') as f:
        remove_digits = str.maketrans('', '', digits)
        for line in f:
            inner_list = [elt.strip() for elt in line.split(' ')]
            inner_list = [i.translate(remove_digits) for i in inner_list][0][3:]
            flist.append(inner_list.split("/"))
            # a.append(inner_list.split("/"))
            flist_first_element_string.append(inner_list.split("/")[0])
            flist_first_element.append(inner_list.split("/"))
            # a.append(inner_list.split("/"))
    return flist, flist_first_element,flist_first_element_string

# calling the reading function and save the list of features(words) in featurs_list
featurs_list,featurs_list1,flist_first_element_string= reading()
print(featurs_list)

#making the word2vec model
model = Word2Vec(featurs_list1, min_count=1)
words = list(model.wv.vocab)
word2vec_model = model[model.wv.vocab]


# transform the n-dimentional model to two-dimentional model
pca = PCA(n_components=2)
result = pca.fit_transform(word2vec_model)


# clustering with k-means 
k = 15 # specify the number of clusters
kmeans = KMeans(n_clusters=k)
y_pred = kmeans.fit_predict(result)
df = pd.DataFrame(y_pred, columns=["cluster"])
df["c1"] = result[:, 0]
df["c2"] = result[:, 1]
df["c3"] = words

df_copy = pd.DataFrame()
df_copy=df.copy()

word_found=False
row_num=[]
for word in df_copy['c3']:
    word_found = False
    for word1 in flist_first_element_string:
        # print(int(df_copy[df_copy['c3'] == word1].index[0]))
        # print(df_copy[df_copy['c3'] == word1].index.item())

        if word==word1:
            print(word)
            word_found=True
            break
    if not word_found:
        print('not fond')
        word_found=False
        print(word)
        print(df_copy[df_copy['c3'] == word].index.item())
        row_num.append(df_copy[df_copy['c3'] == word].index.item())
        # df_copy.drop('1')
df_copy.drop(df_copy.index[row_num],inplace=True)
# df_copy.drop(df_copy.index[436],inplace=True)
# df_copy.drop(df_copy.index[438],inplace=True)
# df_copy.drop(df_copy.index[455],inplace=True)
# df_copy.drop(df_copy.index[477],inplace=True)
# df_copy.drop(df_copy.index[495],inplace=True)

df_copy = df_copy.reset_index(drop=True)

# compare feature list with model words
# flist_first_element_string#list of feature
# words # we have it from model
word_found=False
row_num=[]
for word_index in range(flist_first_element_string.__len__()):
    word_found = False
    for word1 in df_copy['c3']:
        # print(int(df_copy[df_copy['c3'] == word1].index[0]))
        # print(df_copy[df_copy['c3'] == word1].index.item())

        if flist_first_element_string[word_index]==word1:
            print(word1)
            print(word_index)
            word_found=True
            break
    if not word_found:
        print('not fond')
        word_found=False
        print(word1)
        # print(df_copy[df_copy['c3'] == word].index.item())
        # row_num.append(df_copy[df_copy['c3'] == word].index.item())
        row_num.append()
print()
        # df.drop(df.index[[1, 3]], inplace=True)

# visualize the clusters of given data
# list of selected colors to each cluster
# colors = ["black", "red", "orangered", "olive", "g", "teal", "rosybrown", "purple", "c", "crimson",
#           "sienna", "gold", "tan", "magenta", "deepskyblue", "darkgrey"]
#
# for index, row in df.iterrows():
#     plt.scatter(row["c1"], row["c2"], color=colors[int(row["cluster"])])
#     # plt.annotate(row["c3"], xy=(row["c1"], row["c2"]))
#
# plt.show()
diffrence_list=list(set(flist_first_element_string) - set(df_copy['c3']))
print('diffrence',list(set(flist_first_element_string) - set(df_copy['c3'].tolist())))
# print(Diff(flist_first_element_string, word1))



my_set=list(set([x for x in flist_first_element_string if flist_first_element_string.count(x) > 1]))
Dict_dup={}
for x in flist_first_element_string:
    if flist_first_element_string.count(x) > 1:
        Dict_dup[x]=flist_first_element_string.count(x)
dict_index=dict()
list_drop_clos=[]
for key in Dict_dup.keys():
    # index_0=flist_first_element_string.index(key)
    index_0=[i for i,val in enumerate(flist_first_element_string) if val==key]
    if index_0.__len__()==3:
        dict_index[key]=index_0[1:]
        list_drop_clos.extend(index_0[1:])
    else:
        dict_index[key] = index_0[-1]
        list_drop_clos.extend(index_0[1:])

# dict_index
# read  regular clustring data
# read data
df = pd.read_csv("posted_obj_predicted_ave.csv",header=None)
df0 = pd.read_csv("posted_obj_predicted_ave0.csv",header=None)
df1 = pd.read_csv("posted_obj_predicted_ave1.csv",header=None)
df2 = pd.read_csv("posted_obj_predicted_ave2.csv",header=None)
df3 = pd.read_csv("posted_obj_predicted_ave3.csv",header=None)


df_cluster = pd.DataFrame()
df_cluster0 = pd.DataFrame()
df_cluster1 = pd.DataFrame()
df_cluster2 = pd.DataFrame()
df_cluster3 = pd.DataFrame()
#drop regular clustering col
df.drop(df.columns[-1],axis=1,inplace=True)
df0.drop(df0.columns[-1],axis=1,inplace=True)
df1.drop(df1.columns[-1],axis=1,inplace=True)
df2.drop(df2.columns[-1],axis=1,inplace=True)
df3.drop(df3.columns[-1],axis=1,inplace=True)

# drop duplicated cols
# df.drop(df.columns[list_drop_clos],axis=1,inplace=True)# save the results to file
df.drop(df.columns[list_drop_clos],axis=1,inplace=True)
df0.drop(df0.columns[list_drop_clos],axis=1,inplace=True)
df1.drop(df1.columns[list_drop_clos],axis=1,inplace=True)
df2.drop(df2.columns[list_drop_clos],axis=1,inplace=True)
df3.drop(df3.columns[list_drop_clos],axis=1,inplace=True)


all_col_set=set(i for i in range(1365))
drop_col_set=set(list_drop_clos)
my_col=list(all_col_set-drop_col_set)
temp=[]
temp_dict={}
dict_group_index={}
for j in range(15):
    temp=[]
    for i in list(df_copy[df_copy.iloc[:,0]==j].index.values.astype(int)):
        temp.append(my_col[i])
    temp_dict[j]=temp




# code is true till here just check after this point.
#mean over symantic cols clusters
for key in temp_dict.keys():
    df[str(key)+'mean'] = df[ temp_dict[key]].mean(axis=1)
    df0[str(key)+'mean'] = df0[ temp_dict[key]].mean(axis=1)
    df1[str(key)+'mean'] = df1[ temp_dict[key]].mean(axis=1)
    df2[str(key)+'mean'] = df2[ temp_dict[key]].mean(axis=1)
    df3[str(key)+'mean'] = df3[ temp_dict[key]].mean(axis=1)

    df_cluster[str(key) + 'mean'] = df[str(key) + 'mean']
    df_cluster0[str(key) + 'mean'] = df0[str(key) + 'mean']
    df_cluster1[str(key) + 'mean'] = df1[str(key) + 'mean']
    df_cluster2[str(key) + 'mean'] = df2[str(key) + 'mean']
    df_cluster3[str(key) + 'mean'] = df3[str(key) + 'mean']


df_cluster_mean = pd.DataFrame()
df_cluster0_mean = pd.DataFrame()
df_cluster1_mean= pd.DataFrame()
df_cluster2_mean= pd.DataFrame()
df_cluster3_mean = pd.DataFrame()
ave_df_cluster=[]
ave_df_cluster0=[]
ave_df_cluster1=[]
ave_df_cluster2=[]
ave_df_cluster3=[]
for key in range(len(df_cluster.columns)):
    ave_df_cluster.append(df_cluster[str(key)+'mean'].mean())
    ave_df_cluster0.append(df_cluster0[str(key)+'mean'].mean())
    ave_df_cluster1.append(df_cluster1[str(key)+'mean'].mean())
    ave_df_cluster2.append(df_cluster2[str(key)+'mean'].mean())
    ave_df_cluster3.append(df_cluster3[str(key)+'mean'].mean())
path1='D:/rojiyar/my paper/result__/semantic clustring/posted_obj'
df_ave_df_cluster = pd.DataFrame(ave_df_cluster)
df_ave_df_cluster.to_csv(path1+"/postCnnObj_ave_df_cluster.csv",header=False,index=False)

df_ave_df_cluster0 = pd.DataFrame(ave_df_cluster0)
df_ave_df_cluster0.to_csv(path1+"/postCnnObj_ave_df_cluster0.csv",header=False,index=False)

df_ave_df_cluster1 = pd.DataFrame(ave_df_cluster1)
df_ave_df_cluster1.to_csv(path1+"/postCnnObj_ave_df_cluster1.csv",header=False,index=False)

df_ave_df_cluster2 = pd.DataFrame(ave_df_cluster2)
df_ave_df_cluster2.to_csv(path1+"/postCnnObj_ave_df_cluster2.csv",header=False,index=False)

df_ave_df_cluster3 = pd.DataFrame(ave_df_cluster3)
df_ave_df_cluster3.to_csv(path1+"/postCnnObj_ave_df_cluster3.csv",header=False,index=False)

df.to_csv(path1+"/postCnnObj_predicted_symantic.csv", header=False, index=False)
df0.to_csv(path1+"/postCnnObj_predicted_symantic0.csv", header=False, index=False)
df1.to_csv(path1+"/postCnnObj_predicted_symantic1.csv", header=False, index=False)
df2.to_csv(path1+"/postCnnObj_predicted_symantic2.csv", header=False, index=False)
df3.to_csv(path1+"/postCnnObj_predicted_symantic3.csv", header=False, index=False)

df_cluster.to_csv(path1+"/postCnnObj_predicted_symantic_final.csv", header=True, index=False)
df_cluster0.to_csv(path1+"/postCnnObj_predicted_symantic0_final.csv", header=False, index=False)
df_cluster1.to_csv(path1+"/postCnnObj_predicted_symantic1_final.csv", header=False, index=False)
df_cluster2.to_csv(path1+"/postCnnObj_predicted_symantic2_final.csv", header=False, index=False)
df_cluster3.to_csv(path1+"/postCnnObj_predicted_symantic3_final.csv", header=False, index=False)


# df_copy.to_csv("results1.csv")
print('bye')
