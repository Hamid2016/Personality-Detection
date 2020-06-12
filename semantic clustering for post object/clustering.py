from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
path='D:/rojiyar/my paper'
df = pd.read_csv(path+"/zahra_plos_data/FlickrandTwitter/Flickr/postCnnObj.csv",header=None)# read postcnnobj
df_out = pd.read_csv(path+"/zahra_plos_data/FlickrandTwitter/Flickr/postOutcome.csv",header=None) #read postoutcome

cols = [ 1, 2, 3, 4,5]
df_mean_= pd.read_csv(path+"/zahra_plos_data/FlickrandTwitter/Flickr/postOutcome.csv",header=None,  usecols=cols) #remove name colum
df_out = pd.read_csv(path+"/zahra_plos_data/FlickrandTwitter/Flickr/postOutcome.csv",header=None)



km = KMeans(n_clusters=4)

df_new=df.drop(df.columns[0], axis=1) #remove names column
y_predicted = km.fit_predict(df_new)
print(y_predicted)
# df_new['cluster']=y_predicted# add cluser column to our data

df_out[len(df_out.columns)+1] = y_predicted
df_mean_[len(df_mean_.columns)+1] = y_predicted


df0=df_mean_[df_mean_.iloc[:,-1]==0]
df1=df_mean_[df_mean_.iloc[:,-1]==1]
df2=df_mean_[df_mean_.iloc[:,-1]==2]
df3=df_mean_[df_mean_.iloc[:,-1]==3]


df0_mean=df0.mean()
df1_mean=df1.mean()
df2_mean=df2.mean()
df3_mean=df3.mean()

# result for regular clustering
path='D:/rojiyar/my paper/result__/regular clustring'
df_out.to_csv(path+"/posted_obj/postOutcome.csv",index=None)

df0_mean.to_csv(path+"/posted_obj/postOutcome_ave0.csv",index=None)
df1_mean.to_csv(path+"/posted_obj/postOutcome_ave1.csv",index=None)
df2_mean.to_csv(path+"/posted_obj/postOutcome_ave2.csv",index=None)
df3_mean.to_csv(path+"/posted_obj/postOutcome_ave3.csv",index=None)

# result we need for symantic clustring
df_new[len(df_new.columns)+1] = y_predicted
# /home/hamid/PycharmProjects/research/code for project/K-mean semantically on features/cluster-semantic-vectors-master/clustering-symantic-relationship
df0_new=df_new[df_new.iloc[:,-1]==0]
df1_new=df_new[df_new.iloc[:,-1]==1]
df2_new=df_new[df_new.iloc[:,-1]==2]
df3_new=df_new[df_new.iloc[:,-1]==3]


# save main data frame which has df_new + predicted value
df_new.to_csv(r"posted_obj_predicted_ave.csv", header=False, index=False)

# dataframe for cluster 0
df0_new.to_csv(r"posted_obj_predicted_ave0.csv", header=False, index=False)

# dataframe for cluster 1
df1_new.to_csv(r"posted_obj_predicted_ave1.csv",header=False, index=False)

# dataframe for cluster 2
df2_new.to_csv(r"posted_obj_predicted_ave2.csv",header=False, index=False)

# dataframe for cluster 3
df3_new.to_csv(r"posted_obj_predicted_ave3.csv",header=False, index=False)

print(df_out)







