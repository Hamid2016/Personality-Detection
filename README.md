# Personality-Detection
Here is a brief description to what we did in our paper which contains two main parts. We are going to descrip each part in detail


part one:
We Found relationship between each group in first part of our project and each semantic group in second part of our project
•	Use image-based social media platform: Flicker 
•	Collect required data through the Flicker API. Descriptive statistics was shown in following table
•	We conducted an online survey where we asked participants to ﬁll in a personality questionnaire. 
•	To analyze content of the images on Flicker , we used vgg net or cnn algorithm through ImagNet dataset
•	CLUSTERING USERS BASE ON CNN- OBJ (similarity in content of posted images )
•	Calculate the avg of personality of users who belong to each cluster
•	Clustering obj and items in the images  semantically
•	Comparing the result to show 


Regular clustering steps:
1.	Read post, fave or profile outcome
2.	Make object for k-mean clustering for n=4
3.	Remove name column
4.	Do k-mean clustering by using data from previous step
5.	Add prediction column to the data in step 3 (as last column)
6.	Divide data into 4 group based on last column of data. We call it df0 to df3
7.	Mean over rows. We have 280 rows and 5 columns. Then we have 5 columns and one row which means we average all rows over columns
8.	Save data from all average of all four groups. We call it faveOutcome_ave0 to 3 for fave modality 
9.	We read faveOutcome again and remove first column or name column and add prediction column to it. 
10.	Divide to 4 groups based on 4 groups of prediction
11.	Save data as fave_obj_predicted_ave and 0-3. fave_obj_predicted_ave is main faveOutcome data without nae column and with prediction column. fave_obj_predicted_ave0-3 are for each group of prediction. Just added prediction and remove name column. We need for semantic clustering 

Semantic clustering steps:
1.Read columns name. we have 1000_obj_list.text and categories_places365.text for columns and save it in feature list
2.Make word2vec model
3.Get list of words in our model
4.transform the n-dimentional model to two-dimentional model
5.clustring data with k=15
6.make a frame and add prediction column to it(column was resulted from previous step)
7. first column is first row of result we got in step in step 4 and second column as second column of result and finally as third column of the dataframe we put words we got in step 2
8.copy dataframe we created in step 7
9.make row_num which is difference between features list that we got step one and words that words list that we got in step two. We took indexes which are in the list in first step and not in list step two or list (list after training)
10.drop column indexes that we got from step 10
11.Detelet columns which are repeated more than one time (to understand it, a feature can be part of another feature or be contained part of that)   
12.find duplicated feature indexes 
13.drop duplicated columns. Keep only one of them (first index that it showed up)
14.read regular clustering data which we got in regular clustering (fave_obj_predicted_ave 0-3)
15.drop regular clustering result col in fave_obj_predicted_ave 0-3
16.drop columns which are duplicated. Indexes we found in step 12
17.we found all indexes minus duplicated columns earlier. In this step we make a dictionary in which we have 0-14 key to and indexes for each cluster. We call this dictionary temp_dict
18.Mean over semantic cluster columns and add columns to each data frame as 0-14mean. We have 0-3 regular clustering in which we have added 0-14mean columns (For example 0mean column is average columns belong to 0 semantic cluster. As a result, we have 4 regular clusters in all of them we have 15 average cloumns) 
19.we mean over all rows. Therefor we have 4 groups of regular clustering and after applying mean will be 14 columns and only one row because of averaging 


