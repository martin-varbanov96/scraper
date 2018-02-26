import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from collections import Counter as lin_sort
import re
import pdb
import numpy as num
from sklearn.cluster import KMeans as k_means

path_to_input = "data_sets/20181702_post_async_bug_fix.csv"
df = pd.read_csv(path_to_input)


def plot_k_means(input_list):
    working_x = num.array(input_list)
    kmeans = k_means(n_clusters=2)
    kmeans.fit(working_x)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    colors = ["g.", "r."]
    for i in range(len(working_x)):
        plt.plot(working_x[i][0], working_x[i][1], colors[labels[i]])
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="X", zorder=10)
    plt.show()


# Plot the deviation of patterns used
quantities_of_pattern = lin_sort(df[:]["pattern_id"])
# plt.bar(range(len(quantities_of_pattern)), list(quantities_of_pattern.values()), align='center')
# plt.show()

# Find the amount occurs per 10 pages
quantity_links = lin_sort(df[:]["site_id"])
quantity_links_dict = dict()
for key in quantity_links:
    if(quantity_links[key] in quantity_links_dict):
        quantity_links_dict[quantity_links[key]] += 1
    else:
        quantity_links_dict[quantity_links[key]] = 1
# Get quantity of results
quantity_of_sites = len(lin_sort(df[:]["site_id"]))

#Standart k_means
#hits_per_regex = list()
#site_img_dict = dict()
#for i in range(1, 9):
#    df_tmp = df.loc[df["pattern_id"] == i]
#    # df_tmp = lin_sort(df_tmp[:]["site_id"])
#    for index, row in df_tmp.iterrows():
#        if row["site_id"] in site_img_dict:
#            if row["link_img"] in site_img_dict[row["site_id"]]:
#                site_img_dict[row["site_id"]][row["link_img"]][0] += 1
#            else:
#                site_img_dict[row["site_id"]][row["link_img"]] = [1, i]
#        else:
#            site_img_dict[row["site_id"]] = dict()
#            # site_img_dict[row["site_id"]][row["link_img"]] = list()
#            site_img_dict[row["site_id"]][row["link_img"]] = [1, i]
#print(site_img_dict)
#for site in site_img_dict:
#    for img in site_img_dict[site]:
#        plt.scatter(site_img_dict[site][img][0], site_img_dict[site][img][1])
#plt.show()
#coordinates_list = []
#for site in site_img_dict:
#    for image in site_img_dict[site]:
#        coordinates_list.append(site_img_dict[site][image])
#plot_k_means(coordinates_list)



#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#label_encoder_x_1 = LabelEncoder()
#x[:, 1] = label_encoder_x_1.fit_transform(x[:, 1]) 
#label_encoder_x_2 = LabelEncoder()
#x[:, 2] = label_encoder_x_2.fit_transform(x[:, 2]) 
#onehotencoder = OneHotEncoder(categorical_features=[1])
#x = onehotencoder.fit_transform(x).toarray()

#Convert pattern column to separate columns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#transform the pattern_id feature to int
encoding_feature = ["pattern_id"]
enc = LabelEncoder()
enc.fit(encoding_feature)
working_feature = enc.transform(encoding_feature)
working_feature = working_feature.reshape(-1, 1)
ohe = OneHotEncoder(sparse=False)


onehotencoder = OneHotEncoder(categorical_features=[df.columns.tolist().index('pattern_id')])
df = onehotencoder.fit_transform(df)

#convert the pattern_id feature to separate binary features
onehotencoder = OneHotEncoder(categorical_features=working_feature, sparse=False)
df = onehotencoder.fit_transform(df).toarray()


hits_per_regex = list()
site_img_dict = dict()
for i in range(1, 9):
    df_tmp = df.loc[df["pattern_id"] == i]
    # df_tmp = lin_sort(df_tmp[:]["site_id"])
    for index, row in df_tmp.iterrows():
        if row["site_id"] in site_img_dict:
            if row["link_img"] in site_img_dict[row["site_id"]]:
                site_img_dict[row["site_id"]][row["link_img"]][0] += 1
            else:
                site_img_dict[row["site_id"]][row["link_img"]] = [1, i]
        else:
            site_img_dict[row["site_id"]] = dict()
            # site_img_dict[row["site_id"]][row["link_img"]] = list()
            site_img_dict[row["site_id"]][row["link_img"]] = [1, i]
