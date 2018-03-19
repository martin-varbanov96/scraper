import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from collections import Counter as lin_sort
import re
import random
import pdb
import numpy as num
from sklearn.cluster import KMeans as k_means
from sklearn.decomposition import PCA

path_to_input = "data_sets/20180315_new_features.csv"
df = pd.read_csv(path_to_input, sep="	")


#Add occurances of img in a site feature
#working data for clustering
hits_per_regex = list()
site_img_dict = dict()
for index, row in df.iterrows():
    if row["site_id"] in site_img_dict:
        if row["link_img"] in site_img_dict[row["site_id"]]:
            site_img_dict[row["site_id"]][row["link_img"]]["repeat_count"] += 1
        else:
            site_img_dict[row["site_id"]][row["link_img"]] = {
                                "repeat_count": 1,
                                "has_youtube_in_name": row["has_youtube_in_name"],
                                "has_svg_in_name": row["has_svg_in_name"],
                                "has_instagram_in_name": row["has_instagram_in_name"],
                                "has_facebook_in_name": row["has_facebook_in_name"],
                                "has_jpeg_in_name": row["has_jpeg_in_name"],
                                "has_jpg_in_name": row["has_jpg_in_name"],
                                "has_png_in_name": row["has_png_in_name"],
                                "has_logo_in_name": row["has_logo_in_name"],
                                "pattern_id_2": row["pattern_id_2"],
                                "pattern_id_3": row["pattern_id_3"],
                                "pattern_id_4": row["pattern_id_4"],
                                "pattern_id_5": row["pattern_id_5"],
                                "pattern_id_6": row["pattern_id_6"],
                                "is_http": row["is_http"],
                                "img_link": row["link_img"] 
                                }
    else:
        site_img_dict[row["site_id"]] = dict()
        site_img_dict[row["site_id"]][row["link_img"]] = {
            "repeat_count": 1,
            "has_youtube_in_name": row["has_youtube_in_name"],
            "has_svg_in_name": row["has_svg_in_name"],
            "has_instagram_in_name": row["has_instagram_in_name"],
            "has_facebook_in_name": row["has_facebook_in_name"],
            "has_jpeg_in_name": row["has_jpeg_in_name"],
            "has_jpg_in_name": row["has_jpg_in_name"],
            "has_png_in_name": row["has_png_in_name"],
            "has_logo_in_name": row["has_logo_in_name"],
            "pattern_id_2": row["pattern_id_2"],
            "pattern_id_3": row["pattern_id_3"],
            "pattern_id_4": row["pattern_id_4"],
            "pattern_id_5": row["pattern_id_5"],
            "pattern_id_6": row["pattern_id_6"],
            "is_http": row["is_http"],
            "img_link": row["link_img"] 
            }
        
transport_dict_for_df = {
            "repeat_count": [],
            "has_youtube_in_name": [],
            "has_svg_in_name": [],
            "has_instagram_in_name": [],
            "has_facebook_in_name": [],
            "has_jpeg_in_name": [],
            "has_jpg_in_name": [],
            "has_png_in_name": [],
            "has_logo_in_name": [],
            "pattern_id_2": [],
            "pattern_id_3": [],
            "pattern_id_4": [],
            "pattern_id_5": [],
            "pattern_id_6": [],
            "is_http": [],
            "img_link": [] 
            }

for id in site_img_dict:
    for img in site_img_dict[id]:
        for key in site_img_dict[id][img]:
            transport_dict_for_df[key].append(site_img_dict[id][img][key])
after_feature_df = pd.DataFrame(data=transport_dict_for_df)


count_image_count_dict = dict()
for site in site_img_dict:
    for img in site_img_dict[site]:
        if(site_img_dict[site][img][0] in count_image_count_dict):
            count_image_count_dict[site_img_dict[site][img][0]] += 1
        else:
            count_image_count_dict[site_img_dict[site][img][0]] = 1
plt.bar(count_image_count_dict.keys(), count_image_count_dict.values())
plt.show()


#preprocessing data
from sklearn.model_selection import train_test_split        
X = after_feature_df[after_feature_df.columns.difference(['img_link'])]
y = after_feature_df["img_link"]
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

y_test=y_test.tolist()
y_train=y_train.tolist()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# creating dimensioned-reduced data
from sklearn.decomposition import PCA
pca_95 = PCA(.95)
pca_95.fit(X_train)
X_train_PCA_95 = pca_95.transform(X_train)
X_test_PCA_95 = pca_95.transform(X_test)
kmeans_pca_95 = k_means(n_clusters=2)
kmeans_pca_95.fit(X_train_PCA_95)

pca_88 = PCA(.88)
pca_88.fit(X_train)
X_train_PCA_88 = pca_88.transform(X_train)
X_test_PCA_88 = pca_88.transform(X_test)
kmeans_pca_88 = k_means(n_clusters=2)
kmeans_pca_88.fit(X_train_PCA_88)

pca_2_comp = PCA(n_components=2)
pca_2_comp.fit(X_train)
X_train_PCA_2_comp = pca_2_comp.transform(X_train)
X_test_PCA_2_comp = pca_2_comp.transform(X_test)
kmeans_pca_2_comp = k_means(n_clusters=2)
kmeans_pca_2_comp.fit(X_train_PCA_2_comp)

pca_15_comp = PCA(n_components=15)
pca_15_comp.fit(X_train)
X_train_PCA_15_comp = pca_15_comp.transform(X_train)
X_test_PCA_15_comp = pca_15_comp.transform(X_test)
kmeans_pca_15_comp = k_means(n_clusters=2)
kmeans_pca_15_comp.fit(X_train_PCA_15_comp)

#make a final df with predicted results
transport_to_test_df_dict = {"pca_95_kmeans": [],
                             "pca_88_kmeans": [],
                             "pca_2_comp_kmeans": [],
                             "pca_15_comp_kmeans": [],
                             "img_link": [],
                             "manual_img_evaluation": [],
                             }
for key in range(0, len(X_test_PCA_95)):
    #make predictions
    transport_to_test_df_dict["pca_95_kmeans"].append(kmeans_pca_95.predict([X_test_PCA_95[key]]))
    transport_to_test_df_dict["pca_88_kmeans"].append(kmeans_pca_88.predict([X_test_PCA_88[key]]))
    transport_to_test_df_dict["pca_2_comp_kmeans"].append(kmeans_pca_2_comp.predict([X_test_PCA_2_comp[key]]))
    transport_to_test_df_dict["pca_15_comp_kmeans"].append(kmeans_pca_15_comp.predict([X_test_PCA_15_comp[key]]))

    transport_to_test_df_dict["img_link"].append(y_test[key])
    transport_to_test_df_dict["manual_img_evaluation"].append("not evaluated")

after_analysis_df = pd.DataFrame(data=transport_to_test_df_dict)


#preprocessing data
from sklearn.model_selection import train_test_split        
X = after_feature_df[after_feature_df.columns.difference(['img_link'])]
y = after_feature_df["img_link"]
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

y_test=y_test.tolist()
y_train=y_train.tolist()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
x_train_unscaled = X_train
x_test_unscaled = X_test
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

        # Feature selection

# reduce features, based on vairance
from sklearn.feature_selection import VarianceThreshold
probability_val = 0
binom_var_X_train = VarianceThreshold(threshold=(probability_val * (1 - probability_val)))
binom_var_X_train = binom_var_X_train.fit_transform(X_train)
# features are brought down from 15 to 14
# TODO: Apply unsupervised algorithms

for el in working_:
#    plt.plot(working_x[i][0], working_x[i][1], colors[labels[i]])
 #   plt.scatter(centroids[:, 0], centroids[:, 1], marker="X", zorder=10)
      
# TODO: Is there a unsupervised selectKBest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X_k_best = SelectKBest(chi2, k=2).fit(X_train)


# return k_means object
def get_k_means_obj_percentage(X_train):
    kmeans_obj = k_means(n_clusters=2)
    kmeans_obj.fit(X_train)
    return kmeans_obj

# convert data to pca
def get_pca_transfered_data(X_train, X_test, value, n_component=0):
    if n_component:
        pca = PCA(n_components=value)
    else:
        pca = PCA(value)
    pca.fit(X_train)
    return pca.transform(X_train), pca.transform(X_test)
    
# creating dimensioned-reduced data
from sklearn.decomposition import PCA

#k-means on raw data
k_means_raw_data = get_k_means_obj_percentage(X_train)



#95% conf level
X_train_PCA_95, X_test_PCA_95 = get_pca_transfered_data(X_train,
                                                        X_test,
                                                        .95)
kmeans_pca_95 = get_k_means_obj_percentage(X_train_PCA_95)

#88% conf level
X_train_PCA_88, X_test_PCA_88 = get_pca_transfered_data(X_train,
                                                        X_test,
                                                        .88)
kmeans_pca_88 = get_k_means_obj_percentage(X_train_PCA_88)

# leaving 2 coponents
X_train_PCA_2_comp, X_test_PCA_2_comp = get_pca_transfered_data(X_train,
                                                        X_test,
                                                        2,
                                                        n_component=1
                                                        )
kmeans_pca_2_comp = get_k_means_obj_percentage(X_train_PCA_2_comp)

# leaving 2 coponents
X_train_PCA_15_comp, X_test_PCA_15_comp = get_pca_transfered_data(X_train,
                                                        X_test,
                                                        15,
                                                        n_component=1
                                                        )
kmeans_pca_15_comp = get_k_means_obj_percentage(X_train_PCA_15_comp)

#make a final df with predicted results
transport_to_test_df_dict = {"pca_95_kmeans": [],
                             "pca_88_kmeans": [],
                             "pca_2_comp_kmeans": [],
                             "pca_15_comp_kmeans": [],
                             "img_link": [],
                             "manual_img_evaluation": [],
                             }
for key in range(0, len(X_test_PCA_95)):
    #make predictions
    transport_to_test_df_dict["pca_95_kmeans"].append(kmeans_pca_95.predict([X_test_PCA_95[key]]))
    transport_to_test_df_dict["pca_88_kmeans"].append(kmeans_pca_88.predict([X_test_PCA_88[key]]))
    transport_to_test_df_dict["pca_2_comp_kmeans"].append(kmeans_pca_2_comp.predict([X_test_PCA_2_comp[key]]))
    transport_to_test_df_dict["pca_15_comp_kmeans"].append(kmeans_pca_15_comp.predict([X_test_PCA_15_comp[key]]))

    transport_to_test_df_dict["img_link"].append(y_test[key])
    transport_to_test_df_dict["manual_img_evaluation"].append("not evaluated")

after_analysis_df = pd.DataFrame(data=transport_to_test_df_dict)

count = 0
evaluation_95, evaluation_88, evaluation_2_comp, evaluation_15_comp = 0, 0, 0, 0
#after_analysis_df has to be manualy evaluated/edit-ed to work
for index, row in after_analysis_df.iterrows():
    if row["manual_img_evaluation"] == "Logo":
        count+=1
        evaluation_95 = evaluation_95+1 if row["pca_95_kmeans"] == [1] else evaluation_95
        evaluation_88 = evaluation_95+1 if row["pca_95_kmeans"] == [1] else evaluation_95
        evaluation_2_comp = evaluation_2_comp+1 if row["pca_2_comp_kmeans"] == [1] else evaluation_2_comp
        evaluation_15_comp = evaluation_15_comp +1 if row["pca_15_comp_kmeans"] == [1] else evaluation_15_comp 
    elif row["manual_img_evaluation"] == "Not Logo":
        count+=1
        evaluation_95 = evaluation_95+1 if row["pca_95_kmeans"] == [0] else evaluation_95
        evaluation_88 = evaluation_95+1 if row["pca_95_kmeans"] == [0] else evaluation_95
        evaluation_2_comp = evaluation_2_comp+1 if row["pca_2_comp_kmeans"] == [0] else evaluation_2_comp
        evaluation_15_comp = evaluation_15_comp +1 if row["pca_15_comp_kmeans"] == [0] else evaluation_15_comp 
print(evaluation_95/count)
print(evaluation_88/count)
print(evaluation_2_comp/count)
print(evaluation_15_comp/count)
