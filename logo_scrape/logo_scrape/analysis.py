# TODO What if evalution data is not properly mapped

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
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#
##transform the pattern_id feature to int
#encoding_feature = ["pattern_id"]
#enc = LabelEncoder()
#enc.fit(encoding_feature)
#working_feature = enc.transform(encoding_feature)
#working_feature = working_feature.reshape(-1, 1)
#ohe = OneHotEncoder(sparse=False)
#
#
#
#
##onehotencoder = OneHotEncoder(categorical_features=[df.columns.tolist().index('pattern_id')])
#onehotencoder = OneHotEncoder(categorical_features=["pattern_id"])
#df = onehotencoder.fit_transform(df)
#
##convert the pattern_id feature to separate binary features
#onehotencoder = OneHotEncoder(categorical_features=working_feature, sparse=False)
#df = onehotencoder.fit_transform(df).toarray()


                # Adding features
                
#Split pattern_to binary
df = pd.get_dummies(df, columns=["pattern_id"], drop_first=True)

#add a feature: does img_link has "logo in name"
pattern_logo = r".+logo"
tmp_has_logo_arr = list()
for index, row in df.iterrows():
    if re.match(pattern_logo, str(row["link_img"]), flags=re.IGNORECASE):
        tmp_has_logo_arr.append(1)
    else:
        tmp_has_logo_arr.append(0)        
df["has_logo_in_name"] = tmp_has_logo_arr
del tmp_has_logo_arr

#has png in img_link feature
pattern_logo = r".+png"
tmp_has_logo_arr = list()
for index, row in df.iterrows():
    if re.match(pattern_logo, str(row["link_img"]), flags=re.IGNORECASE):
        tmp_has_logo_arr.append(1)
    else:
        tmp_has_logo_arr.append(0)        
df["has_png_in_name"] = tmp_has_logo_arr
del tmp_has_logo_arr



#has jpg in img_link feature
pattern_logo = r".+jpg"
tmp_has_logo_arr = list()
for index, row in df.iterrows():
    if re.match(pattern_logo, str(row["link_img"]), flags=re.IGNORECASE):
        tmp_has_logo_arr.append(1)
    else:
        tmp_has_logo_arr.append(0)        
df["has_jpg_in_name"] = tmp_has_logo_arr
del tmp_has_logo_arr


#has jpeg in img_link feature
pattern_logo = r".+jpeg"
tmp_has_logo_arr = list()
for index, row in df.iterrows():
    if re.match(pattern_logo, str(row["link_img"]), flags=re.IGNORECASE):
        tmp_has_logo_arr.append(1)
    else:
        tmp_has_logo_arr.append(0)        
df["has_jpeg_in_name"] = tmp_has_logo_arr
del tmp_has_logo_arr

#has svg in img_link feature
pattern_logo = r".+svg"
tmp_has_logo_arr = list()
for index, row in df.iterrows():
    if re.match(pattern_logo, str(row["link_img"]), flags=re.IGNORECASE):
        tmp_has_logo_arr.append(1)
    else:
        tmp_has_logo_arr.append(0)        
df["has_svg_in_name"] = tmp_has_logo_arr
del tmp_has_logo_arr



#has svg in img_link feature
pattern_logo = r".+svg"
tmp_has_logo_arr = list()
for index, row in df.iterrows():
    if re.match(pattern_logo, str(row["link_img"]), flags=re.IGNORECASE):
        tmp_has_logo_arr.append(1)
    else:
        tmp_has_logo_arr.append(0)        
df["has_svg_in_name"] = tmp_has_logo_arr
del tmp_has_logo_arr


#has twitter in img_link feature
pattern_logo = r".+twitter"
tmp_has_logo_arr = list()
for index, row in df.iterrows():
    if(
       re.match(pattern_logo, str(row["link_img"]), flags=re.IGNORECASE) or 
       re.match(pattern_logo, str(row["current_link"]), flags=re.IGNORECASE) or
       re.match(pattern_logo, str(row["link_id"]), flags=re.IGNORECASE)
       ):
        tmp_has_logo_arr.append(1)
    else:
        tmp_has_logo_arr.append(0)        
df["has_twitter_in_name"] = tmp_has_logo_arr
del tmp_has_logo_arr

#has instagram in img_link feature
pattern_logo = r".+instagram"
tmp_has_logo_arr = list()
for index, row in df.iterrows():
    if(
       re.match(pattern_logo, str(row["link_img"]), flags=re.IGNORECASE) or 
       re.match(pattern_logo, str(row["current_link"]), flags=re.IGNORECASE) or
       re.match(pattern_logo, str(row["link_id"]), flags=re.IGNORECASE)
       ):
        tmp_has_logo_arr.append(1)
    else:
        tmp_has_logo_arr.append(0)        
df["has_instagram_in_name"] = tmp_has_logo_arr
del tmp_has_logo_arr

#has facebook in img_link feature
pattern_logo = r".+facebook"
tmp_has_logo_arr = list()
for index, row in df.iterrows():
    if(
       re.match(pattern_logo, str(row["link_img"]), flags=re.IGNORECASE) or 
       re.match(pattern_logo, str(row["current_link"]), flags=re.IGNORECASE) or
       re.match(pattern_logo, str(row["link_id"]), flags=re.IGNORECASE)
       ):
        tmp_has_logo_arr.append(1)
    else:
        tmp_has_logo_arr.append(0)        
df["has_facebook_in_name"] = tmp_has_logo_arr
del tmp_has_logo_arr

#has youtube in img_link feature
pattern_logo = r".+youtube"
tmp_has_logo_arr = list()
for index, row in df.iterrows():
    if(
       re.match(pattern_logo, str(row["link_img"]), flags=re.IGNORECASE) or 
       re.match(pattern_logo, str(row["current_link"]), flags=re.IGNORECASE) or
       re.match(pattern_logo, str(row["link_id"]), flags=re.IGNORECASE)
       ):
        tmp_has_logo_arr.append(1)
    else:
        tmp_has_logo_arr.append(0)        
df["has_youtube_in_name"] = tmp_has_logo_arr
del tmp_has_logo_arr

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
        
        
        #TEST which parameter is sufficient

#Binary features
features_to_be_tested = ["has_youtube_in_name",
                         "has_svg_in_name",
                         "has_instagram_in_name",
                         "has_facebook_in_name",
                         "has_jpeg_in_name",
                         "has_jpg_in_name",
                         "has_png_in_name",
                         "has_logo_in_name",
                         "pattern_id_2",
                         "pattern_id_3",
                         "pattern_id_4",
                         "pattern_id_5",
                         "pattern_id_6",
                         "is_http"
                         ]

#barplot features:
for feature in features_to_be_tested:
    plt.bar([0,1], [len(df[df[feature]==0]), len(df[df[feature]==1])])
    plt.show()


from scipy import stats
for feature in features_to_be_tested:
    # None of them have uniform distribution
    print(stats.kstest(df[feature], stats.uniform(loc=0.0, scale=1.0).cdf))

        # Try ML algorithms
        
#feature that hold the amount of appearances of an image per 10 pages of a site
count_image_count_dict = dict()
for site in site_img_dict:
    for img in site_img_dict[site]:
        if(site_img_dict[site][img][0] in count_image_count_dict):
            count_image_count_dict[site_img_dict[site][img][0]] += 1
        else:
            count_image_count_dict[site_img_dict[site][img][0]] = 1
plt.bar(count_image_count_dict.keys(), count_image_count_dict.values())
plt.show()



#Attemp of linear regression with exponent func
from sklearn.svm import SVR
expon_dist_feat_y = count_image_count_dict.values()
expon_dist_feat_x = list(count_image_count_dict.keys())
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
y_rbf = svr_rbf.fit(expon_dist_feat_x ,
                    expon_dist_feat_y )
# Failed, we can't rely on regression, our problem is unsupervised

#make a test dict
test_dict = dict()
for i in range(0,10):
    el_key = random.choice(list(site_img_dict.keys()))
    test_dict[el_key] = site_img_dict[el_key]
    del site_img_dict[el_key]
    
#make plots between features
for site in site_img_dict:
    for img in site_img_dict[site]:
        plt.scatter(site_img_dict[site][img]["repeat_count"],site_img_dict[site][img]["repeat_count"])

#Break down data to a matrix of coordinates
cordinates_fetures_all = list()
for site in site_img_dict:
    for img in site_img_dict[site]:
        cordinates_fetures_all.append(site_img_dict[site][img])

working_x = num.array(cordinates_fetures_all)
kmeans_auto = k_means(n_clusters=2, algorithm="auto")
kmeans_auto.fit(working_x)
centroids_auto = kmeans_auto.cluster_centers_

kmeans_full = k_means(n_clusters=2, algorithm="full")
kmeans_full.fit(working_x)
centroids_full = kmeans_full.cluster_centers_

kmeans_elkan = k_means(n_clusters=2, algorithm="elkan")
kmeans_elkan.fit(working_x)
centroids_elkan = kmeans_elkan.cluster_centers_

#get the difference between the algorithms:
# TODO: test the results
print(centroids_auto - centroids_full )
print(centroids_auto - centroids_elkan )
print(centroids_full - centroids_elkan)

#Izpolzvame PCA, za da mahnem nenujnite feature-i
pca = PCA(n_components=2)
pca.fit(working_x)
#Imame 0.9 za dispersiqta, koeto bi trqbvalo da oznachava, che ne sme izgubili mnogo informaciq
print(pca.explained_variance_ratio_)
train_X = pca.transform(working_x)

pca_var = PCA(.95)
pca_var .fit(working_x)
print(pca_var .explained_variance_ratio_)
#Ostavi ni 3 feature-a
train_X = pca_var.transform(working_x)

test_val = test_dict[828]['i/logo_affilired1.png']
for id in test_dict:
    for img in test_dict[id]:
        test_val = pca_var.transform([test_dict[id][img]])
        kmeans_auto_pca_var = k_means(n_clusters=2)
        kmeans_auto_pca_var.fit(train_X)
        print(kmeans_auto_pca_var.predict(test_val))
        print(img)

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

count = 0
evaluation_95, evaluation_88, evaluation_2_comp, evaluation_15_comp = 0, 0, 0, 0
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
# Results:
#
#print(evaluation_88/count)
#print(evaluation_2_comp/count)
#print(evaluation_15_comp/count)
#0.12903225806451613
#0.12903225806451613
#0.12903225806451613
        