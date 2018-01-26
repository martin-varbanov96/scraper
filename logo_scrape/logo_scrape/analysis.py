import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter as lin_sort
import re
import pdb

path_to_input = "../data_sets/201801210024_full_data_afterreimplementation.csv"
df = pd.read_csv(path_to_input)


#Plot the deviation of patterns used
quantities_of_pattern = lin_sort(df[:]["pattern_id"])
# plt.bar(range(len(quantities_of_pattern)), list(quantities_of_pattern.values()), align='center')
# plt.show()

#Find the amount occurs per 10 pages
quantity_links = lin_sort(df[:]["site_id"])
quantity_links_dict = dict()
for key in quantity_links:
    if(quantity_links[key] in quantity_links_dict):
        quantity_links_dict[quantity_links[key]] +=1
    else:
        quantity_links_dict[quantity_links[key]] = 1
#Get quantity of results
quantity_of_sites = len(lin_sort(df[:]["site_id"]))
print(quantity_of_sites)
print("*"*70)
print(quantity_links)

#gather working data
#For debugging, doesn't properly devide by patterns
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
print(site_img_dict)  

for site in site_img_dict:
    for img in site_img_dict[site]:
        plt.scatter(site_img_dict[site][img][0], site_img_dict[site][img][1])
plt.show()



