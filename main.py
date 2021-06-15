import numpy as np
import pandas as pd
from helper.k_means import fit
from sklearn.cluster import KMeans

emoji_df = pd.read_csv("emoji_data.csv",index_col=0)

emoji_data = np.array(emoji_df)

kmeans = KMeans(n_clusters=15, random_state=0).fit(emoji_data)

print(kmeans.labels_)

print(fit(emoji_data,15))