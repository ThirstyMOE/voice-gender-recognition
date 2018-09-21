import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans

"""
    Dataset from https://www.kaggle.com/primaryobjects/voicegender/home
"""

pd.set_option("display.max_columns", 1000)
pd.set_option("display.max_rows", 20)

def data_preprocessing(debug=False):
    source_df = pd.read_csv("voice.csv")

    # X or the input
    input_df = source_df.drop(["label"], axis=1)
    input_df = source_df[["meanfun"]]

    labels_df = source_df[["label"]]

    if debug:
        print(source_df.head())
        print("\n\n\n")
        print(input_df.head())
        print("\n\n\n")
        print(labels_df)

    return input_df, labels_df

def evaluate_clustering(original_labels, cluster_labels):
    original_labels = convert_non_numerical_data(original_labels)

    match_count = 0
    for i in range(len(original_labels)):
        if original_labels[i] == cluster_labels[i]:
            match_count += 1

    match_percentage = 100.0 * match_count / len(original_labels)
    print("Match Accuracy: ", match_percentage)

def convert_non_numerical_data(labels):
    def male_or_female(label):
        if label == "male":
            return 0
        else:
            return 1
    labels = list(map(lambda x : male_or_female(x), labels))
    return labels

input_df, labels_df = data_preprocessing()
clf = KMeans(n_clusters=2)

# input_df = [[0, 1], [0,2], [1,2], [0,3]]  # A dummy dataset
clf.fit(input_df)

clusters_ = clf.cluster_centers_
labels_ = clf.labels_

# print(clusters_)
# print(np.unique(labels_))

evaluate_clustering(labels_df["label"], labels_)
