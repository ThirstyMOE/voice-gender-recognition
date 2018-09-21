import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans

"""
    Dataset from https://www.kaggle.com/primaryobjects/voicegender/home

    Using Sci Kit Learn's machine learning libraries to do basic data clustering.
    Pandas for dataset manipulation and preprocessing
    Numpy for numerical computations (Currently only for getting unique values in list)
"""

# Change the display options for displaying the dataframe to the screen
pd.set_option("display.max_columns", 1000)
pd.set_option("display.max_rows", 20)

def data_preprocessing(debug=False):
    """
        param: If debug is true, the heads of the original, featureset, and label data_frames
            will be printed to the console. Else, there will be no output to console
        post: returns two data_frames for the featuresets and labels respectively
    """
    # Read in the original dataset from Kaggle
    source_df = pd.read_csv("voice.csv")

    # X or the featuresets
    input_df = source_df.drop(["label"], axis=1)
    input_df = source_df[["meanfun"]]

    # y or the labels
    labels_df = source_df[["label"]]

    # Debug console printing logic
    if debug:
        print(source_df.head())
        print("\n\n\n")
        print(input_df.head())
        print("\n\n\n")
        print(labels_df)
    # returns the featureset and labels data_frames
    return input_df, labels_df

def evaluate_clustering(original_labels, cluster_labels):
    """
        pre: the two parameters must be some form of 1D array-like
        param: original_labels are a 1D arraylike of the original dataset's label column
            cluster_labels are a 1D arraylike of the model's predicted labels from clusters
        post: prints console output to show the percentage of matches between the two label sets
    """
    # Convert the original "male" "female" labels into 0 and 1
    original_labels = convert_non_numerical_data(original_labels)

    # Keep track of how many matches between labels there are. Based on how clustering is
    # calculated, the match_count will be very high (95%) or very low (5%)
    match_count = 0
    # Calculate the number of matches
    for i in range(len(original_labels)):
        if original_labels[i] == cluster_labels[i]:
            match_count += 1

    # Display the percentage of matches to console
    match_percentage = 100.0 * match_count / len(original_labels)
    print("Match Accuracy: ", match_percentage)

def convert_non_numerical_data(labels):
    """
        pre: the parameter labels must be a 1D arraylike
        param: labels is a 1D arraylike that will be modified and mapped to numerical categories
            instead of string categorical data
        post: Returns list as a 1D arraylike with numerically formatted data
    """
    # Returns a 0 or 1 based on whether a label is male or female respectively
    def male_or_female(label):
        if label == "male":
            return 0
        else:
            return 1
    labels = list(map(lambda x : male_or_female(x), labels))
    return labels

# Preprocess the data into featureset and label data_frames
input_df, labels_df = data_preprocessing()
# Create the clustering model
clf = KMeans(n_clusters=2)

# input_df = [[0, 1], [0,2], [1,2], [0,3]]  # A dummy dataset
# Give the featuresets to the model to train
clf.fit(input_df)

# Get the clusters coordinates back from the model
clusters_ = clf.cluster_centers_
# Get the labels of the clustered data back from the model
labels_ = clf.labels_

# For printing the clusters and label categories to the console
# print(clusters_)
# print(np.unique(labels_))

# Display the percentage of matches to the console
evaluate_clustering(labels_df["label"], labels_)
