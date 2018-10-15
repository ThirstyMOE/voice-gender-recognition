import pandas as pd
from sklearn.svm import SVC

"""
    Dataset from https://www.kaggle.com/primaryobjects/voicegender/home

    Using Sci Kit Learn's machine learning libraries to do basic binary classification.
    Pandas for dataset manipulation and preprocessing
    Numpy for numerical computations (Currently only for getting unique values in list)
"""

percent_training = 0.8

# Change the display options for displaying the dataframe to the screen
pd.set_option("display.max_columns", 1000)
pd.set_option("display.max_rows", 20)

def data_preprocessing():
    """
        post: Returns the training features, labels, and test features, labels in the form of
            pandas data_frames, from the Kaggle dataset as a source
    """
    source_df = pd.read_csv("voice.csv")

    # Accessed at https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows on 9-21-2018
    source_df = source_df.sample(frac=1).reset_index(drop=True)

    training_test_split_index = round(percent_training * len(source_df))

    training_source_df = source_df[0:training_test_split_index]
    test_source_df = source_df[training_test_split_index:len(source_df)]

    # TODO: Can be condensed somehow
    training_feature_df = training_source_df.drop(["label"], axis=1)
    test_feature_df = test_source_df.drop(["label"], axis=1)
    training_feature_df = training_feature_df[["meanfun", "meanfreq"]]  # Do not do kurt
    test_feature_df = test_feature_df[["meanfun", "meanfreq"]]

    training_label_df = training_source_df[["label"]]
    test_label_df = test_source_df[["label"]]

    return training_feature_df, training_label_df["label"], test_feature_df, test_label_df["label"]

def evaluate_model(classifier, test_features_df, test_labels_df):
    test_labels_array = test_label_df.values
    predictions = classifier.predict(test_features_df)
    correct = 0
    for index in range(len(test_labels_array)):
        if predictions[index] == test_labels_array[index]:
            correct += 1
    print("Accuracy", 100.0 * correct / len(test_labels_df))

training_feature_df, training_label_df, test_feature_df, test_label_df = data_preprocessing()

# print(training_feature_df)
# print("\n\n\n")
# print(test_feature_df)
# print("\n\n\n")
# print(training_label_df)
# print("\n\n\n")
# print(test_label_df)

clf = SVC()
clf.fit(training_feature_df, training_label_df)
evaluate_model(clf, test_feature_df, test_label_df)
