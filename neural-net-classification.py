import pandas as pd
import numpy as np
import tensorflow as tf

'''
    neural-net-classification

    Uses tensorflow for creating a model for binary classification on the voice.csv dataset.
    Based off of model used in Sentdex's Deep Learning series
'''

# How much of the data should be training?
percent_training = 0.8

train_x, train_y, test_x, test_y = data_preprocessing()

# Number of dimensions (features) in featureset
n_input_dimension = 0

# How many nodes in hidden layer x?
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# Number of classes in the labelled dataset
n_classes = 2

# Train across the dataset for how many epochs?
n_epochs = 10
batch_size = 100

x = tf.placeholder(dtype="float", shape=[1, 2])  # TODO: Is this even the right shape?
y = tf.placeholder(dtype=tf.float32, shape=None)  # TODO: should we use tf.float32 or "float"?


def data_preprocessing():
    '''
        takes in data from the local voice.csv file to extract featuresets and labels
            and does the training test data split.
        post: returns training inputs, training labels, test inputs, test labels all as python lists
            and has pandas dataframes inside.
    '''

    original_df = pd.read_csv("voice.csv")
    # Accessed at https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows on 9-21-2018
    original_df = source_df.sample(frac=1).reset_index(drop=True)  # Shuffle rows

    input_df = original_df[["meanfreq"], ["meanfun"]]
    label_df = original_df[["label"]]

    n_input_dimension = len(original_df)  # TODO: Right dimension? I need n space here, not m

    training_test_split_index = round(percent_training * len(original_df))

    training_input_df = input_df[ : training_test_split_index]
    test_input_df = input_df[training_test_split_index : len(original_df)]

    training_label_df = label_df[ : training_test_split_index]
    test_label_df = label_df[training_test_split_index : len(original_df)]

    # https://medium.com/when-i-work-data/converting-a-pandas-dataframe-into-a-tensorflow-dataset-752f3783c168
    # https://www.tensorflow.org/api_docs/python/tf/estimator/inputs/pandas_input_fn
    train_x = list(training_input_df)
    train_y = list(training_label_df)

    test_x = list(test_input_df)
    test_y = list(test_label_df)

    return train_x, train_y, test_x, test_y


def make_neural_network_model(data):
    '''
        pre: takes in data, a input tensor-vector in the dimension of n_input_dimension
        post: returns the output layer of the model from the tf computation graph
    '''

    hidden_layer_1 = {
        "weights" : tf.Variable(tf.random_normal([n_input_dimension, n_nodes_hl1])),
        "biases" : tf.Variable(tf.random_normal([n_nodes_hl1]))
    }
    hidden_layer_2 = {
        "weights" : tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
        "biases" : tf.Variable(tf.random_normal([n_nodes_hl2]))
    }
    hidden_layer_3 = {
        "weights" : tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
        "biases" : tf.Variable(tf.random_normal([n_nodes_hl3]))
    }
    output_layer = {
        "weights" : tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
        "biases" : tf.Variable(tf.random_normal([n_classes]))
    }

    # Make computation graph for hidden layers
    layer1 = tf.add(tf.matmul(data, hidden_layer_1["weights"]), hidden_layer_1["biases"])
    layer1 = tf.nn.relu(layer1)

    layer2 = tf.add(tf.matmul(data, hidden_layer_2["weights"]), hidden_layer_2["biases"])
    layer2 = tf.nn.relu(layer2)

    layer3 = tf.add(tf.matmul(data, hidden_layer_3["weights"]), hidden_layer_3["biases"])
    layer3 = tf.nn.relu(layer3)

    output = tf.add(tf.matmul(data, output_layer["weights"]), output_layer["biases"])  # TODO: regular addition??

    return output


def train_neural_network(x):
    '''
        pre: Take in an input tf.placeholder tensor
        post: executes all computations necessary to construct and train neural network.
            The model will be saved in checkpoints with prefix: "neural_net.ckpt"
    '''

    prediction = make_neural_network_model(x)

    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimization_function = tf.train.AdamOptimizer().minimize(cost_function)

    # Create a operation set to save checkpoints of the model's tf variables https://www.tensorflow.org/guide/saved_model
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                # Take batch of data
                start = i
                end = start + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                # Feed in batched tensors to the placeholder tensors from the top
                _, c = sess.run([optimizer, cost], feed_dict={x : batch_x, y : batch_y})
                # Add the cost calculated to the epoch_loss
                epoch_loss += c
                # Move onto next batch of data
                # TODO: maybe some buffered reading in order to scale back on RAM used
                i += batch_size
            # Print out the epoch loss metrics
            print("EPOCH", epoch + 1, "out of", n_epochs, " --  LOSS:", epoch_loss)
            # Save model progress https://www.tensorflow.org/guide/saved_model
            save_path = saver.save(sess, "/tmp/neural_net.ckpt")
            print("Model saved in path: %s" % save_path)

        # Evaluation metrics
        # returns bool tensor for (x == y)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        # returns accuracy tensor of averaged correct tensor
        accuracy = tf.reduce_mean(tf.case(correct, "float"))
        # run the tensor computations in a default session
        print("Accuracy:", accuracy.eval(feed_dict={x : test_x, y : test_y}))

# Run training method with the x placeholder tensor
train_neural_network(x)
