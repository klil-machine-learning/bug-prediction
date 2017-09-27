# utility function for hexbug challenge
import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from matplotlib import patches
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf

def rmse(preds, y):
    return np.sqrt(np.mean((preds-y)**2))

def regressor(model, X, y, X_test, y_test, debug=True):
    t0 = time.time()
    model.fit(X, y)
    score = rmse(model.predict(X), y)
    test_score = rmse(model.predict(X_test), y_test)
    t1 = time.time()
    if debug:
        print("Model trained ({} seconds). \nTrain score: {}, test score: {}".format(t1-t0, score, test_score))
    return model, test_score


def nn_run(layer_sizes, X_train, X_test, y_train, y_test, epochs=1500, debug=False):
    num_input_params = X_train.shape[1]
    num_output_params = y_train.shape[1]

    X = tf.placeholder(tf.float32, shape=[None, num_input_params])
    y = tf.placeholder(tf.float32, shape=[None, num_output_params])

    prev_layer = X
    for layer_size in layer_sizes:
        layer = tf.contrib.layers.fully_connected(prev_layer, layer_size)
        prev_layer = layer
    hypothesis = tf.contrib.layers.fully_connected(prev_layer, num_output_params)

    # rmse
    cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(hypothesis, y))))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for step in range(epochs):
        opt = sess.run(optimizer, feed_dict={X: X_train, y: y_train})
        if debug and step % 200 == 0:
            print("========== step ", step)
            print("train cost: ", sess.run(cost, feed_dict={X: X_train, y: y_train}))
            print("test cost: ", sess.run(cost, feed_dict={X: X_test, y: y_test}))
    return sess.run(cost, feed_dict={X: X_test, y: y_test})

def multi_run(prms_dic, debug=False):
    prms_dic["rmse_results"] = np.zeros((len(prms_dic["n_trains"]), 
                                         len(prms_dic["n_labels"]), 
                                         len(prms_dic["models"])))
#     prms_dic["score_df"] =  np.zeros((len(prms_dic["n_trains"]), 
#                                          len(prms_dic["n_labels"]), 
#                                          len(prms_dic["models"])))
    for i1, n_train in enumerate(prms_dic["n_trains"]):
        for i2, n_label in enumerate(prms_dic["n_labels"]):
            if prms_dic["data_sampler"] == "random":
                train_all, label_all = split_all_videos_random(prms_dic["n_samples"], n_train, n_label, 
                                                                   prms_dic["train_features"], 
                                                                   prms_dic["label_features"], 
                                                                   debug=debug)
            if prms_dic["data_sampler"] == "ordered":
                train_all, label_all = split_all_videos(n_train, n_label, 
                                                            prms_dic["train_features"], 
                                                            prms_dic["label_features"], 
                                                            debug=debug)
                prms_dic["n_samples"] = len(train_all)
            for i3, model in enumerate(prms_dic["models"]):
                X_train, X_test, y_train, y_test = train_test_split(train_all, label_all, test_size=0.2)
    #             print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
                if model == "nn":
                    score = nn_run(prms_dic["fc_nn_sizes"], X_train, X_test, y_train, y_test, debug=debug)
                else:
                    _, score = regressor(model, X_train, y_train, X_test, y_test, debug=debug)
                # summarize results
                print("score: {}, model: {}, n_samples: {}, n_train: {}, n_label: {}".format(
                    score.mean(), type(model).__name__, prms_dic["n_samples"], n_train, n_label))
#                 prms_dic["score_df"][i1, i2, i3] = score
                prms_dic["rmse_results"][i1, i2, i3] = score.mean()
    return prms_dic

def single_run(prms_dic, debug=False):
    if prms_dic["data_sampler"] == "random":
        train_all, label_all = split_all_videos_random(prms_dic["n_samples"], prms_dic["n_trains"], prms_dic["n_labels"], 
                                                           prms_dic["train_features"], 
                                                           prms_dic["label_features"], 
                                                           debug=debug)
    if prms_dic["data_sampler"] == "ordered":
        train_all, label_all = split_all_videos(prms_dic["n_trains"], prms_dic["n_labels"], 
                                                    prms_dic["train_features"], 
                                                    prms_dic["label_features"], 
                                                    debug=debug)
        prms_dic["n_samples"] = len(train_all)
    X_train, X_test, y_train, y_test = train_test_split(train_all, label_all, test_size=0.2)
#             print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    model, score = regressor(prms_dic["models"], X_train, y_train, X_test, y_test, debug=debug)
    # summarize results
    print("score: {}, model: {}, n_samples: {}, n_train: {}, n_label: {}".format(
        score.mean(), type(model).__name__, prms_dic["n_samples"], prms_dic["n_trains"], prms_dic["n_labels"]))
#                 prms_dic["score_df"][i1, i2, i3] = score
    prms_dic["rmse_results"] = score.mean()
    return model, prms_dic

def show_video_trajectory(video_file):
    df = pd.read_csv("features/" + video_file)
    df = df.set_index('frame number')
    plt.plot(df.C_x.values, df.C_y.values,'-')
    plt.plot(df.A_x.values, df.A_y.values,'-r')
    plt.plot(df.B_x.values, df.B_y.values,'-g')
    # plot surrounding box
    plt.plot([125, 728, 728, 125, 125],[16, 16, 452, 452, 16], 'k-')
    # plot mid circle
    circ = plt.Circle((433,236),72,color='k', fill=False)
    plt.gca().add_patch(circ)
    return plt.gcf()


def show_sample(train, label, ind=0):
    fig, ax = plt.subplots(figsize=(7,5))
    train_xs = train.iloc[ind].loc[:, 'C_x'].values
    train_ys = train.iloc[ind].loc[:, 'C_y'].values
    label_xs = label.iloc[ind].loc[:, 'C_x'].values
    label_ys = label.iloc[ind].loc[:, 'C_y'].values
    ax.plot(train_xs, train_ys, 'b.', label_xs, label_ys, 'go')
    plt.gca().plot()
    # plot surrounding box
    plt.gca().plot([125, 728, 728, 125, 125],[16, 16, 452, 452, 16], 'k-')
    # plot mid circle
    plt.gca().add_patch(plt.Circle((433,236),72,color='k', fill=False))
    plt.show()


def show_pred(train, label, preds, ind=0):
    fig, ax = plt.subplots(figsize=(7,5))
    train_xs = train.iloc[ind].loc[:, 'C_x'].values
    train_ys = train.iloc[ind].loc[:, 'C_y'].values
    label_xs = label.iloc[ind].loc[:, 'C_x'].values
    label_ys = label.iloc[ind].loc[:, 'C_y'].values
    preds_xs = preds.iloc[ind].loc[:, 'C_x'].values
    preds_ys = preds.iloc[ind].loc[:, 'C_y'].values
    ax.plot(train_xs, train_ys, 'b.', label_xs, label_ys, 'go', preds_xs, preds_ys, 'ro')
    plt.gca().plot()
    # plot surrounding box
    plt.gca().plot([125, 728, 728, 125, 125],[16, 16, 452, 452, 16], 'k-')
    # plot mid circle
    plt.gca().add_patch(plt.Circle((433,236),72,color='k', fill=False))


def fetch_one_sample(df, ind, n_train_frames, n_label_frames, train_features, label_features):
    train = df.iloc[ind:ind + n_train_frames].loc[:, train_features].values.reshape(1, (n_train_frames) * len(
        train_features))
    labels = df.iloc[ind + n_train_frames:ind + n_train_frames + n_label_frames].loc[:, label_features].values.reshape(
        1, (n_label_frames) * len(label_features))
    return train, labels


def split_all_videos(n_train_frames, n_label_frames,
                     train_features = ['C_x', 'C_y'],
                     label_features = ['C_x', 'C_y'], debug=True):
    train_all = None
    label_all = None
    for filename in os.listdir('features'):

        if filename.endswith(".csv"):
            df = pd.read_csv("features/" + filename)
            df = df.reset_index(drop=True)
            total_len = n_train_frames + n_label_frames
            length = df.shape[0]
            index = 0
            while index + total_len < length:
                train, label = fetch_one_sample(df, index, n_train_frames, n_label_frames, train_features,
                                                label_features)
                if train_all is None:
                    train_all = train
                    label_all = label
                else:
                    train_all = np.vstack((train_all, train))
                    label_all = np.vstack((label_all, label))
                # print("index {}".format(index))
                index += total_len

    train_index = pd.MultiIndex.from_product([range(n_train_frames), train_features])
    train_df = pd.DataFrame(train_all, columns=train_index)

    label_index = pd.MultiIndex.from_product([range(n_label_frames), label_features])
    label_df = pd.DataFrame(label_all, columns=label_index)

    if debug:
        print("train_df {} label_df {}".format(train_df.shape, label_df.shape))
    return train_df, label_df


def split_all_videos_random(num_items, n_train_frames, n_label_frames,
                            train_features = ['C_x', 'C_y'],
                            label_features = ['C_x', 'C_y'], debug=True):
    train_all = None
    label_all = None
    dfs = []
    total_len = n_train_frames + n_label_frames
    for filename in os.listdir('features'):
        if filename.endswith(".csv"):
            df = pd.read_csv("features/" + filename)
            dfs.append(df)
    for i in range(num_items):
        df = random.choice(dfs)
        len = df.shape[0]
        index = random.randint(0, len - total_len)
        train, label = fetch_one_sample(df, index, n_train_frames, n_label_frames, train_features, label_features)
        if train_all is None:
            train_all = train
            label_all = label
        else:
            train_all = np.vstack((train_all, train))
            label_all = np.vstack((label_all, label))

    train_index = pd.MultiIndex.from_product([range(n_train_frames), train_features])
    train_df = pd.DataFrame(train_all, columns=train_index)

    label_index = pd.MultiIndex.from_product([range(n_label_frames), label_features])
    label_df = pd.DataFrame(label_all, columns=label_index)

    if debug:
        print("train_df {} label_df {}".format(train_df.shape, label_df.shape))
    return train_df, label_df


