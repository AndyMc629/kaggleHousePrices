# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
# Function to normalise data columns to range (0,1).
# takes a pandas dataframe as input, returns pd df
# output.


def normalise(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].astype(np.float32).max()
        min_value = df[feature_name].astype(np.float32).min()
        # only normalise if values are non-zero
        if max_value != 0 or min_value != 0:
            result[feature_name] = (
                    df[feature_name].astype(np.float32) - min_value
                    )/(max_value - min_value)
    return result

# general function to find indicex where condition is true.


def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]

# function to train the data and output Coefficient of det on train.
# clf=sklearn model object.
# Pass train dataframe in order to retrieve correct labels,
# function will return plot coefficients for all features with
# coefficients greater than importance threshold
# also pass target variable now to make it more general.


def train_and_evaluate(clf, X_train, y_train, train_df,
                       importance_threshold, target_variable):
    # perform training
    clf.fit(X_train, y_train)
    # print R^2
    print "Coefficient of determination on training set:",
    clf.score(X_train, y_train)
    # get labels, exclude target variable
    labels = list((train_df.drop(target_variable, axis=1)).columns.values)
    # make dict of features and coeff values
    coeffs = dict(zip(labels, clf.coef_))
    # trim the coeff dict, based on importance threshold
    trimmed_coeffs = {key: value for key,
                      value in coeffs.iteritems()
                      if (abs(value) > importance_threshold)}

    print "Trimmed =", trimmed_coeffs

    plt.figure()
    plt.bar(range(len(trimmed_coeffs)),
            trimmed_coeffs.values(),
            align='center')
    plt.xticks(range(len(trimmed_coeffs)), trimmed_coeffs.keys(), rotation=45)
    plt.tight_layout()
    plt.savefig("coeffs.pdf")
    plt.close()
