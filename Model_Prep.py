import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def distribution_table(X_train, X_test, y_train, y_test):
    """
    :param X_train: Pandas series, training dataset of T1D features
    :param X_test: Pandas series, test dataset of T1D features
    :param y_train: Pandas series, train dataset matching predictions
    :param y_test: Pandas series, test dataset matching predictions
    :return: A pandas dataframe showing distribution between each feature label in Train and Test Sets
    """

    table = dict()
    features_list = list(X_train.columns)[1:]
    table['Positive Feature'] = features_list
    table['Train %'] = np.zeros(len(features_list))
    table['Test %'] = np.zeros(len(features_list))
    table['Delta %'] = np.zeros(len(features_list))

    for i, feature in enumerate(features_list):
        x_train = list(X_train[feature])
        x_test = list(X_test[feature])
        for idx in range(len(X_train[feature])):
            if (x_train[idx] == 'Yes') or (x_train[idx] == 1) or (x_train[idx] == 'Male'):
                table['Train %'][i] += 1
        for idx in range(len(X_test[feature])):
            if (x_test[idx] == 'Yes') or (x_test[idx] == 1) or (x_train[idx] == 'Male'):
                table['Test %'][i] += 1

    for i, feature in enumerate(features_list):
        table['Train %'][i] = round((table['Train %'][i]/len(X_train[feature]))*100)
        table['Test %'][i] = round((table['Test %'][i]/len(X_test[feature]))*100)
        table['Delta %'][i] = abs(table['Train %'][i] - table['Test %'][i])
    return pd.DataFrame.from_dict(table)

def feature_label_plots(c_samp):
    """
    :param c_samp: Pandas series, dataset of T1D features
    :return: Pandas dataframe containing the relationship between feature and label
    """

    features_list = list(c_samp.columns)[1:]
    for feature in features_list:
        if feature == 'Diagnosis':
            continue
        feat = dict()
        uniques = np.unique(c_samp[feature])
        for elem in uniques:
            feat[elem] = {}
            feat[elem]['Positive'] = 0
            feat[elem]['Negative'] = 0
            for idx, value in enumerate(c_samp[feature]):
                if c_samp[feature][idx] == elem:
                    if c_samp['Diagnosis'][idx] == 'Positive':
                        feat[elem]['Positive'] += 1
                    else:
                        feat[elem]['Negative'] += 1
        df = pd.DataFrame.from_dict(feat)
        df_trans = df.T
        df_trans.plot.bar(rot=0, title=feature)
        plt.ylabel('Counts')

def encode_data(c_samp):
    """
    :param data: Pandas series, dataset of T1D features (with labels)
    :return: Pandas dataframe of One-Hot vectors (Encoded features)
    """
    encoded_table = c_samp.copy()
    encoded_table = encoded_table.replace('Yes', 1)
    encoded_table = encoded_table.replace('No', 0)
    encoded_table = encoded_table.replace('Male', 1)
    encoded_table = encoded_table.replace('Female', 0)
    encoded_table = encoded_table.replace('Positive', 1)
    encoded_table = encoded_table.replace('Negative', 0)

    unique_ages = np.unique(encoded_table['Age'])
    for elem in unique_ages:
        encoded_table.insert(0, str(elem), 0)
    for idx, age in enumerate(encoded_table['Age']):
        encoded_table[str(age)][idx] = 1
    encoded_table = encoded_table.drop('Age', axis=1)

    encoded_table = encoded_table.astype(int)
    Diagnosis = encoded_table['Diagnosis'].copy()
    encoded_table = encoded_table.drop('Diagnosis', axis=1)

    return encoded_table, Diagnosis