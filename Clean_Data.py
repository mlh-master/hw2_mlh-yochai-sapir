"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_typeo_and_miss(T1D_features):
    """
    :param T1D_features: Pandas series of T1D features
    :return: A pandas dataframe containing the "clean" features
    """

    c_t1d = T1D_features.copy()  # copy the dataframe
    columns_list = list(c_t1d.columns)
    for column in columns_list:
        if column == 'Age' or column == 'Family History':
            c_t1d[column] = pd.to_numeric(c_t1d[column], errors='coerce', downcast='float')
        elif column == 'Gender':
            for i, val in enumerate(c_t1d[column]):
                if val != 'Male' and val != 'Female':
                    c_t1d[column][i] = np.nan
        elif column == 'Diagnosis':
            for i, val in enumerate(c_t1d[column]):
                if val != 'Negative' and val != 'Positive':
                    c_t1d[column][i] = np.nan
        else:
            for i, val in enumerate(c_t1d[column]):
                if val != 'Yes' and val != 'No':
                    c_t1d[column][i] = np.nan
    return c_t1d


def nan2value_samp(c_t1d):
    """

    :param c_t1d: Pandas series of T1D features with missing values
    :return: A pandas dataframe containing the "clean" features
    """

    columns_list = list(c_t1d.columns)
    for column in columns_list:
        none_nan = c_t1d[column].dropna()
        freq = dict()
        for element in none_nan:
            if element in freq:
                freq[element] += 1
            else:
                freq[element] = 1

        for number in freq.keys():
            freq[number] = freq[number] / len(none_nan)
        elements = list(freq.keys())
        probs = list(freq.values())

        Q = pd.DataFrame(c_t1d[column])
        idx_na = Q.index[Q[column].isna()].tolist()
        for i in idx_na:
            Q.loc[i] = np.random.choice(elements, 1, probs)[0]
        c_t1d[column] = Q
    return c_t1d