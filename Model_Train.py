import numpy as np
from sklearn.model_selection import StratifiedKFold as SKFold
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as rfc


def pred_log(logreg, X_train, y_train, X_test, flag=False):
    """

    :param logreg: An object of the class LogisticRegression
    :param X_train: Training set samples
    :param y_train: Training set labels
    :param X_test: Testing set samples
    :param flag: A boolean determining whether to return the predicted probabilities of the classes
    :return: A two elements tuple containing the predictions and the weightning matrix
    """

    logreg.fit(X_train, y_train)
    if flag:
        y_pred_log = logreg.predict_proba(X_test)
    else:
        y_pred_log = logreg.predict(X_test)
    w_log = logreg.coef_

    return y_pred_log, w_log


def LR_cv_kfold(X, y, C, penalty, K):
    """

    :param X: Training set samples
    :param y: Training set labels
    :param C: A list of regularization parameters
    :param penalty: A list of types of norm
    :param K: Number of folds
    :return: A dictionary
    """
    kf = SKFold(n_splits=K)
    validation_dict = []
    for c in C:
        for p in penalty:
            logreg = LogisticRegression(solver='liblinear', penalty=p, C=c, max_iter=10000)
            loss_val_vec = np.zeros(K)
            k = 0
            for train_idx, val_idx in kf.split(X, y):
                x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
                y_val, w = pred_log(logreg, x_train, y[train_idx], x_val, flag=True)
                loss_val_vec[k] = log_loss(y[val_idx], y_val)

            validation_dict.append({'C': c, 'penalty': p, 'mu': np.mean(loss_val_vec), 'sigma': np.std(loss_val_vec)})
    return validation_dict


def LSVM_cv_kfold(X, y, C, gamma, K):
    """

    :param X: Training set samples
    :param y: Training set labels
    :param C: A list of regularization parameters
    :param gamma: A list of kernel coefficient
    :param K: Number of folds
    :return: A dictionary
    """
    kf = SKFold(n_splits=K)
    validation_dict = []
    for c in C:
        for g in gamma:
            model = SVC(kernel='linear', gamma=g, C=c)
            loss_val_vec = np.zeros(K)
            k = 0
            for train_idx, val_idx in kf.split(X, y):
                x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
                model.fit(x_train, y[train_idx])
                y_pred = model.predict(x_val)
                loss_val_vec[k] = log_loss(y[val_idx], y_pred)

            validation_dict.append({'C': c, 'gamma': g, 'mu': np.mean(loss_val_vec), 'sigma': np.std(loss_val_vec)})
    return validation_dict


def RFC_cv_kfold(X, y, n_estimators, K):
    """

    :param X: Training set samples
    :param y: Training set labels
    :param n_estimators: A list of the number of trees in the forest
    :param K: Number of folds
    :return: A dictionary
    """
    kf = SKFold(n_splits=K)
    validation_dict = []
    for estimator in n_estimators:
        clf = rfc(n_estimators=estimator)
        loss_val_vec = np.zeros(K)
        k = 0
        for train_idx, val_idx in kf.split(X, y):
            x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
            clf.fit(x_train, y[train_idx])
            y_pred = clf.predict(x_val)
            loss_val_vec[k] = log_loss(y[val_idx], y_pred)

        validation_dict.append({'n_estimators': estimator, 'mu': np.mean(loss_val_vec), 'sigma': np.std(loss_val_vec)})
    return validation_dict


def NLSVM_cv_kfold(X, y, C, kernels, K):
    """

    :param X: Training set samples
    :param y: Training set labels
    :param C: A list of regularization parameters
    :param kernels: A list of kernels
    :param K: Number of folds
    :return: A dictionary
    """
    kf = SKFold(n_splits=K)
    validation_dict = []
    for c in C:
        for ker in kernels:
            model = SVC(kernel=ker, C=c)
            loss_val_vec = np.zeros(K)
            k = 0
            for train_idx, val_idx in kf.split(X, y):
                x_train, x_val = X.iloc[train_idx], X.iloc[val_idx]
                model.fit(x_train, y[train_idx])
                y_pred = model.predict(x_val)
                loss_val_vec[k] = log_loss(y[val_idx], y_pred)

            validation_dict.append({'C': c, 'Kernel': ker, 'mu': np.mean(loss_val_vec), 'sigma': np.std(loss_val_vec)})
    return validation_dict