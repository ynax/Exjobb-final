'''
    To import user functions
'''
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import pandas as pd
import numpy as np

import seaborn as sns


def read_data(name, folder='Data'):
    try:
        df = pd.read_csv('{}/{}.csv'.format(folder, name)).drop(['Unnamed: 0'], axis=1)
    except:
        try:
            df = pd.read_csv('{}/{}.csv'.format(folder, name), decimal=',', sep=';').drop(['Unnamed: 0'], axis=1)
        except:
            df = pd.read_excel('{}/{}.xlsx'.format(folder, name)).drop(['Unnamed: 0'], axis=1)
    return df


def get_sets(df, class_col='Class'):
    target = df.pop(class_col)
    return df, target


def scale_data(X, scaler='MinMax'):
    if scaler == 'MinMax':
        scl = MinMaxScaler(feature_range=(-1, 1))
    elif scaler == 'Standard':
        scl = StandardScaler()
    Xnew = scl.fit_transform(X)
    return pd.DataFrame(Xnew), scl


def Kfold_split_data(X, Y, k=5, shuffle=False):
    kf = KFold(n_splits=k, random_state=None, shuffle=shuffle)

    Ksets = dict()
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        Ksets[i] = {
            'X train': X.loc[train_index],
            'Y train': Y.loc[train_index],
            'X test': X.loc[test_index],
            'Y test': Y.loc[test_index],
        }
    return Ksets


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def tsne_plot(x1, y1, save_name="graph.png",
              title='tSNE',
              save=False, show=True, verbose=False,
              alpha_F=0.8, alpha_G=0.8,
              marker_F='o', marker_G='o'
              ):

    tsne = TSNE(n_components=2, random_state=0, verbose=verbose)
    X_t = tsne.fit_transform(x1)

    fig = plt.figure(figsize=(12, 8))
    plt.scatter(X_t[np.where(y1 == 0), 0], X_t[np.where(y1 == 0), 1], marker=marker_G, color='g', linewidth='1', alpha=alpha_G, label='Non Fraud')
    plt.scatter(X_t[np.where(y1 == 1), 0], X_t[np.where(y1 == 1), 1], marker=marker_F, color='r', linewidth='1', alpha=alpha_F, label='Fraud')

    plt.legend(loc='best')
    plt.title(title)
    plt.savefig(save_name) if save else None
    plt.show() if show else None
    return fig


def evaluate_features(X, Y,
                      n_jobs=1,
                      evaluator='Decision',
                      criterion='entropy',
                      n_estimators=500,
                      min_feats=None,
                      max_feats=None,
                      rng=0):

    if evaluator == 'Extra':
        clf = ExtraTreesClassifier(n_estimators=n_estimators, criterion=criterion, random_state=rng, n_jobs=n_jobs)
    elif evaluator == 'Decision':
        clf = DecisionTreeClassifier(criterion=criterion)
    elif evaluator == 'Random':
        clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, random_state=rng, n_jobs=n_jobs)

    clf = clf.fit(X, Y)
    importance_listy = clf.feature_importances_

    column_order = np.arange(0, len(importance_listy))
    column_order = list(reversed([x for _, x in sorted(zip(importance_listy, column_order))]))
    return column_order, list(reversed(sorted(importance_listy)))


def plot_feature_importance(columns, importance,
                            #                           x='Feature',
                            #                           y='Importance Weight',
                            title='Decision Tree',
                            xlabel='Features',
                            ylabel='Importance Weights',
                            title_fontsize=20,
                            x_fontsize=20,
                            y_fontsize=20,
                            show=True,
                            x_tick_size=20,
                            y_tick_size=20,
                            sort=True,
                            ascending=False,
                            ax=None,
                            ):
    df = pd.DataFrame(columns=['X', 'y'])
    df['X'] = columns
    df['y'] = importance
    fig, ax1 = plt.subplots(figsize=[1 / 2 * df.shape[0], 8])
    if sort:
        df = df.sort_values(['y'], ascending=ascending).reset_index(drop=True)
        ax = sns.barplot(x=df.index, y=df['y'], ax=ax)
        ax.set_xticklabels(df['X'])
    else:
        ax = sns.barplot(x=df['X'], y=df['y'], ax=ax)
    ax.set_title(
        title,
        fontsize=title_fontsize  # title font size
    )

    ax.set_xlabel(xlabel, fontsize=x_fontsize)
    ax.set_ylabel(ylabel, fontsize=y_fontsize)

    ax.tick_params(axis='x', labelsize=x_tick_size)
    ax.tick_params(axis='y', labelsize=y_tick_size)
#     axs.append(ax)
    if show:
        plt.show()
    return fig, ax


def getMetrics(Y_real, Y_pred, flip_predict=False):
    def add2Series(series, name, val):
        series[name] = val

    if flip_predict:
        Y_pred = 1 - Y_pred
    low_values_flags = Y_pred < 0  # Where values are low
    Y_pred[low_values_flags] = 0
    metricSeries = pd.Series()

    balanced_accuracy = balanced_accuracy_score(Y_real, Y_pred)
    add2Series(metricSeries, 'Balanced Accuracy', balanced_accuracy)

    MCC = matthews_corrcoef(Y_real, Y_pred)
    add2Series(metricSeries, 'MCC', MCC)

    acc = accuracy_score(Y_real, Y_pred)
    add2Series(metricSeries, 'Accuracy', acc)

    f1 = f1_score(Y_real, Y_pred, labels=list(set(Y_pred)))
    add2Series(metricSeries, 'F1-Score', f1)

    precision = precision_score(Y_real, Y_pred, labels=list(set(Y_pred)))
    add2Series(metricSeries, 'Precision', precision)

    recall = recall_score(Y_real, Y_pred)
    add2Series(metricSeries, 'Recall', recall)

    tn, fp, fn, tp = confusion_matrix(Y_real, Y_pred).ravel()

    add2Series(metricSeries, 'True Negatives', tn)
    add2Series(metricSeries, 'False Positives', fp)
    add2Series(metricSeries, 'False Negatives', fn)
    add2Series(metricSeries, 'True Positives', tp)

    return metricSeries


def plot_boxchart(run_df,
                  x='No. Features',
                  y='MCC',
                  hue=None,
                  title=None,
                  xlabel='No. Features Used',
                  ylabel='MCC',
                  title_fontsize=20,
                  x_fontsize=20,
                  y_fontsize=20,
                  show=True,
                  x_tick_size=20,
                  y_tick_size=20,
                  ):
    #     fig, ax = plt.subplots(figsize=[1/4*run_df.shape[0],10])
    fig, ax = plt.subplots(figsize=[20, 10])
    ax = sns.boxplot(x=x,
                     y=y,
                     data=run_df,
                     hue=hue,
                     )

    ax.set_title(
        title,
        #         xlabel=xlabel,
        fontsize=title_fontsize  # title font size
    )

    ax.set_xlabel(xlabel, fontsize=x_fontsize)
    ax.set_ylabel(ylabel, fontsize=y_fontsize)

    ax.tick_params(axis='x', labelsize=x_tick_size)
    ax.tick_params(axis='y', labelsize=y_tick_size)
#     axs.append(ax)
    if show:
        plt.show()
    return fig, ax
