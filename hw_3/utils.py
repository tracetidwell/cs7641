import itertools
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_curve, roc_auc_score

from scipy.spatial.distance import cdist

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, 
                        train_sizes=np.linspace(.1, 1.0, 5), figsize=(10,8)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure(figsize=figsize)
    #plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_roc_curve(title, labels, predictions, figsize=(10, 8)):
    
    fpr, tpr, thrsh = roc_curve(labels, predictions)
    roc_auc = roc_auc_score(labels, predictions)
    
    plt.figure(figsize=figsize)
    plt.grid()
    plt.plot(fpr, tpr, color='b', label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('{} Receiver Operating Characteristic'.format(title))
    plt.legend()


def load_eye_data():
    df = pd.read_csv('eye_data.txt', sep=',')
    x = df.drop('eye', axis=1)
    y = df[['eye']]
    return x, y

def split_eye_data():
    x, y = load_eye_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=True, random_state=24)
    return x_train, x_test, y_train, y_test

def scale_eye_data():
    x_train, x_test, y_train, y_test = split_eye_data()
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)


def load_bank_data():
    df = pd.read_csv('bank-additional-full.csv', sep=';')
    categorical = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    df = pd.get_dummies(df, columns=categorical, drop_first=True)
    df['y'].replace(['no', 'yes'], [0, 1], inplace=True)
    x = df.drop('y', axis=1)
    y = df[['y']]
    return x, y

def split_bank_data():
    x, y = load_bank_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, shuffle=True, random_state=24)
    return x_train, x_test, y_train, y_test

def elbow_method(x, K_range, distance='euclidean'):
    # k means determine k
    distortions = []
    for k in K_range:
        k_means = KMeans(n_clusters=k).fit(x)
        distortions.append(sum(np.min(cdist(x, k_means.cluster_centers_, distance), axis=1)) / x.shape[0])

    # Plot the elbow
    plt.figure(figsize=(10, 8))
    plt.plot(K_range, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    #plt.title('The Elbow Method showing the optimal k')
    plt.show()


def BIC_scores(x, n_components_range):

    # Number of samples per component
    n_samples = 500

    # Generate random sample, two components
    # np.random.seed(0)
    # C = np.array([[0., -0.1], [1.7, .4]])
    # X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
    #           .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

    lowest_bic = np.infty
    bic = []
    #n_components_range = range(1, 7)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type, max_iter=500)
            gmm.fit(x)
            bic.append(gmm.bic(x))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange'])
    clf = best_gmm
    bars = []

    # Plot the BIC scores
    plt.figure(figsize=(14, 12))
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    #plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)

    # Plot the winner
    # splot = plt.subplot(2, 1, 2)
    # Y_ = clf.predict(x)
    # for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
    #                                            color_iter)):
    #     v, w = linalg.eigh(cov)
    #     if not np.any(Y_ == i):
    #         continue
    #     plt.scatter(x[Y_ == i, 0], x[Y_ == i, 1], .8, color=color)

    #     # Plot an ellipse to show the Gaussian component
    #     angle = np.arctan2(w[0][1], w[0][0])
    #     angle = 180. * angle / np.pi  # convert to degrees
    #     v = 2. * np.sqrt(2.) * np.sqrt(v)
    #     ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
    #     ell.set_clip_box(splot.bbox)
    #     ell.set_alpha(.5)
    #     splot.add_artist(ell)

    # plt.xticks(())
    # plt.yticks(())
    # plt.title('Selected GMM: full model, 2 components')
    # plt.subplots_adjust(hspace=.35, bottom=.02)
    plt.show()
    return bic


def show_class_splits(clusters, y):
    k = len(set(clusters))
    for i in range(k):
        print(sum(y[clusters==i].values == 0), sum(y[clusters==i].values == 1))
        