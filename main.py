import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import datasets


# TODO рисовать центры кластеров, перед каждым расчетом новых центров рисовать график

def predict(x, mu):
    r = []
    for c in mu:
        r.append(np.sqrt(np.sum(np.square(c - x))))

    return np.argmin(r)


def cluster_points(X, mu):
    clusters = {}
    for x in X:
        cluster_num = predict(x, mu)
        try:
            clusters[cluster_num].append(x)
        except KeyError:
            clusters[cluster_num] = [x]

    return clusters


def has_converged(mu, oldmu):
    return set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])


def reevaluate_centers(clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis=0))
    return newmu


def find_centers(X, k):
    # Initialize to K random centers
    oldmu = []
    mu = [X[c] for c in np.random.randint(X.shape[0], size=k)]
    clusters = {}
    while not has_converged(mu, oldmu):
        oldmu = mu
        clusters = cluster_points(X, mu)
        mu = reevaluate_centers(clusters)
    return mu, clusters


def main(X, k):
    u, clusters = find_centers(X, k)
    data1 = pd.DataFrame(data=X)
    pred = []
    for i in X:
        pred.append(predict(i, u))

    data1['target'] = pd.Series(pred, index=data1.index)

    g = sns.FacetGrid(data1, hue='target', palette="Set1", size=5)
    g.map(plt.scatter, 0, 1, s=100, linewidth=.5, edgecolor="white")
    g.add_legend()
    plt.show()


if __name__ == '__main__':
    X, Y = datasets.make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=1,
                                        n_clusters_per_class=1)
    main(X, 6)

    X, Y = datasets.make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=2,
                                        n_clusters_per_class=1)
    main(X, 3)

    X, Y = datasets.make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=2)
    main(X, 2)

    X, Y = datasets.make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=2,
                                        n_clusters_per_class=1, n_classes=3)
    main(X, 3)

    X, Y = datasets.make_blobs(n_samples=500, n_features=2, centers=5)
    main(X, 5)

    X, Y = datasets.make_gaussian_quantiles(n_samples=500, n_features=2, n_classes=6)
    main(X, 6)
