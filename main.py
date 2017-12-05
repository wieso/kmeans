import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import datasets

from dist_metrics import dist_metrics


def predict(x, mu, **kwargs):
    dist_metric = kwargs['dist_metric']
    if dist_metric != 'power':
        r = dist_metrics[dist_metric](x, mu)
    else:
        r = dist_metrics[dist_metric](x, mu, kwargs['power_root'], kwargs['power_power'])

    return np.argmin(r)


def cluster_points(X, mu, **kwargs):
    clusters = {}
    for x in X:
        cluster_num = predict(x, mu, dist_metric=kwargs['dist_metric'], )
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


def find_centers(X, k, **kwargs):
    # Initialize to K random centers
    oldmu = []
    mu = [X[c] for c in np.random.randint(X.shape[0], size=k)]
    clusters = {}
    while not has_converged(mu, oldmu):
        oldmu = mu
        clusters = cluster_points(X, mu, **kwargs)
        mu = reevaluate_centers(clusters)
        if kwargs['step_plot']:
            plot_data_with_centers(X, mu, **kwargs)

    return mu, clusters


def plot_data_with_centers(data, centers, **kwargs):
    data1 = pd.DataFrame(data=data)
    pred = []
    for i in data:
        pred.append(predict(i, centers, **kwargs))

    data1['target'] = pd.Series(pred, index=data1.index)

    g = sns.FacetGrid(data1, hue='target', palette="Set1", size=5)
    g.map(plt.scatter, 0, 1, s=100, linewidth=.5, edgecolor="white")
    # for ax in g.axes.flat:
    #     for center in centers:
    #         ax.plot(center[0], center[1], 'kv', markersize=15)
    g.add_legend()
    plt.show()


def main(X, k, dist_metric='euclid', power_root=2, power_power=2, step_plot=True, **kwargs):
    """
    Parameters
    ----------
    X: list
        Dataset
    k: int
        Number of centers
    dist_metric: {'euclid', 'euclid_square', 'manhattan', 'chebyshev', 'power'}
        Distance metric
    power_root: int
        Root for 'power' metric
    power_power: int
        Power for 'power' metric
    step_plot: bool
        Plot for an every iteration
    """
    kwargs['dist_metric'] = dist_metric
    kwargs['power_root'] = power_root
    kwargs['power_power'] = power_power
    kwargs['step_plot'] = step_plot
    u, clusters = find_centers(X, k, **kwargs)
    plot_data_with_centers(X, u, **kwargs)
    # data1 = pd.DataFrame(data=X)
    # pred = []
    # for i in X:
    #     pred.append(predict(i, u))
    #
    # data1['target'] = pd.Series(pred, index=data1.index)
    #
    # g = sns.FacetGrid(data1, hue='target', palette="Set1", size=5)
    # g.map(plt.scatter, 0, 1, s=100, linewidth=.5, edgecolor="white")
    # for ax in g.axes.flat:
    #     for center in u:
    #         ax.plot(center[0], center[1], 'kv', markersize=15)
    # g.add_legend()
    # plt.show()


if __name__ == '__main__':
    n_samples = 5000

    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8, center_box=(-2000, 2000), cluster_std=50)
    main(blobs[0], 3, step_plot=False)

    # noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
    #                                       noise=.05)
    #
    # main(noisy_circles[0], 2)
    #
    # noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    #
    # main(noisy_moons[0], 2)
    #
    # blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    # main(blobs[0], 3)
    #
    # no_structure = np.random.rand(n_samples, 2), None
    # main(no_structure[0], 2)
    #
    # # Anisotropicly distributed data
    # random_state = 170
    # X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    # transformation = [[0.6, -0.6], [-0.4, 0.8]]
    # X_aniso = np.dot(X, transformation)
    # aniso = (X_aniso, y)
    #
    # main(X_aniso, 2)
    #
    # # blobs with varied variances
    # varied = datasets.make_blobs(n_samples=n_samples,
    #                              cluster_std=[1.0, 2.5, 0.5],
    #                              random_state=random_state)
    # main(varied[0], 2)

    # X, Y = datasets.make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=1,
    #                                     n_clusters_per_class=1)
    # main(X, 2)
    # 
    # X, Y = datasets.make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=2,
    #                                     n_clusters_per_class=1)
    # main(X, 3)
    # 
    # X, Y = datasets.make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=2)
    # main(X, 2)
    # 
    # X, Y = datasets.make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=2,
    #                                     n_clusters_per_class=1, n_classes=3)
    # main(X, 3)
    # 
    # X, Y = datasets.make_blobs(n_samples=500, n_features=2, centers=5)
    # main(X, 4, dist_metric='manhattan')

    # X, Y = datasets.make_gaussian_quantiles(n_samples=500, n_features=2, n_classes=6)
    # main(X, 6)
    #
    # X, Y = datasets.make_circles(n_samples=500, factor=0.5)
    # main(X, 4, dist_metric='manhattan')
