from sklearn.cluster import KMeans
import numpy as np


class MultiKmeans:
    """
    Implements a multi round k-means by calling k-means successively with different initializations.
    """

    def __init__(self, *args, rounds=1, **kwargs):
        self.rounds = rounds
        # the first kmeans is always initialized with kmeans++, the others are random
        inits = ['k-means++'] + ['random'] * (rounds - 1)
        self.kmeans = [KMeans(*args, **kwargs, init=inits[r], random_state=r) for r in range(rounds)]

    def fit(self, *args, **kwargs):
        self.kmeans = [km.fit(*args, **kwargs) for km in self.kmeans]
        return self

    @property
    def cluster_centers_(self):
        return [km.cluster_centers_ for km in self.kmeans]

    def predict(self, *args, top_clusters=1, **kwargs):
        if top_clusters == 1:
            return np.stack([km.predict(*args, **kwargs) for km in self.kmeans])
        else:
            # requires alternative format: 1 round per cluster,
            # each round only has negative values (ignored) and "current_cluster"
            k_clusters = self.kmeans[0].n_clusters
            sample_count = args[0].shape[0]
            rounds = []
            for r in range(self.rounds):
                topk = self.kmeans[r].transform(args[0]).argsort(-1)[:, :top_clusters, np.newaxis]
                k_clusters_range = np.array([np.array(range(k_clusters))]
                                            * sample_count * top_clusters).reshape((sample_count, top_clusters, k_clusters))
                is_in_bucket = (topk == k_clusters_range).sum(axis=1)
                idx_if_in_bucket = is_in_bucket * k_clusters_range[:, 0, :]

                # initialize with unique negative indexes, will be ignored
                per_cluster = -1 * np.arange(1, (sample_count * k_clusters + 1), dtype=int).reshape((sample_count, k_clusters))
                not_in_bucket = np.logical_not(is_in_bucket)
                per_cluster = per_cluster * not_in_bucket  # set to zero, if in bucket
                per_cluster = (idx_if_in_bucket + per_cluster).transpose((1, 0))
                rounds.append(per_cluster)
            return np.concatenate(rounds)  # shape = (top_clusters * self.rounds, samples)

    def score(self, *args, **kwargs):
        return np.stack([km.score(*args, **kwargs) for km in self.kmeans])

    def transform(self, *args, **kwargs):
        return np.stack([km.transform(*args, **kwargs) for km in self.kmeans])
