#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement clustered self attention."""

import torch
import torch.autograd
from torch.nn import Module
from torch.nn.init import normal_

from fast_transformers.aggregate import clustered_aggregate, clustered_broadcast
from fast_transformers.clustering.hamming import cluster
from fast_transformers.hashing import compute_hashes
from fast_transformers.attention.clustered_attention import _GroupQueries


class SimulatedClusteredAttention(Module):
    """Use LSH and clustering in the resulting Hamming space to group queries
    that will have minimal L2 distance from each other.

    Given the queries, keys, and values as Q, K, and V respectively, we
    first cluster the queries in "C" groups and compute the "C" query centroids
    Q_c.

    We now use to the centroids Q_c to compute the attention using:

        V'_c = softmax(Q_c.mm(K.t()), dim=-1).mm(V).

    Now the computed values V'_c are "broadcasted" back to the query members
    of the corresponding cluster.

    Arguments
    ---------
        clusters: How many clusters to group the queries into
        iterations: The number of lloyd iterations to perform (default: 10)
        bits: How many bits to use for the hash (default: 32)
        hash_bias: If true, hamming distance proportional to L2 distance
                   If false, hamming distance proportional to cosine distance
                   (default: True)
    """
    def __init__(self, clusters, iterations=10, bits=32, hash_bias=True):
        super().__init__()
        self.clusters = clusters
        self.iterations = iterations
        self.bits = bits
        self.hash_bias = hash_bias

    def _create_query_groups(self, Q, lengths):
        N, H, L, E = Q.shape

        # Compute the hashes for all the queries
        planes = Q.new_empty((self.bits, E+1))
        normal_(planes)
        if not self.hash_bias:
            planes[:, -1] = 0
        hashes = compute_hashes(Q.view(N*H*L, E), planes).view(N, H, L)

        # Cluster the hashes and return the cluster index per query
        clusters, counts = cluster(
            hashes,
            lengths,
            clusters=self.clusters,
            iterations=self.iterations,
            bits=self.bits
        )
        sorted_clusters, sorted_indx = torch.sort(clusters, dim=-1)
        return (sorted_clusters, counts), sorted_indx

    def _group_queries(self, Q, groups, lengths):
        """Aggregate the Qs based on the index of cluster they belong to. Make
        sure to allow for gradient propagation backwards from the grouped
        queries to each query."""
        q_grouped = _GroupQueries.apply(Q, *groups, lengths)
        return q_grouped

    def forward(self, queries, keys, attn_mask, query_lengths):
        queries = queries.permute(0, 2, 1, 3).contiguous()
        keys = keys.permute(0, 2, 1, 3).contiguous()

        N, H, L, E = queries.shape
        _, _, S, D = keys.shape

        # Cluster the queries into groups
        groups, sorted_indx = self._create_query_groups(queries, query_lengths)

        # Re-organize queries so that first group belong to first cluster
        # next to second cluster and so on. This improves kernel implementations.
        # Note that this step is introduced after NeurIPS submission and
        # now the complexity is O(N log(N)).
        q_offset = torch.arange(N*H, device=queries.device).unsqueeze(-1) * L
        q_flat = (sorted_indx.view(N*H, -1) + q_offset).reshape(-1)
        s_queries = queries.reshape(-1, E).index_select(0, q_flat).view(N, H, L, E)

        # Aggregate the re-arranged queries.
        Q_grouped = self._group_queries(s_queries, groups, query_lengths)

        # Compute the attention
        QK = torch.einsum("nhle,nhse->nhls", Q_grouped, keys)

        # matches = groups.unsqueeze(-1)
        # matches = torch.ones(N, H, L, L).int()
        import ipdb; ipdb.set_trace()


if __name__ == '__main__':

    from utils import get_length_mask, subsequent_mask

    sim = SimulatedClusteredAttention(3)

    q = torch.randn(4, 2, 10, 8)
    k = torch.randn(4, 2, 10, 8)
    lengths = torch.tensor([6, 7, 8, 10]).int()
    mask = get_length_mask(lengths)
    sim(q, k, mask, lengths)
