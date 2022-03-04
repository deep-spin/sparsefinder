"""
Usage:
python3 convert_kmeans_to_torch_centroids.py --data $data -r 4 -s 4 -l 6 --share-projectors --cluster-rounds 1
python3 convert_kmeans_to_torch_centroids.py --data $data -r 4 -s 8 -l 6 --share-projectors --cluster-rounds 1
python3 convert_kmeans_to_torch_centroids.py --data $data -r 4 -s 12 -l 6 --share-projectors --cluster-rounds 1
python3 convert_kmeans_to_torch_centroids.py --data $data -r 4 -s 16 -l 6 --share-projectors --cluster-rounds 1
python3 convert_kmeans_to_torch_centroids.py --data $data -r 4 -s 20 -l 6 --share-projectors --cluster-rounds 1
"""

import argparse
import os
import pickle
import torch
import numpy as np


from group_projections import predict_clusters


def predict_clusters_torch(x_low, centroids, cdist=False):
    bs = x_low.shape[0]
    expanded_centroids = centroids.unsqueeze(0).expand(bs, -1, -1, -1, -1).double()
    # add `num_runs` dimension
    # (batch_size, num_heads, 1, q_seq_len, num_projections)
    expanded_x_low = x_low.unsqueeze(2).double()
    # q_dists.shape is (batch, num_heads, num_runs, q_seq_len, num_centroids)
    if cdist:
        x_dists = torch.cdist(expanded_x_low, expanded_centroids, p=2) ** 2
    else:
        # more stable
        x_dists = torch.sum((expanded_x_low.unsqueeze(-2) - expanded_centroids.unsqueeze(-3))**2, dim=-1)
    # q_clustered.shape is (batch, num_heads, num_runs, q_seq_len)
    x_clustered = torch.argmin(x_dists, dim=-1)
    # transpose to get `num_runs` as different hashing rounds
    # q_clustered.shape is (batch, num_heads, q_seq_len, num_runs)
    x_clustered = x_clustered.transpose(2, 3)
    return x_clustered.squeeze()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', dest='rounds', help='Hashing rounds', type=int, default=4)
    parser.add_argument('-s', dest='bucket_size', type=int, default=16, help='Bucket size')
    parser.add_argument('-l', '--num-layers', type=int)
    parser.add_argument('--data', type=str)
    parser.add_argument('--share-projectors', action='store_true', help='Whether to share projectors.')
    parser.add_argument('--cluster-rounds', type=int, default=1)
    parser.add_argument('--top_clusters', type=int, default=1, help='use "top_clusters" closest to each point.')
    parser.add_argument('--rescale-queries', action='store_true', help='Rescale Qs by sqrt(d) to fix fairseq data.')
    parser.add_argument('--add-causal-mask', action='store_true', help='Whether to use causal mask too.')
    parser.add_argument('--concat-q-and-k', action='store_true', help='Whether to concat the input - single proj.')
    parser.add_argument('--skip-proj', action='store_true', help='Whether to skip projection to low dim.')
    args = parser.parse_args()

    checks = 0
    centroids = []
    for layer in range(args.num_layers):
        # e.g. train20k-h8-new-attentions-40-1000000.pt_4r_4s_1n_0l_shared.pickle
        base_folder = 'kmeans/skip-proj' if args.skip_proj else 'kmeans'
        kmeans_path = "{}/{}_{}r_{}s_{}n_{}l_{}{}.pickle".format(
            base_folder,
            os.path.basename(args.data),
            args.rounds,  # projected vectors size
            args.bucket_size,  # how many clusters
            args.cluster_rounds,  # how many runs
            layer,
            'shared' if args.share_projectors else 'indep',
            '_concat' if args.concat_q_and_k else ''
        )
        print('Loading pretrained kmeans from: {}'.format(kmeans_path))
        with open(kmeans_path, 'rb') as handle:
            clusters_per_head = pickle.load(handle)
        centroids_l = np.stack([h.cluster_centers_ for h in clusters_per_head])
        # centroids_l.shape is (num_heads, num_runs, num_clusters, projection_size)
        centroids_torch = torch.from_numpy(centroids_l)

        # verify that torch and sklearn predictions match for dummy inputs
        num_heads = len(clusters_per_head)
        num_samples = 100
        bs = 8
        shape = (bs, num_heads, num_samples, args.rounds)
        dummy_q = 100 * torch.randn(*shape) + torch.randn(*shape) ** 2
        dummy_k = 100 * torch.randn(*shape) + torch.randn(*shape) ** 2
        clusters_q = predict_clusters(dummy_q, clusters_per_head).squeeze()
        clusters_k = predict_clusters(dummy_k, clusters_per_head).squeeze()
        clusters_q_torch = predict_clusters_torch(dummy_q, centroids_torch)
        clusters_k_torch = predict_clusters_torch(dummy_k, centroids_torch)
        check1 = torch.allclose(clusters_q.float(), clusters_q_torch.float())
        check2 = torch.allclose(clusters_k.float(), clusters_k_torch.float())
        checks += int(check1) + int(check2)
        print(check1, check2)
        centroids.append(centroids_torch)

    print('Checks: {} of {}'.format(checks, 2*args.num_layers))
    if checks < 2*args.num_layers:
        print('Warning! Sklearn and Torch predictions do not match!')
    else:
        print('Ok!')

    base_folder = 'centroids/skip-proj' if args.skip_proj else 'centroids'
    centroids_path = "{}/{}_{}r_{}s_{}n_{}{}.pickle".format(
        base_folder,
        os.path.basename(args.data),
        args.rounds,  # projected vectors size
        args.bucket_size,  # how many clusters
        args.cluster_rounds,  # how many runs
        'shared' if args.share_projectors else 'indep',
        '_concat' if args.concat_q_and_k else ''
    )
    if not os.path.exists('centroids'):
        os.mkdir('centroids')
    if args.skip_proj and not os.path.exists('centroids/skip-proj'):
        os.mkdir('centroids/skip-proj')
    print('Saving torch centroids to: {}'.format(centroids_path))
    torch.save(centroids, centroids_path)
