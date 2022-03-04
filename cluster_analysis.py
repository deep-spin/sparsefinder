import argparse
import os
import pickle
from collections import Counter;
from pprint import pprint
from extender_layers import HeadwiseLinearProjection, HeadwiseMLPProjection, HeadwiseSkipProjection

import torch
from entmax import entmax15
from scipy.stats import entropy

from group_projections import group_by_buckets
from qk_dataset import QKDataset
from utils import configure_seed, dot_product_and_mask, get_ground_truth, get_length_mask


def load_projectors(path):
    if not torch.cuda.is_available():
        map_location = torch.device('cpu')
    else:
        map_location = None
    return torch.load(path, map_location=map_location)


def get_layer_projectors(loaded_proj, layer):
    l_projs = loaded_proj[layer]
    proj_q_dict, proj_k_dict = l_projs['q'], l_projs['k']

    if len(proj_q_dict) == 4:
        n_heads, d, hid_dim = proj_q_dict['w1'].shape
        n_heads, hid_dim, rounds = proj_q_dict['w2'].shape
        proj_q = HeadwiseMLPProjection(n_heads, d, rounds, hid_dim)
        proj_k = HeadwiseMLPProjection(n_heads, d, rounds, hid_dim)
    elif len(proj_q_dict) == 2:
        n_heads, d, rounds = proj_q_dict['w'].shape
        proj_q = HeadwiseLinearProjection(n_heads, d, rounds)
        proj_k = HeadwiseLinearProjection(n_heads, d, rounds)
    else:
        raise TypeError("unknown kind of projector")

    proj_q.load_state_dict(proj_q_dict)
    proj_k.load_state_dict(proj_k_dict)

    if torch.cuda.is_available():
        proj_q.cuda()
        proj_k.cuda()
    return proj_q, proj_k


def load_kmeans(base_fname, layer):
    # train20k-h8-new-attentions-40-1000000.pt_4r_4s_1n_0l_shared.pickle
    fname = base_fname.format(layer)
    print('Loading pretrained kmeans from: ', fname)
    with open(fname, 'rb') as handle:
        clusters_per_head = pickle.load(handle)
    return clusters_per_head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', dest='rounds', help='Hashing rounds', type=int, default=4)
    parser.add_argument('-s', dest='bucket_size', type=int, default=16, help='Bucket size')
    parser.add_argument('--batch', help='Batch size', default=16, type=int)
    parser.add_argument('--data', help='Data produced by a real encoder (.pt file), as processed by split-attention.py')
    parser.add_argument('--load', default=None, help='Optional path to load projector, if provided evaluate only')
    parser.add_argument('--load-clusters', type=str, default=None, help='Path to pickled kmeans model')
    parser.add_argument('--seed', type=int, help='Random seed for everything', default=666)
    parser.add_argument('--share-projectors', action='store_true', help='Whether to share projectors.')
    parser.add_argument('--cluster-rounds', type=int, default=1)
    parser.add_argument('--top_clusters', type=int, default=1, help='use "top_clusters" closest to each point.')
    parser.add_argument('--rescale-queries', action='store_true', help='Rescale Qs by sqrt(d) to fix fairseq data.')
    parser.add_argument('--add-causal-mask', action='store_true', help='Whether to use causal mask too.')
    parser.add_argument('--concat-q-and-k', action='store_true', help='Whether to concat the input - single proj.')
    parser.add_argument('--skip-proj', action='store_true', help='Whether to skip projection to low dim.')

    args = parser.parse_args()
    pprint(vars(args))

    configure_seed(args.seed)
    dataset = QKDataset(args.data, args.batch, to_cuda=True, rescale_queries=args.rescale_queries)
    d = dataset.d
    num_heads = dataset.num_heads
    num_layers = dataset.num_layers
    bucket_size = args.bucket_size
    print('Num layers: {}'.format(num_layers))
    print('Num heads: {}'.format(num_heads))
    print('Head dim: {}'.format(d))

    if not args.skip_proj:
        loaded_proj = load_projectors(args.load)

    # e.g. train20k-h8-new-attentions-40-1000000.pt_4r_4s_1n_0l_shared.pickle
    base_folder = 'kmeans/skip-proj' if args.skip_proj else 'kmeans'
    cluster_load_str = "{}/{}_{}r_{}s_{}n_{{}}l_{}{}.pickle".format(
        base_folder,
        os.path.basename(args.data),
        args.rounds,  # projected vectors size
        args.bucket_size,  # how many clusters
        args.cluster_rounds,  # how many runs
        'shared' if args.share_projectors else 'indep',
        '_concat' if args.concat_q_and_k else ''
    )

    ent_fname = 'entropies/entropy{}.temp'.format(bucket_size)
    print('Saving entropies to: {}'.format(ent_fname))
    with open(ent_fname, 'w') as f:

        for layer in range(num_layers):
            print('Layer: %d' % layer)
            if not args.skip_proj:
                proj_q, proj_k = get_layer_projectors(loaded_proj, layer)
            else:
                input_size = d if not args.concat_q_and_k else 2 * d
                proj_q = proj_k = HeadwiseSkipProjection(num_heads, input_size, input_size)
                args.rounds = input_size

            clusters_per_head = load_kmeans(cluster_load_str, layer)

            # analyze here
            proj_q.eval()
            proj_k.eval()

            for q, k, lengths in dataset.get_eval_batch(layer):

                # batch_size = q.shape[0]
                # n = q.shape[2]
                # dots = dot_product_and_mask(q, k, lengths, causal=args.add_causal_mask)
                # att_dist = entmax15(dots, dim=-1)
                # inds, num_positive = get_ground_truth(dots, lengths, att_dist)
                # mask = get_length_mask(lengths, n)

                if args.concat_q_and_k:
                    q_low = k_low = proj_q(torch.cat([q, k], dim=-1))
                    print(q.shape)
                    print(q_low.shape)
                    # k_low = proj_k(torch.cat([q, k], dim=-1))
                else:
                    # (batch, heads, n, projections)
                    q_low = proj_q(q)
                    k_low = proj_k(k)
                    print(q.shape)
                    print(q_low.shape)

                batch_buckets_q, batch_buckets_k = group_by_buckets(
                    q_low,
                    k_low,
                    bucket_size,
                    lengths,
                    enforce_same_size=True,
                    temperature=1.0,
                    clusters_per_head=clusters_per_head,
                    top_clusters=args.top_clusters
                )

                # make sure entropy.temp file doesn't exist beforehand, since we are appending to it
                print(dataset.eval_q.shape)
                assert dataset.batch_size == dataset.eval_q.shape[0], "please set --batch to evalset-size: {}".format(dataset.eval_q.shape[0])
                eval_size = dataset.eval_q.shape[0]

                q_count = [str(sum([entropy(list(Counter(batch_buckets_q.squeeze(-1)[b, h, :].numpy()).values()), base=bucket_size) for b in range(eval_size)]) / eval_size) for h in range(num_heads)]
                k_count = [str(sum([entropy(list(Counter(batch_buckets_k.squeeze(-1)[b, h, :].numpy()).values()), base=bucket_size) for b in range(eval_size)]) / eval_size) for h in range(num_heads)]
                entr = '\t'.join([str(bucket_size), str(layer)] + q_count + k_count)
                print('Writing entropies...')
                f.write(entr + '\n')

            # free memory
            del proj_q
            del proj_k
            del clusters_per_head
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
