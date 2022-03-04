import argparse
# import pickle5 as pickle
import math
from pprint import pprint

import numpy as np
import torch
from entmax import entmax15
from torch.optim import Adam

from clustering_blocks import KmeansManager, RoutingManager, RoutingExtendedManager
from extender_layers import HeadwiseLinearProjection, HeadwiseMLPProjection, HeadwiseSkipProjection, load_projectors, \
    get_layer_projectors
from group_projections_blocks import group_qk
from plot_utils import plot_train_and_eval_loss
from qk_dataset import QKDataset
from stats import compute_bucket_entropy, compute_gold_sparsity, compute_sparsity, compute_recall, \
    compute_exact_fraction
from utils import get_key_mask, configure_seed, get_ground_truth, dot_product_and_mask, \
    update_mean, append_tensor_cpu, get_length_mask, subsequent_mask, \
    get_ground_truth_for_tied_inputs, blockify


def batch_compute_loss(
    q_low,
    k_low,
    sorted_k_inds,
    num_positive,
    lengths,
    margin=1.0,
    dist='l2',
):
    """
    The loss is computed as

    loss = relu(margin + ||q - k_pos||^2 - ||q - k_neg||^2)

    k_pos are all positive keys for a given query
    k_neg are a number of negative keys sampled for each query

    Args:
        q_low: Tensor (batch, heads, n, projections)
        k_low: Tensor (batch, heads, n, projections)
        sorted_k_inds: Tensor (batch, heads, n, n) with the sorted indices of
            keys for each query (descending) according to dot products.
            sorted_k_inds[i, j, q, 0] has the highest scoring key for query q.
            The positive keys for each query are the first `num_positive`.
        num_positive: Tensor (batch, heads, n) with the number of actual positive
            keys for each query. Padding queries must have 0 positive keys.
        lengths: Tensor (batch,)
        margin: float. Default: 1.0
        dist: `l2` distance or `min` distance
    """
    max_length = lengths.max()
    if max_length < q_low.shape[2]:
        q_low = q_low[:, :, :max_length]
        k_low = k_low[:, :, :max_length]
        num_positive = num_positive[:, :, :max_length]
        sorted_k_inds = sorted_k_inds[:, :, :max_length, :max_length]

        # it may happen that many padding positions have an arbitrary order, and
        # sorted_k_inds points to something past the max_length. It makes no
        # difference in the loss computation but may cause problems in indexing.
        sorted_k_inds.clamp_(0, max_length - 1)

    # limit the number of positive keys per query to save compute
    max_num_positive = min(num_positive.max().item(), sorted_k_inds.shape[-1])
    positive_inds = sorted_k_inds[:, :, :, :max_num_positive]

    # recover info
    batch_size, num_heads, n, proj_size = q_low.shape

    # sample negative indices randomly.
    # This is maybe too complicated but warrant that we sample uniformly
    # from all negative keys for each query. We sample a random float for
    # each possible key, mask out the positive ones with inf (which are in
    # a different quantity for each query) as well as padding positions,
    # sort them and pick the corresponding indices.
    sample_score = torch.rand([batch_size, num_heads, n, n])
    sample_score = sample_score.to(num_positive.device)

    # first mask padding KEYS; we'll deal with padding queries later
    length_mask = get_length_mask(lengths, n).view(batch_size, 1, 1, n)
    sample_score.masked_fill_(~length_mask, np.inf)

    # mask the positions of positive keys
    key_mask = get_key_mask(num_positive, n)
    sample_score.masked_fill_(key_mask, np.inf)

    # we expect to find a number of negative keys at least equal to the
    # number of positive ones!

    # these are indices to k_neg
    inds = sample_score.argsort(-1)[:, :, :, :max_num_positive]
    k_low_inds = sorted_k_inds.gather(3, inds)

    # (batch, heads, n * max_num_pos, proj_size)
    inds = k_low_inds.view(batch_size, num_heads, -1).unsqueeze(3).expand(-1, -1, -1, proj_size)

    k_neg = k_low.gather(2, inds).view(batch_size, num_heads, n, max_num_positive, proj_size)

    # (batch, num_heads, n, max_num_pos, proj_size)
    diff_neg = q_low.unsqueeze(3) - k_neg

    inds = positive_inds.reshape(batch_size, num_heads, -1).unsqueeze(3).expand(-1, -1, -1, proj_size)
    k_pos = k_low.gather(2, inds)
    k_pos = k_pos.view(batch_size, num_heads, n, max_num_positive, proj_size)
    diff_pos = q_low.unsqueeze(3) - k_pos

    if dist == 'min':
        # min squared diff
        l2_sq_pos, _ = torch.min(diff_pos ** 2, dim=-1)
        l2_sq_neg, _ = torch.min(diff_neg ** 2, dim=-1)
    elif dist == 'l2':
        # L2 squared norm of the diffs: (batch, heads, n, bucket_size)
        l2_sq_pos = torch.sum(diff_pos ** 2, dim=-1)
        l2_sq_neg = torch.sum(diff_neg ** 2, dim=-1)
    else:
        raise NotImplementedError

    # zero out positions in in which there is no actual positive key
    # and the same number of negative keys
    key_mask = get_key_mask(num_positive, max_num_positive)
    l2_sq_pos.masked_fill_(~key_mask, 0)
    l2_sq_neg.masked_fill_(~key_mask, 0)

    # sum the l2 norms for all queries (still separated by head)
    # (batch, head, n)
    querywise_loss = torch.relu(margin + l2_sq_pos.sum(-1) - l2_sq_neg.sum(-1))

    # now mask padding positions - (batch, 1, n)
    length_mask = get_length_mask(lengths, n).unsqueeze(1)
    masked = querywise_loss.masked_fill(~length_mask, 0)
    headwise_loss = masked.sum(2) / lengths.unsqueeze(1)
    loss = headwise_loss.mean()

    return loss


def train_model(
    dataset: QKDataset,
    proj_q: HeadwiseLinearProjection,
    proj_k: HeadwiseLinearProjection,
    layer: int,
    args: argparse.Namespace,
):
    """
    Train the model for a given number of steps

    Args:
        dataset: Dataset object providing data
        proj_q: module to project queries to a lower dimensionality
        proj_k: module to project keys to a lower dimensionality
        layer: which layer this is running on (maybe remove in the future)
        args: hyperparameters used for training and validation
    """
    proj_k.train()
    proj_q.train()
    parameters = list(proj_q.parameters()) + list(proj_k.parameters())
    adam = Adam(parameters, lr=args.lr, weight_decay=args.l2)
    adam.zero_grad()
    losses = []
    curves_loss = []
    curves_recall = []
    curves_sparsity = []
    curves_gold_sparsity = []
    current_step = 0

    for epoch in range(args.epochs):
        for i, (q, k, lengths) in enumerate(dataset.get_train_batch(layer)):
            # queries and keys are (batch, heads, n, dim)
            # lengths are (n,)
            if torch.cuda.is_available():
                q = q.cuda()
                k = k.cuda()
                lengths = lengths.cuda()

            batch_size, num_heads, n, hidden_size = q.size()
            dots = dot_product_and_mask(q, k, lengths, causal=args.add_causal_mask)

            # (batch, heads, n, n)
            att_dist = entmax15(dots, dim=-1)

            if args.concat_q_and_k:
                # key_inds is (batch, heads, n, n)
                # num_positive is (batch, heads, n)
                key_inds, num_positive = get_ground_truth_for_tied_inputs(att_dist, lengths, args.add_causal_mask)

                # choose proj_q to be the one that receives concat
                # (batch, heads, n, projections)
                q_low = k_low = proj_q(torch.cat([q, k], dim=-1))
                # k_low = proj_k(torch.cat([q, k], dim=-1))
            else:
                # key_inds is (batch, heads, n, n)
                # num_positive is (batch, heads, n)
                key_inds, num_positive = get_ground_truth(att_dist, lengths)

                # (batch, heads, n, projections)
                q_low = proj_q(q)
                k_low = proj_k(k)

            if args.block_size > 1:
                # n -> n_blocks
                q_low, k_low, lengths, att_dist = blockify(q_low, k_low, lengths, att_dist, block_size=args.block_size)
                # key_inds: (batch, heads, n, n) -> (batch, heads, n_blocks, n_blocks)
                # num_positive: (batch, heads, n) -> (batch, heads, n_blocks)
                if args.concat_q_and_k:
                    key_inds, num_positive = get_ground_truth_for_tied_inputs(att_dist, lengths, args.add_causal_mask)
                else:
                    key_inds, num_positive = get_ground_truth(att_dist, lengths)

            # compute loss
            loss = batch_compute_loss(q_low, k_low, key_inds, num_positive, lengths,
                                      margin=args.margin, dist=args.train_dist)

            # need to normalize by accum_steps since we have a loss.mean() in batch_compute_loss
            loss = loss / args.accumulation_steps
            loss.backward()

            if (i + 1) % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                adam.step()
                adam.zero_grad()

                with torch.no_grad():
                    current_step += 1
                    losses.append(loss.item())

                    pairwise_conns, batch_buckets_q, batch_buckets_k = group_qk(
                        q_low,
                        k_low,
                        lengths,
                        strategy=args.grouping_strategy,
                        return_buckets=True,
                        clusters=None,
                        **vars(args)
                    )
                    # create pairwise_mask for padding
                    n_ = math.ceil(n / float(args.block_size)) if args.block_size > 1 else n
                    pad_mask = get_length_mask(lengths, n_)
                    causal_mask = subsequent_mask(n_).bool().to(pad_mask.device) if args.add_causal_mask else None

                    # compute metrics for train
                    train_loss = loss.item()
                    train_gold_sparsity = compute_sparsity((att_dist > 0).float(), pad_mask, causal_mask)
                    train_sparsity = compute_sparsity(pairwise_conns.float(), pad_mask, causal_mask)
                    train_recall = compute_recall(pairwise_conns.float(), att_dist.float(), pad_mask, causal_mask)

                    if args.eval_steps > 0 and (i + 1) % args.eval_steps == 0:

                        # compute metrics for val
                        eval_result = eval_model(dataset, proj_q, proj_k, layer, args, verbose=False)
                        eval_loss = eval_result['loss'].mean().item()
                        eval_gold_sparsity = eval_result['gold_sparsity'].mean().item()
                        eval_sparsity = eval_result['sparsity'].mean().item()
                        eval_recall = eval_result['recall'].mean().item()

                        # save tuple of train and val metrics to plot curves
                        curves_loss.append((train_loss, eval_loss))
                        curves_gold_sparsity.append((train_gold_sparsity.mean().item(), eval_gold_sparsity))
                        curves_sparsity.append((train_sparsity.mean().item(), eval_sparsity))
                        curves_recall.append((train_recall.mean().item(), eval_recall))

                    print('({}/{}) Loss: {:.6f}   Gold Sp: {:.4f}   Graph Sp: {:.4f}   Recall: {:.4f}'.format(
                        i + 1,
                        dataset.num_train // dataset.batch_size + 1,
                        train_loss,
                        train_gold_sparsity.mean().item(),
                        train_sparsity.mean().item(),
                        train_recall.mean().item(),
                    ), end='\r')

    if len(curves_loss) > 0:
        plot_train_and_eval_loss(layer, 'loss', curves_loss)
        plot_train_and_eval_loss(layer, 'recall', curves_recall)
        plot_train_and_eval_loss(layer, 'sparsity', curves_sparsity)
        plot_train_and_eval_loss(layer, 'gold_sparsity', curves_gold_sparsity)

    return np.mean(losses)


def eval_model(
    dataset: QKDataset,
    proj_q: HeadwiseLinearProjection,
    proj_k: HeadwiseLinearProjection,
    layer: int,
    args: argparse.Namespace,
    clusters=None,
    verbose=True
):
    proj_q.eval()
    proj_k.eval()

    computed_samples = 0
    mean_gold_sparsity = None
    mean_recall = None
    mean_exact = None
    mean_sparsity = None
    buckets_q = None
    buckets_k = None
    mean_loss = None

    for q, k, lengths in dataset.get_eval_batch(layer):
        if torch.cuda.is_available():
            q = q.cuda()
            k = k.cuda()
            lengths = lengths.cuda()

        batch_size, num_heads, n, hidden_size = q.size()

        # compute the entmax distribution as a ground truth
        # dots is (batch, heads, n, n)
        dots = dot_product_and_mask(q, k, lengths, causal=args.add_causal_mask)
        att_dist = entmax15(dots, dim=-1)

        if args.concat_q_and_k:
            # key_inds is (batch, heads, n, n)
            # num_positive is (batch, heads, n)
            key_inds, num_positive = get_ground_truth_for_tied_inputs(att_dist, lengths, args.add_causal_mask)

            # choose proj_q to be the one that receives concat
            # (batch, heads, n, projections)
            q_low = k_low = proj_q(torch.cat([q, k], dim=-1))
            # k_low = proj_k(torch.cat([q, k], dim=-1))
        else:
            # key_inds is (batch, heads, n, n)
            # num_positive is (batch, heads, n)
            key_inds, num_positive = get_ground_truth(att_dist, lengths)

            # (batch, heads, n, projections)
            q_low = proj_q(q)
            k_low = proj_k(k)

        if args.block_size > 1:
            # n -> n_blocks
            q_low, k_low, lengths, att_dist = blockify(q_low, k_low, lengths, att_dist, block_size=args.block_size)
            # key_inds: (batch, heads, n, n) -> (batch, heads, n_blocks, n_blocks)
            # num_positive: (batch, heads, n) -> (batch, heads, n_blocks)
            if args.concat_q_and_k:
                key_inds, num_positive = get_ground_truth_for_tied_inputs(att_dist, lengths, args.add_causal_mask)
            else:
                key_inds, num_positive = get_ground_truth(att_dist, lengths)

        # todo: check if blocks are ok
        # import ipdb; ipdb.set_trace()

        pairwise_conns, batch_buckets_q, batch_buckets_k = group_qk(
            q_low,
            k_low,
            lengths,
            strategy=args.grouping_strategy,
            return_buckets=True,
            clusters=clusters,
            **vars(args)
        )

        # create pairwise_mask for padding
        n_ = math.ceil(n / float(args.block_size)) if args.block_size > 1 else n
        pad_mask = get_length_mask(lengths, n_)
        causal_mask = subsequent_mask(n_).bool().to(pad_mask.device) if args.add_causal_mask else None

        # compute metrics
        batch_mean_gold_sparsity = compute_sparsity((att_dist > 0).float(), pad_mask, causal_mask)
        batch_mean_sparsity = compute_sparsity(pairwise_conns.float(), pad_mask, causal_mask)
        batch_mean_recall = compute_recall(pairwise_conns.float(), att_dist.float(), pad_mask, causal_mask)
        batch_mean_exact = compute_exact_fraction(pairwise_conns.float(), att_dist.float(), pad_mask, causal_mask)

        mean_gold_sparsity = update_mean(mean_gold_sparsity, computed_samples, batch_mean_gold_sparsity, batch_size)
        mean_sparsity = update_mean(mean_sparsity, computed_samples, batch_mean_sparsity, batch_size)
        mean_recall = update_mean(mean_recall, computed_samples, batch_mean_recall, batch_size)
        mean_exact = update_mean(mean_exact, computed_samples, batch_mean_exact, batch_size)

        # for printing bucket entropy
        if batch_buckets_q is not None and verbose:
            buckets_q = append_tensor_cpu(buckets_q, batch_buckets_q)
            buckets_k = append_tensor_cpu(buckets_k, batch_buckets_k)

        # also get the loss value
        batch_loss = batch_compute_loss(q_low, k_low, key_inds, num_positive, lengths, args.margin, args.train_dist)
        mean_loss = update_mean(mean_loss, computed_samples, batch_loss, batch_size)
        computed_samples += batch_size

    if verbose:
        print('')
        print(
            'Recall (keys found at least once): {:.4f}   '
            'Exact fraction (full entmax recover): {:.4f}   '
            'Gold Sparsity: {:.4f}   '
            'Graph sparsity: {:.4f}'.format(
                mean_recall.mean().item(),
                mean_exact.mean().item(),
                mean_gold_sparsity.mean().item(),
                mean_sparsity.mean().item())
              )

        if buckets_q is not None:
            entropy_q = compute_bucket_entropy(buckets_q, return_distribution=False)
            entropy_k = compute_bucket_entropy(buckets_k, return_distribution=False)
            print('Query bucket entropy: {:.4f}    Key bucket entropy: {:.4f}'.format(entropy_q, entropy_k))

    result = {
        'recall': mean_recall,
        'exact': mean_exact,
        'sparsity': mean_sparsity,
        'gold_sparsity': mean_gold_sparsity,
        'loss': mean_loss
    }
    return result


def main(args, dataset):
    configure_seed(args.seed)

    if args.temperature > 1:
        print('Temperature set to {}; it will work but was intended to be lower than 1.'.format(args.temp))

    if args.concat_q_and_k:
        # quick fix; this needs to be handled better in the future
        print('Tying projectors since concat(q, k) == concat(k, q)')
        args.share_projectors = True

    args.hidden_size = dataset.d
    args.num_heads = dataset.num_heads
    args.num_layers = dataset.num_layers
    recalls = []
    exacts = []
    graph_sparsities = []
    gold_sparsities = []
    projectors = []

    pprint(vars(args))
    print('Num layers: {}'.format(args.num_layers))
    print('Num heads: {}'.format(args.num_heads))
    print('Head dim: {}'.format(args.hidden_size))
    print('Num epochs: {}'.format(args.epochs))

    # load projections
    loaded_projs = None
    if args.load is not None and not args.skip_projectors:
        loaded_projs = load_projectors(args.load)

    # load clusters
    loaded_clusters = None
    if args.grouping_strategy == 'clustering_kmeans':
        loaded_clusters = KmeansManager(args)
        loaded_clusters.load_if_exists()
    elif args.grouping_strategy == 'simulated_routing':
        loaded_clusters = RoutingManager(args)
        loaded_clusters.load_if_exists()
    elif args.grouping_strategy == 'simulated_routing_extended':
        loaded_clusters = RoutingExtendedManager(args)
        loaded_clusters.load_if_exists()

    for layer in range(args.num_layers):
        print('Layer: {}'.format(layer))

        if args.skip_projectors:
            input_size = args.hidden_size if not args.concat_q_and_k else 2 * args.hidden_size
            args.rounds = input_size
            proj_q = proj_k = HeadwiseSkipProjection(args.num_heads, input_size, input_size)
            proj_q.cuda()
            proj_k.cuda()
        else:
            if loaded_projs is not None:
                proj_q, proj_k = get_layer_projectors(loaded_projs, layer)
                proj_q.cuda()
                proj_k.cuda()
            else:
                input_size = args.hidden_size if not args.concat_q_and_k else 2 * args.hidden_size
                if args.hidden_size is not None:
                    proj_q = proj_k = HeadwiseMLPProjection(args.num_heads, input_size, args.rounds, args.hidden_size)
                    if not args.share_projectors:
                        proj_k = HeadwiseMLPProjection(args.num_heads, input_size, args.rounds, args.hidden_size)
                else:
                    proj_q = proj_k = HeadwiseLinearProjection(args.num_heads, input_size, args.rounds)
                    if not args.share_projectors:
                        proj_k = HeadwiseLinearProjection(args.num_heads, input_size, args.rounds)
                proj_q.cuda()
                proj_k.cuda()
                train_model(dataset, proj_q, proj_k, layer, args)

        # if not loaded, learn clusters (for sparsefinder kmeans and routing transformer)
        if loaded_clusters is not None:
            loaded_clusters.current_layer = layer
            if not loaded_clusters.loaded:
                loaded_clusters.learn(dataset, layer, proj_q, proj_k)

        # validation
        with torch.no_grad():
            torch.cuda.empty_cache()  # avoid keeping stuff from train_model
            result = eval_model(dataset, proj_q, proj_k, layer, args, clusters=loaded_clusters, verbose=True)
            recalls.append(result['recall'].cpu().numpy())
            exacts.append(result['exact'].cpu().numpy())
            graph_sparsities.append(result['sparsity'].cpu().numpy())
            gold_sparsities.append(result['gold_sparsity'].cpu().numpy())

        # to save projectors
        if args.save is not None:
            projectors.append({'q': proj_q.state_dict(), 'k': proj_k.state_dict()})

        # free memory
        del proj_q
        del proj_k
        torch.cuda.empty_cache()

        # flush output
        print('')

    # print stats to screen
    recalls = np.array(recalls)
    exacts = np.array(exacts)
    graph_sparsities = np.array(graph_sparsities)
    gold_sparsities = np.array(gold_sparsities)

    mean_recall = recalls.mean(0)
    mean_exact = exacts.mean(0)
    mean_sparsity = graph_sparsities.mean(0)

    format_heads = lambda v: ' '.join(['{:.4f}'.format(s) for s in v])
    print('Gold Sparsity -- mean {:.4f}'.format(gold_sparsities.mean()))
    print('Graph Sparsity -- mean {}'.format(format_heads(mean_sparsity)))
    print('Recall -- mean {}'.format(format_heads(mean_recall)))
    print('Exact -- mean {}'.format(format_heads(mean_exact)))
    print('Gold sparsities: {}'.format(format_heads(gold_sparsities.flatten())))

    # print to gdocs
    # L0 H0 -> L0 H1 -> ... -> LN HN
    str_vals = ['{:.4f} {:.4f}'.format(x, y) for x, y in zip(graph_sparsities.flatten(), recalls.flatten())]
    # overall (average heads as well)
    str_vals.append('{:.4f} {:.4f}'.format(mean_sparsity.mean(), mean_recall.mean()))
    print(' '.join(str_vals))
    print('---')
    str_vals = ['{:.4f}'.format(x) for x in exacts.flatten()]
    str_vals.append('{:.4f}'.format(exacts.mean()))
    print(' '.join(str_vals))

    # save clusters
    if loaded_clusters is not None and not loaded_clusters.loaded:
        print('Saving {} centroids'.format(len(loaded_clusters.centroids)))
        loaded_clusters.save()

    # save projections
    if args.save and args.load is None:  # if loaded, we didn't train the model, no point in saving
        print('Saving projectors to {}'.format(args.save))
        torch.save(projectors, args.save)


def get_arg_parser():
    parser = argparse.ArgumentParser()

    # projections
    parser.add_argument('-r', '--rounds', type=int, default=4, help='Hashing rounds')
    parser.add_argument('--hidden-size', type=int, default=None, help='If not None, projectors are MLPs')
    parser.add_argument('--share-projectors', action='store_true', help='Whether to share projectors.')
    parser.add_argument('--skip-projectors', action='store_true', help='Whether to skip projection to low dim.')
    parser.add_argument('--rescale-queries', action='store_true', help='Rescale Qs by sqrt(d) to fix fairseq data')
    parser.add_argument('--concat-q-and-k', action='store_true', help='Whether to concat the q and k (single proj)')

    # grouping
    parser.add_argument('--block-size', type=int, default=1, help='size of each chunk')
    parser.add_argument('--grouping-strategy', type=str, default='clustering_kmeans')
    parser.add_argument('--num-clusters', type=int, default=4, help='number of clusters')
    parser.add_argument('--top_clusters', type=int, default=1, help='use "top_clusters" closest to each point')
    parser.add_argument('--cluster-rounds', type=int, default=1, help='number of independent k-means runs')
    parser.add_argument('--threshold', default=0.5, type=float, help="distance threshold")
    parser.add_argument('--temperature', type=float, default=1., help='Temperature coefficient before tanh')
    parser.add_argument('--window-size', type=int, default=0, help='Window around each token receiving attention')
    parser.add_argument('--bucket-size', type=int, default=16, help='Bucket size')
    parser.add_argument('--same-size', action='store_true', help='Enforce buckets to have the same num of qs and ks')

    # constraints
    parser.add_argument('--add-cls-mask', action='store_true', help='Whether to use cls mask')
    parser.add_argument('--add-last-mask', action='store_true', help='Whether to use last token mask')
    parser.add_argument('--add-causal-mask', action='store_true', help='Whether to use causal mask')

    # overall
    parser.add_argument('--data', help='Data produced by a real encoder (.pt file), as processed by split-attention.py')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save', help='Path to save projections')
    parser.add_argument('--load', default=None, help='Path to load projector. If provided, only evaluation is run')

    # train
    parser.add_argument('--lr', help='Learning rate', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--l2', type=float, default=0.)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--train-dist', type=str, choices=['l2', 'min'], default='l2')
    parser.add_argument('--accumulation-steps', type=int, default=1, help='Steps of gradient accumulation')
    parser.add_argument('--eval-steps', type=int, default=0, help='Steps to wait to perform validation')

    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    print('Loading data... (it might take a few minutes)')
    dataset = QKDataset(args.data, args.batch_size, rescale_queries=args.rescale_queries)
    print('Done!')

    main(args, dataset)
