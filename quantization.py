import itertools
from pprint import pprint

import torch
from torch.optim import Adam
from entmax import entmax15
import numpy as np
from extender_layers import HeadwiseLinearProjection, HeadwiseMLPProjection, HeadwiseSkipProjection

import os
import pickle
# import pickle5 as pickle
import math
import argparse

from group_projections import group_by_distance, group_by_buckets, learn_clusters
from plot_utils import plot_train_and_eval_loss, plot_distance_and_buckets
from qk_dataset import QKDataset
from stats import compute_bucket_entropy, compute_bucket_recall, compute_bucket_sparsity, compute_bucket_counts, \
    compute_gold_sparsity, compute_bucket_exact_fraction
from utils import get_key_mask, configure_seed, get_ground_truth, dot_product_and_mask, \
    update_mean, append_tensor_cpu, update_graph_stats, get_window_positions, get_length_mask, subsequent_mask, \
    unsqueeze_as
from routing_transformer import routing_simulated_attention, learn_routing_centroids, save_kmeans_obj


def batch_compute_loss(
        q_low, k_low, sorted_k_inds, num_positive, lengths, margin=1.0,
        sample_from_same_bucket=False, dist='l2', concat_q_and_k=False,
        att_dist=None, add_causal_mask=False
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
        num_positive: Tensor (batch, n) with the number of actual positive
            keys for each query. Padding queries must have 0 positive keys.
        lengths: Tensor (batch,)
        margin: float. Default: 1.0
        sample_from_same_bucket: whether to sample negative keys only when they
            are clustered together with a given key. This may be necessary when
            buckets are not enforced to have the same number of keys and queries
        (?) same_number_negative: whether to include the same number of negative
            keys as the number of existing positive ones in the loss. This
            should be False if buckets have different sizes, True otherwise.
    """
    # fix case for concat_q_and_k:
    if concat_q_and_k:
        # if   qi -> kj
        # then qj -> ki
        L = att_dist.shape[2]
        # eps = 1e-12
        eps = att_dist[att_dist > 0].min().item() / 2
        att_dist_sym = att_dist.masked_fill((att_dist > 0).transpose(-1, -2), eps)

        # mask padding and causal
        mask = get_length_mask(lengths, L).unsqueeze(1).unsqueeze(2)
        if add_causal_mask:
            sub_mask = subsequent_mask(L).bool().to(mask.device)
            sub_mask = unsqueeze_as(sub_mask, mask, dim=0)  # add batch and head dims to sub_mask
            mask = mask & sub_mask
        att_dist_sym.masked_fill_(~mask, 0)

        # number of non-null keys for each query
        num_positive_new = (att_dist_sym > 0).sum(-1)
        sorted_k_inds_new = att_dist_sym.argsort(-1, descending=True)
        # zero positive counts past sentence end
        mask = get_length_mask(lengths, L)
        num_positive_new.masked_fill_(~mask.unsqueeze(1), 0)

        # overwrite
        sorted_k_inds = sorted_k_inds_new
        num_positive = num_positive_new

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

    batch_size, num_heads, n, num_proj = q_low.shape

    if sample_from_same_bucket:
        # on second thought, this strategy seems flawed bc the number of
        # available negative keys decreases as the accuracy increases, which
        # actually *increases* the loss since there will be less negative
        # differences to push the loss down
        raise NotImplementedError
        # # bucketized contains the contents of q_low split in buckets
        # # bucketized[i, j, k, l, m] = x[i, j, inds[i, j, k * num_buckets + l, m], m]
        # # bucket_inds[i, j, k, l] has the bucket assigned to projection l of the
        # # k-th token at head j at batch item i
        # # bucketized: (batch, heads, num_buckets, bucket_size, projections)
        # # inds: (batch, heads, num_buckets * bucket_size, projections)
        # bucketized_q, inds_q, bucket_inds_q = split_in_buckets(
        #     q_low, bucket_size)
        # bucketized_k, inds_k, bucket_inds_k = split_in_buckets(
        #     k_low, bucket_size)
        #
        # # for every query, get all the keys in the bucket it was assigned to
        # # k_per_q is (batch, heads, n, bucket_size, projections)
        # # index up to n to remove padding
        # inds = bucket_inds_q.unsqueeze(3).expand(
        #     -1, -1, -1, max_num_positive, -1)
        # k_per_q = torch.gather(bucketized_k, 2, inds)[:, :, :n]
        #
        # # now compute the diff between q and each k in the same bucket
        # diff = q_low.unsqueeze(3) - k_per_q
        #
        # # which are the keys in the buckets that the queries fell?
        # # inds_k_per_q is (batch, heads, n, bucket_size, projections)
        # inds = bucket_inds_q[:, :, :n].unsqueeze(3).expand(
        #     -1, -1, -1, max_num_positive, -1)
        # inds_k_per_q = torch.gather(inds_k, 2, inds)
        #
        # # which keys in the buckets are correct?
        # # we'll need to broadcast the ground truth keys and the indices to check
        # # every one
        # a = positive_inds.unsqueeze(-1).unsqueeze(3)
        # b = inds_k_per_q.unsqueeze(4)
        # matches = a == b
        #
        # # mask is (batch, heads, n, bucket_size, projections)
        # # 1 for positive keys, 0 otherwise
        # mask = matches.sum(3)
        #
        # # we want the diffs between q and negative k that fell in the same
        # # bucket so we zero out the diffs between q and positive ones
        # diff_neg = diff.masked_fill(mask, 0)
        #
        # # delete big tensors from memory
        # del mask, matches
        #
        # # infinity values appear in padding positions
        # diff_neg[torch.isinf(diff_neg)] = 0
    else:
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

        # (batch, heads, n * max_num_pos, num_proj)
        inds = k_low_inds.view(batch_size, num_heads, -1).unsqueeze(3).expand(-1, -1, -1, num_proj)

        k_neg = k_low.gather(2, inds).view(batch_size, num_heads, n, max_num_positive, num_proj)

        # (batch, num_heads, n, max_num_pos, num_proj)
        diff_neg = q_low.unsqueeze(3) - k_neg

    inds = positive_inds.reshape(batch_size, num_heads, -1).unsqueeze(3).expand(-1, -1, -1, num_proj)
    k_pos = k_low.gather(2, inds)
    k_pos = k_pos.view(batch_size, num_heads, n, max_num_positive, num_proj)
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
        epochs: int,
        max_train_steps: int,
        bucket_size: int,
        proj_q: HeadwiseLinearProjection,
        proj_k: HeadwiseLinearProjection,
        lr: float,
        l2: float,
        margin: float,
        layer: int,
        accumulation_steps: int = 1,
        train_dist='l2',
        add_causal_mask=False,
        add_cls_mask=False,
        add_last_mask=False,
        eval_dist: str = None,
        eval_dist_threshold: float = 0.5,
        eval_dist_p: float = 2.0,
        eval_enforce_same_size: bool = True,
        eval_temperature: float = 1.,
        eval_window: int = None,
        eval_steps: int = 10,
        eval_dataset: QKDataset = None,
        concat_q_and_k: bool = False
):
    """
    Train the model for a given number of steps

    Args:
        dataset: Dataset object providing data
        epochs: number of epochs
        max_train_steps: maximum number of training steps
        bucket_size: the number of tokens per bucket. If the bucket size is
            allowed to be variable, this is used to compute the total number of
            available buckets, supposing they had the same size.
        proj_q: module to project queries to a lower dimensionality
        proj_k: module to project keys to a lower dimensionality
        lr: learning rat
        l2: L2 penalty
        margin: margin used in the loss (scalar value)
        layer: which layer this is running on (maybe remove in the future)
        dist: name of the distance function, in case keys are chosen based on
            distance instead of quantization
        dist_threshold: distance threshold. Used only if `dist` is not None
        dist_p: p parameter of distance function
        enforce_same_size: whether to enforce all buckets to have the same
            number of tokens. In case a sentence needs padding, it goes to its
            end; padded buckets are in practice the only ones with a smaller
            size.
        temperature: coefficient to multiply values before applying a tanh, in
            case `enforce_same_size` is False
        window: window of tokens around each query that is automatically
            included in the set of its attended keys. Must be an even number,
            i.e., left half + query itself + right half.
            Keys falling inside this window are not computed in the loss (i.e.,
            the model is stimulated to look outside the window).
        accumulation_steps: for gradient accumulation
        eval_steps: how many steps to wait to perform validation
        eval_dataset:
    """
    proj_k.train()
    proj_q.train()
    parameters = list(proj_q.parameters()) + list(proj_k.parameters())
    adam = Adam(parameters, lr=lr, weight_decay=l2)
    adam.zero_grad()
    losses = []
    recalls = []
    sparsities = []
    curves_loss = []
    curves_recall = []
    curves_sparsity = []
    current_step = 0

    for epoch in range(epochs):
        if current_step > max_train_steps:
            break

        for i, (q, k, lengths) in enumerate(dataset.get_train_batch(layer)):
            # queries and keys are (batch, heads, n, dim)
            # lengths are (n,)
            if torch.cuda.is_available():
                q = q.cuda()
                k = k.cuda()
                lengths = lengths.cuda()

            n = q.shape[2]
            dots = dot_product_and_mask(q, k, lengths, causal=add_causal_mask)

            # (batch, heads, n, n)
            att_dist = entmax15(dots, dim=-1)

            # key_inds is (batch, heads, n, n)
            # num_positive is (batch, heads, n)
            key_inds, num_positive = get_ground_truth(dots, lengths, att_dist, causal=add_causal_mask)

            if concat_q_and_k:
                # choose proj_q to be the one that receives concat
                # (batch, heads, n, projections)
                q_low = k_low = proj_q(torch.cat([q, k], dim=-1))
                # k_low = proj_k(torch.cat([q, k], dim=-1))
            else:
                # (batch, heads, n, projections)
                q_low = proj_q(q)
                k_low = proj_k(k)

            loss = batch_compute_loss(
                q_low, k_low, key_inds, num_positive, lengths, margin,
                dist=train_dist, concat_q_and_k=concat_q_and_k, att_dist=att_dist, add_causal_mask=add_causal_mask
            )

            # need to normalize by accum_steps since we have a loss.mean() in batch_compute_loss
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                if current_step > max_train_steps:
                    break

                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                adam.step()
                adam.zero_grad()

                with torch.no_grad():
                    current_step += 1
                    losses.append(loss.item())
                    gold_sparsity = compute_gold_sparsity(att_dist, lengths,
                                                          causal=add_causal_mask)

                    if eval_window is not None:
                        window_inds = get_window_positions(n, eval_window, causal=add_causal_mask).to(q.device)
                    else:
                        window_inds = None

                    if eval_dist is not None:
                        sparsity, recall, exact, _ = group_by_distance(
                            q_low,
                            k_low,
                            lengths,
                            att_dist,
                            dist=eval_dist,
                            threshold=eval_dist_threshold,
                            p=eval_dist_p,
                            window_inds=window_inds,
                            num_positive=num_positive,
                            causal=add_causal_mask,
                            add_cls_mask=add_cls_mask
                        )
                        recalls.append(recall)
                        sparsities.append(sparsity)
                    else:
                        buckets_q, buckets_k = group_by_buckets(
                            q_low,
                            k_low,
                            bucket_size,
                            lengths,
                            enforce_same_size=eval_enforce_same_size,
                            temperature=eval_temperature
                        )
                        # entropy = compute_bucket_entropy(buckets_q, False)
                        recall = compute_bucket_recall(
                            buckets_q, buckets_k,
                            window_inds,
                            causal=add_causal_mask,
                            add_cls_mask=add_cls_mask,
                            add_last_mask=add_last_mask,
                            gold_p=att_dist,
                            lengths=lengths
                        )
                        batch_graph_stats = compute_bucket_sparsity(
                            buckets_q, buckets_k, lengths, window_inds,
                            causal=add_causal_mask,
                            add_cls_mask=add_cls_mask,
                            add_last_mask=add_last_mask,
                        )
                        recalls.append(recall.mean().item())
                        sparsities.append(batch_graph_stats['sparsity'].mean().item())

                    if eval_steps > 0 and (i + 1) % eval_steps == 0:
                        result, eval_loss = eval_model(
                            dataset=eval_dataset,
                            proj_q=proj_q,
                            proj_k=proj_k,
                            layer=layer,
                            bucket_size=bucket_size,
                            dist=eval_dist,
                            dist_threshold=eval_dist_threshold,
                            dist_p=eval_dist_p,
                            enforce_same_size=eval_enforce_same_size,
                            temperature=eval_temperature,
                            window=eval_window,
                            return_loss=True,
                            loss_margin=margin,
                            train_dist=train_dist,
                            add_causal_mask=add_causal_mask,
                            add_cls_mask=add_cls_mask,
                            add_last_mask=add_last_mask,
                            concat_q_and_k=concat_q_and_k,
                            verbose=False
                        )
                        avg_train_loss = np.mean(losses[-eval_steps:])
                        avg_train_recall = np.mean(recalls[-eval_steps:])
                        avg_train_sparsity = np.mean(sparsities[-eval_steps:])
                        curves_loss.append((avg_train_loss, eval_loss))
                        curves_recall.append((avg_train_recall, result['recall'].mean()))
                        curves_sparsity.append((avg_train_sparsity, result['graph_sparsity'].mean()))

                    print("%d/%d Loss: %.6f   Key recall: %.4f   Gold Sparsity: %.4f" %
                          (i + 1, dataset.num_train // dataset.batch_size + 1,
                           loss.item(), recall.mean().item(), gold_sparsity.mean().item()),
                          end='\r')

    if len(curves_loss) > 0:
        plot_train_and_eval_loss(layer, 'loss', curves_loss)
        plot_train_and_eval_loss(layer, 'recall', curves_recall)
        plot_train_and_eval_loss(layer, 'sparsity', curves_sparsity)

    return np.mean(losses)


def eval_model(
        dataset: QKDataset,
        proj_q: HeadwiseLinearProjection,
        proj_k: HeadwiseLinearProjection,
        layer: int, bucket_size: int,
        dist: str = None,
        dist_threshold: float = 0.5,
        dist_p: float = 2.0,
        enforce_same_size: bool = True,
        temperature: float = 1.,
        window: int = None,
        return_loss=False,
        loss_margin=1.0,
        train_dist='l2',
        plot_dist_x_buckets=False,
        plot_dist='euclidean',
        plot_bins=-1,
        clusters_per_head=None,
        top_clusters=1,
        add_causal_mask=False,
        add_cls_mask=False,
        add_last_mask=False,
        concat_q_and_k=False,
        verbose=True
):
    """
    Args:
        dataset: dataset providing data
        proj_q: projection module for queries
        proj_k: projection module for keys
        layer: number of the layer we get the data for (yeah, maybe that should
            already have been set in the dataset class)
        bucket_size: size of each bucket in number of queries or keys. In case
            of variable-size buckets, this value determines the number of total
            buckets, as `num_buckets = max_length / bucket_size`
        enforce_same_size: enforce that all buckets have the same size
        temperature: coefficient applied before tanh in case of variable-sized
            buckets
        window: window of tokens around each query that is automatically
            included in the set of its attended keys. Must be an even number,
            i.e., left half + query itself + right half.

    Return:
        Result tuple with:
        mean recall, mean graph (theoretical) sparsity, mean computation
        sparsity. The latter takes into account the amount of padding necessary
        to compute attentions within each batch item. In practice, the amount
        of necessary compute may be even larger when considering sequences of
        differente sizes in the same batch.
    """
    proj_q.eval()
    proj_k.eval()
    num_proj = proj_q.proj_dim
    num_heads = dataset.num_heads

    computed_samples = 0
    mean_gold_sparsity = None
    mean_recall = None
    mean_exact = None
    graph_sparsity = None
    compute_sparsity = None
    buckets_q = None
    buckets_k = None
    graph_stats = None
    mean_loss = None
    dist_x_buckets = None

    for q, k, lengths in dataset.get_eval_batch(layer):
        batch_size = q.shape[0]
        if torch.cuda.is_available():
            q = q.cuda()
            k = k.cuda()
            lengths = lengths.cuda()

        # compute the entmax distribution as a ground truth
        # dots is (batch, heads, n, n)
        dots = dot_product_and_mask(q, k, lengths, causal=add_causal_mask)
        att_dist = entmax15(dots, dim=-1)

        inds, num_positive = get_ground_truth(dots, lengths, att_dist, causal=add_causal_mask)

        # number of non-null keys for each query
        n = q.shape[2]
        num_heads = q.shape[1]
        if dist_x_buckets is None:
            dist_x_buckets = [[] for _ in range(num_heads)]

        # mask out counts at padding positions
        mask = get_length_mask(lengths, n)
        num_dense_operations = lengths ** 2
        batch_mean_gold_sparsity = compute_gold_sparsity(att_dist, lengths, causal=add_causal_mask)

        if concat_q_and_k:
            # choose proj_q to be the one that receives concat
            # (batch, heads, n, projections)
            q_low = k_low = proj_q(torch.cat([q, k], dim=-1))
            # k_low = proj_k(torch.cat([q, k], dim=-1))
        else:
            # (batch, heads, n, projections)
            q_low = proj_q(q)
            k_low = proj_k(k)

        if return_loss:
            batch_loss = batch_compute_loss(
                q_low, k_low, inds, num_positive, lengths, loss_margin,
                dist=train_dist, concat_q_and_k=concat_q_and_k, att_dist=att_dist, add_causal_mask=add_causal_mask
            )

        num_buckets = math.ceil(n / bucket_size)
        window_inds = None if window is None else get_window_positions(n, window).to(q.device)

        if dist is not None:
            batch_graph_sparsity, batch_mean_recall, batch_mean_exact, p_thresholded = \
                group_by_distance(
                    q_low,
                    k_low,
                    lengths,
                    att_dist,
                    dist=dist,
                    threshold=dist_threshold,
                    p=dist_p,
                    window_inds=window_inds,
                    num_positive=num_positive,
                    causal=add_causal_mask,
                    add_cls_mask=add_cls_mask,
                    add_last_mask=add_last_mask,
                    layer_id=layer,
                    clusters_per_head=clusters_per_head,
                    top_clusters=top_clusters
                )
            graph_stats = None
            operations = lengths ** 2
        else:
            batch_buckets_q, batch_buckets_k = group_by_buckets(
                q_low,
                k_low,
                bucket_size,
                lengths,
                enforce_same_size=enforce_same_size,
                temperature=temperature,
                clusters_per_head=clusters_per_head,
                top_clusters=top_clusters
            )

            if enforce_same_size:
                # the actual sparsity depends on sentence length; even if there are
                # longer ones in the batch, the shorter ones will be padded with inf
                # values that effectively reduce the number of used buckets to the
                # minimum necessary
                num_actual_buckets = torch.ceil(lengths.float() / bucket_size)
                ops_per_head = num_proj * num_actual_buckets * bucket_size ** 2
                operations = ops_per_head * num_heads
                counts_q = None
                counts_k = None
            else:
                counts_q, counts_k = compute_bucket_counts(batch_buckets_q, batch_buckets_k, num_buckets, mask)
                max_queries = counts_q.max(-1)[0]
                max_keys = counts_k.max(-1)[0]
                # todo: fix this
                operations_per_round = max_keys * max_queries * num_buckets
                operations = operations_per_round.sum(-1)

            if plot_dist_x_buckets:
                # bach_buckets_q.shape (batch, heads, n_q, num_proj)
                # common_buckets.shape (batch, heads, n_q, n_k, num_proj)
                common_buckets = batch_buckets_q.unsqueeze(-2) == batch_buckets_k.unsqueeze(-3)
                # (batch, n) -> (batch, 1, n_q, n_k)
                pairwise_mask = mask.unsqueeze(-1) & mask.unsqueeze(1)
                pairwise_mask = pairwise_mask.unsqueeze(1)
                # ignore masked elements
                common_buckets.masked_fill_(~pairwise_mask.unsqueeze(-1), False)
                # num_common_buckets.shape (batch, heads, n_q, n_k)
                num_common_buckets = common_buckets.int().sum(-1)
                if plot_dist == 'cos':
                    qproj_norm = q_low.norm(p=2, dim=-1).unsqueeze(-1)
                    kproj_norm = k_low.norm(p=2, dim=-1).unsqueeze(-1)
                    cos_sim = (q_low / qproj_norm) @ (k_low / kproj_norm).transpose(-1, -2)
                    pairwise_ds = 1 - cos_sim
                elif plot_dist == 'euclidean':
                    pairwise_ds = torch.cdist(q_low, k_low, p=2.0)
                else:
                    bs, nh, slen, _ = q_low.shape
                    pairwise_ds = torch.rand(bs, nh, slen, slen, device=q_low.device)
                for h in range(num_heads):
                    f_b = [num_common_buckets[i, h, :lengths[i], :lengths[i]].flatten().tolist() for i in range(batch_size)]
                    f_b = list(itertools.chain(*f_b))
                    f_p = [pairwise_ds[i, h, :lengths[i], :lengths[i]].flatten().tolist() for i in range(batch_size)]
                    f_p = list(itertools.chain(*f_p))
                    dist_x_buckets[h].extend(list(zip(f_p, f_b)))

            batch_mean_recall = compute_bucket_recall(
                batch_buckets_q, batch_buckets_k, window_inds,
                causal=add_causal_mask, add_cls_mask=add_cls_mask, add_last_mask=add_last_mask,
                gold_p=att_dist, lengths=lengths
            )
            batch_graph_stats = compute_bucket_sparsity(
                batch_buckets_q, batch_buckets_k, lengths, window_inds,
                causal=add_causal_mask, add_cls_mask=add_cls_mask, add_last_mask=add_last_mask
            )
            batch_mean_exact = compute_bucket_exact_fraction(
                batch_buckets_q, batch_buckets_k, window_inds,
                causal=add_causal_mask, add_cls_mask=add_cls_mask, add_last_mask=add_last_mask,
                gold_p=att_dist, lengths=lengths
            )
            # store outside graph_stats for compatibility with "if dist is not None:"
            batch_graph_sparsity = batch_graph_stats['sparsity']

            graph_stats = update_graph_stats(graph_stats, computed_samples, batch_graph_stats, batch_size)

            if dist is None and verbose:
                buckets_q = append_tensor_cpu(buckets_q, batch_buckets_q)
                buckets_k = append_tensor_cpu(buckets_k, batch_buckets_k)

        if window is not None:
            # it is added because it is computed independently; it's not viable
            # to check beforehand which windows are already covered and remove them
            # from the overall computation
            operations += n * window

        sparsities = 1 - (operations / num_dense_operations.float())
        batch_compute_sparsity = sparsities.mean()
        batch_compute_sparsity = batch_compute_sparsity.repeat(num_heads)
        mean_gold_sparsity = update_mean(mean_gold_sparsity, computed_samples, batch_mean_gold_sparsity, batch_size)
        mean_recall = update_mean(mean_recall, computed_samples, batch_mean_recall, batch_size)
        mean_exact = update_mean(mean_exact, computed_samples, batch_mean_exact, batch_size)
        graph_sparsity = update_mean(graph_sparsity, computed_samples, batch_graph_sparsity, batch_size)
        compute_sparsity = update_mean(compute_sparsity, computed_samples, batch_compute_sparsity, batch_size)
        if return_loss:
            mean_loss = update_mean(mean_loss, computed_samples, batch_loss, batch_size)
        computed_samples += batch_size

    if plot_dist_x_buckets:
        plot_type = 'scatter' if plot_bins == 0 else 'hist'
        plot_distance_and_buckets(
            layer,
            num_heads,
            dist_x_buckets,
            plot_dist=plot_dist,
            plot_type=plot_type,
            plot_bins=plot_bins
        )

    if verbose:
        print('Recall (keys found at least once): %.4f    '
              'Exact fraction (full entmax recover): %.4f    '
              'Gold Sparsity: %.2f%%    '
              'Computation sparsity: %.2f%%    '
              'Graph sparsity: %.2f%%' %
              (mean_recall.mean().item(),
               mean_exact.mean().item(),
               100 * mean_gold_sparsity.mean().item(),
               100 * compute_sparsity.mean().item(),
               100 * graph_sparsity.mean().item()))

    if dist is None:
        if verbose:
            entropy_q, dist_q = compute_bucket_entropy(buckets_q, True)
            entropy_k, dist_k = compute_bucket_entropy(buckets_k, True)
            print('Query bucket entropy: %.4f    Key bucket entropy: %.4f' % (entropy_q, entropy_k))

    result = {
        'recall': mean_recall,
        'exact': mean_exact,
        'graph_sparsity': graph_sparsity,
        'computation_sparsity': compute_sparsity,
        'gold_sparsity': mean_gold_sparsity,
        'stats': graph_stats,
    }

    if return_loss:
        return result, mean_loss
    return result


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


def main(args, dataset):
    configure_seed(args.seed)

    if args.temp > 1:
        print('Temperature set to %f; it will work but was intended to be lower than 1.' % args.temp)
    if args.window == 0:
        print("--window set to 0; setting instead to None, to avoid forcing each token to attend to itself")
        args.window = None
    if args.concat_q_and_k:
        # quick fix; this needs to be handled better in the future
        args.share_projectors = True

    d = dataset.d
    num_heads = dataset.num_heads
    num_layers = dataset.num_layers
    bucket_size = args.bucket_size
    train_steps = float('inf') if args.epochs is not None else args.steps
    if args.grouping != "dist" and args.dist is not None:
        print("Warning: --dist was provided, but --grouping != 'dist'. "
              "For backward compatibility, setting --grouping = dist")
        args.grouping = "dist"

    pprint(vars(args))
    print('Num layers: {}'.format(num_layers))
    print('Num heads: {}'.format(num_heads))
    print('Head dim: {}'.format(d))
    print('Train steps: {}'.format(train_steps))
    print('Num epochs: {}'.format(args.epochs))

    recalls = []
    exacts = []
    graph_sparsities = []
    compute_sparsities = []
    gold_sparsities = []
    projectors = []
    clusters_per_head_all = None

    if args.load is not None and not args.skip_proj:
        loaded_proj = load_projectors(args.load)

    for layer in range(num_layers):
        print('Layer: %d' % layer)

        if args.skip_proj:
            input_size = d if not args.concat_q_and_k else 2 * d
            proj_q = proj_k = HeadwiseSkipProjection(num_heads, input_size, input_size)
            args.rounds = input_size

        else:
            if args.load is None:
                input_size = d if not args.concat_q_and_k else 2 * d
                if args.hidden_size is not None:
                    proj_q = proj_k = HeadwiseMLPProjection(num_heads, input_size, args.rounds, args.hidden_size)
                    if not args.share_projectors:
                        proj_k = HeadwiseMLPProjection(num_heads, input_size, args.rounds, args.hidden_size)
                else:
                    proj_q = proj_k = HeadwiseLinearProjection(num_heads, input_size, args.rounds)
                    if not args.share_projectors:
                        proj_k = HeadwiseLinearProjection(num_heads, input_size, args.rounds)

                if torch.cuda.is_available():
                    proj_k.cuda()
                    proj_q.cuda()

                train_model(
                    dataset,
                    args.epochs,
                    args.steps,
                    bucket_size,
                    proj_q,
                    proj_k,
                    args.lr,
                    args.l2,
                    args.margin,
                    layer,
                    train_dist=args.train_dist,
                    accumulation_steps=args.accum,
                    add_causal_mask=args.add_causal_mask,
                    add_cls_mask=args.add_cls_mask,
                    add_last_mask=args.add_last_mask,
                    concat_q_and_k=args.concat_q_and_k,
                    eval_dist=args.dist,
                    eval_dist_threshold=args.dist_t,
                    eval_dist_p=args.dist_p,
                    eval_enforce_same_size=args.same_size,
                    eval_temperature=args.temp,
                    eval_window=args.window,
                    eval_steps=args.eval_steps,
                    eval_dataset=dataset,
                )
            else:
                proj_q, proj_k = get_layer_projectors(loaded_proj, layer)

        if args.grouping == 'kmeans':
            shared = 'shared' if args.share_projectors else 'indep'
            concat = '_concat' if args.concat_q_and_k else ''
            base_folder = 'kmeans/skip-proj' if args.skip_proj else 'kmeans'
            base_folder2 = 'centroids/skip-proj' if args.skip_proj else 'centroids'
            fname = '{}/{}_{}r_{}s_{}n_{}l_{}{}.pickle'.format(
                base_folder,
                os.path.basename(args.data),
                args.rounds,  # projected vectors size
                args.bucket_size,  # how many clusters
                args.cluster_rounds,  # how many runs
                layer,
                shared,
                concat
            )
            fname2 = '{}/{}_{}r_{}s_{}n_{}{}.pickle'.format(
                base_folder2,
                os.path.basename(args.data),
                args.rounds,  # projected vectors size
                args.bucket_size,  # how many clusters
                args.cluster_rounds,  # how many runs
                shared,
                concat
            )
            if not os.path.exists(fname):
                if not os.path.exists('kmeans'):
                    os.mkdir('kmeans')
                # todo bucket_size is instead n_buckets when applicable; improve name
                print('Training Kmeans...')
                clusters_per_head = learn_clusters(dataset, layer, proj_q, proj_k,
                                                   n_clusters=bucket_size, rounds=args.cluster_rounds,
                                                   concat_q_and_k=args.concat_q_and_k)
                print('Saving kmeans to: ', fname)
                with open(fname, 'wb') as handle:
                    pickle.dump(clusters_per_head, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                if not os.path.exists(fname2):
                    print('Loading pretrained kmeans from: ', fname)
                    with open(fname, 'rb') as handle:
                        clusters_per_head = pickle.load(handle)
                else:
                    print('Loading pretrained kmeans from: ', fname2)
                    if clusters_per_head_all is None:
                        clusters_per_head_all = torch.load(fname2, map_location=lambda storage, loc: storage)
                    clusters_per_head = clusters_per_head_all[layer]

        else:
            if args.dist == 'routing':
                # just use first init kmeans as centroids
                from routing_transformer import KmeansAttention
                km = KmeansAttention(args.bucket_size, None, dataset.num_heads, args.rounds)
                clusters_per_head = km.cuda()

            elif args.dist == 'routing_trained':
                # learn good centroids for routing transformer (always retrain because it is fast)
                print('Training routing kmeans for layer {}'.format(layer))
                clusters_per_head = learn_routing_centroids(
                    dataset,
                    proj_q,
                    proj_k,
                    layer,
                    num_clusters=bucket_size,
                    topk_window_size=None,  # balanced clusters with size of seq_len // num_clusters + 1
                    concat_q_and_k=args.concat_q_and_k,
                )
                if 'roberta' in args.data:
                    lp = 'entmax_roberta'
                else:
                    lp = 'en_fr' if 'en-fr' in args.data else 'en_de'
                # lp = os.path.basename(args.data)
                save_kmeans_obj(clusters_per_head, layer, lp=lp)

            elif args.dist == 'routing_extender':
                from routing_transformer import KmeansAttention
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
                print('Loading extender centroids from:', centroids_path)
                centroids = torch.load(centroids_path, map_location=lambda storage, loc: storage)
                centroids = centroids[layer][:, 0]  # get just the first run
                km = KmeansAttention(args.bucket_size, None, dataset.num_heads, args.rounds)
                km = km.cuda()
                km.kmeans.load_means(centroids.cuda())
                clusters_per_head = km

            else:
                clusters_per_head = None

        with torch.no_grad():
            torch.cuda.empty_cache()  # avoid keeping stuff from train_model
            result = eval_model(
                dataset,
                proj_q,
                proj_k,
                layer,
                bucket_size,
                dist=args.dist,
                dist_threshold=args.dist_t,
                dist_p=args.dist_p,
                enforce_same_size=args.same_size,
                temperature=args.temp,
                window=args.window,
                plot_dist_x_buckets=args.plot_dist_x_buckets,
                plot_dist=args.plot_dist,
                plot_bins=args.plot_bins,
                clusters_per_head=clusters_per_head,
                top_clusters=args.top_clusters,
                add_causal_mask=args.add_causal_mask,
                add_cls_mask=args.add_cls_mask,
                add_last_mask=args.add_last_mask,
                concat_q_and_k=args.concat_q_and_k
            )

            recalls.append(result['recall'].cpu().numpy())
            exacts.append(result['exact'].cpu().numpy())
            graph_sparsities.append(result['graph_sparsity'].cpu().numpy())
            compute_sparsities.append(result['computation_sparsity'].cpu().numpy())
            gold_sparsities.append(result['gold_sparsity'].cpu().numpy())

        if args.save is not None:
            projectors.append({'q': proj_q.state_dict(), 'k': proj_k.state_dict()})

        # free memory
        del proj_q
        del proj_k
        torch.cuda.empty_cache()

        # flush output
        print('')

    recalls = np.array(recalls)
    exacts = np.array(exacts)
    graph_sparsities = np.array(graph_sparsities)
    compute_sparsities = np.array(compute_sparsities)
    gold_sparsities = np.array(gold_sparsities)

    mean_recall = recalls.mean(0)
    mean_exact = exacts.mean(0)
    mean_gsparsity = graph_sparsities.mean(0)
    mean_csparsity = compute_sparsities.mean(0)

    format_heads = lambda v: ' '.join(['{:.4f}'.format(s) for s in v])
    print('Gold sparsities: {}'.format(format_heads(gold_sparsities.flatten())))
    print('Gold Sparsity -- mean {:.4f}'.format(gold_sparsities.mean()))
    print('Recall -- mean {}'.format(format_heads(mean_recall)))
    print('Exact -- mean {}'.format(format_heads(mean_exact)))
    print('Graph Sparsity -- mean {}'.format(format_heads(mean_gsparsity)))
    print('Compute sparsity -- mean {}'.format(format_heads(mean_csparsity)))

    # L0 H0 -> L0 H1 -> ... -> LN HN
    str_vals = ['{:.4f} {:.4f}'.format(x, y) for x, y in zip(graph_sparsities.flatten(), recalls.flatten())]

    # overall (average heads as well)
    str_vals.append('{:.4f} {:.4f}'.format(mean_gsparsity.mean(), mean_recall.mean()))

    print(' '.join(str_vals))

    print('---')
    str_vals = ['{:.4f}'.format(x) for x in exacts.flatten()]
    str_vals.append('{:.4f}'.format(exacts.mean()))
    print(' '.join(str_vals))

    if args.save and args.load is None:  # if loaded, we didn't train the model, no point in saving
        torch.save(projectors, args.save)


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', dest='rounds', help='Hashing rounds', type=int, default=4)
    parser.add_argument('-s', dest='bucket_size', type=int, default=16, help='Bucket size')
    parser.add_argument('--same-size', dest='same_size', action='store_true',
                        help='Enforce that buckets have the same number of queries and keys')
    parser.add_argument('--batch', help='Batch size', default=16, type=int)
    parser.add_argument('--data', help='Data produced by a real encoder (.pt file), as processed by split-attention.py')
    parser.add_argument('--lr', help='Learning rate', type=float, default=0.01)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--steps', default=float('inf'), type=float)
    parser.add_argument('--hidden-size', type=int)
    parser.add_argument('--l2', default=0., type=float)
    parser.add_argument('--margin', default=1.0, type=float)
    parser.add_argument('--grouping', type=str, choices=['quantize', 'dist', 'kmeans'], default='quantize')
    parser.add_argument('--cluster-rounds', type=int, default=1)
    parser.add_argument('--dist', type=str,
                        choices=['cos', 'euclidean', 'l2', 'direction', 'minkowski', 'random', 'bigbird', 'reformer',
                                 'reformer_rounds', 'longformer', 'openai_sparse', 'routing', 'dotproduct_entmax',
                                 'routing_extender', 'routing_trained', 'min', 'min_norm'])
    parser.add_argument('--dist-t', default=[0.5], type=float, nargs='+', help="distance thresholds for each head")
    parser.add_argument('--dist-p', default=2.0, type=float, help="distance p-norm")
    parser.add_argument('-v', action='store_true', help='Verbose', dest='verbose')
    parser.add_argument('--temp', type=float, default=1., help='Temperature coefficient before tanh')
    parser.add_argument('--window', type=int, help='Window around each token receiving attention')
    parser.add_argument('--accum', type=int, help='Steps of gradient accumulation', default=1)
    parser.add_argument('--save', help='Optional path to save projections')
    parser.add_argument('--load', default=None, help='Optional path to load projector, if provided evaluate only')
    parser.add_argument('--train-dist', type=str, choices=['l2', 'min'], default='l2')
    parser.add_argument('--eval-steps', type=int, help='Steps to wait to perform validation during training', default=0)
    parser.add_argument('--plot-dist', type=str, choices=['cos', 'euclidean', 'random'], default='euclidean')
    parser.add_argument('--plot-bins', type=int, help='Bins to plot dist x buckets histogram (-1 = auto)', default=-1)
    parser.add_argument('--plot-dist-x-buckets', action='store_true', help='Whether to plot dist x buckets.')
    parser.add_argument('--seed', type=int, help='Random seed for everything', default=666)
    parser.add_argument('--share-projectors', action='store_true', help='Whether to share projectors.')
    parser.add_argument('--top_clusters', type=int, default=1, help='use "top_clusters" closest to each point.')
    parser.add_argument('--rescale-queries', action='store_true', help='Rescale Qs by sqrt(d) to fix fairseq data.')
    parser.add_argument('--add-causal-mask', action='store_true', help='Whether to use causal mask too.')
    parser.add_argument('--add-cls-mask', action='store_true', help='Whether to use cls mask too.')
    parser.add_argument('--add-last-mask', action='store_true', help='Whether to use last token mask too.')
    parser.add_argument('--concat-q-and-k', action='store_true', help='Whether to concat the input - single proj.')
    parser.add_argument('--skip-proj', action='store_true', help='Whether to skip projection to low dim.')
    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    print('Loading data...')
    dataset = QKDataset(args.data, args.batch, rescale_queries=args.rescale_queries)
    main(args, dataset)
