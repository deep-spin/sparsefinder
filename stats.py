import math

import torch
from torch import Tensor

from utils import get_length_mask, subsequent_mask, unsqueeze_as, neighbours_mask


def compute_bucket_entropy(
    buckets: torch.Tensor,
    return_distribution: bool = True
):
    """
    Computes the entropy for a specific bucketed tensor.

    Args:
        buckets: Tensor (batch, heads, n, projections)
        return_distribution (bool): whether to return the distribution of the
            number of elements per bucket
    """
    # TODO: return entropy per head
    values, counts = torch.unique(buckets, return_counts=True)
    p = counts.float() / counts.sum()
    log_p = torch.log(p)
    log_p[torch.isinf(log_p)] = 0
    entropy = -torch.sum(p * log_p).item()
    if return_distribution:
        return entropy, p
    return entropy


def compute_bucket_recall(
    buckets_q,
    buckets_k,
    window_inds=None,
    causal=False,
    add_cls_mask=False,
    add_last_mask=False,
    gold_p=None,
    lengths=None
):
    """
    Computes accuracy as the ratio of "gold" keys attended by queries
    considering all buckets.

    Args:
        buckets_q: Tensor (batch, heads, n, projections)
        buckets_k: Tensor (batch, heads, n, projections)
        positive_inds: Tensor (batch, heads, n, k) indices of the keys sorted
            according to relevance to each query; may contain padding
        num_positive: Tensor (batch, heads, n) with the number of actual
            positive key per query
        lengths: Tensor (batch,) with the length of queries
        window_inds: Tensor (n, window_size) with the positions of keys attended
            by each query in all batch items and heads. The keys found here are
            added to the set found via bucketing.
    """
    batch_size = buckets_q.shape[0]
    n = buckets_q.shape[2]

    # cross all Q and K to find out when they match
    # shape is (batch, heads, query, key, projection)
    qk = buckets_q.unsqueeze(3) == buckets_k.unsqueeze(2)

    # consider matches across any projection; (batch, heads, query, key)
    qk = qk.sum(4) > 0
    # qk = qk.any(-1)

    if window_inds is not None:
        window_size = window_inds.shape[-1]
        win_mask = neighbours_mask(n, window_size).unsqueeze(0).unsqueeze(1).to(qk.device).bool()
        qk |= win_mask
        # r = torch.arange(n).view(-1, 1).to(qk.device)
        # qk[:, :, r, window_inds] = True

    if add_cls_mask:
        qk[:, :, 0, :] = True
        qk[:, :, :, 0] = True

    if add_last_mask is True:
        qk[:, :, -1, :] = True
        qk[:, :, :, -1] = True

    pad_mask = get_length_mask(lengths, n)
    causal_mask = None
    if causal:
        causal_mask = subsequent_mask(n).bool().to(pad_mask.device)

    recall_per_head = compute_recall(qk.float(), gold_p.float(), pad_mask, causal_mask=causal_mask)

    return recall_per_head


def compute_bucket_sparsity(
    buckets_q: Tensor,
    buckets_k: Tensor,
    lengths: Tensor,
    window_inds: Tensor = None,
    causal=False,
    add_cls_mask=False,
    add_last_mask=False,
):
    """
    Compute the graph sparsity across all projection rounds; i.e., as if queries
    could look at any key it shared at least one bucket with.

    Args:
        buckets_q: Tensor (batch, heads, n, projections)
        buckets_k: Tensor (batch, heads, n, projections)
        lengths: Tensor (batch, )
        window_inds: Tensor (n, window_size) with the positions attended
           inside a window around each query
        counts_q: Tensor (heads, num_buckets) counting the number of
           queries in each bucket
        counts_k: Tensor (heads, num_buckets) counting the number of
            keys in each bucket
    """
    batch_size = buckets_q.shape[0]
    n = buckets_q.shape[2]

    # cross all Q and K to find out when they match
    # shape is (batch, heads, query, key, projection)
    qk = buckets_q.unsqueeze(3) == buckets_k.unsqueeze(2)

    # consider matches across any projection; (batch, heads, query, key)
    qk = qk.sum(4) > 0

    if window_inds is not None:
        window_size = window_inds.shape[-1]
        win_mask = neighbours_mask(n, window_size).unsqueeze(0).unsqueeze(1).to(qk.device).bool()
        qk |= win_mask
        # r = torch.arange(n).view(-1, 1).to(qk.device)
        # qk[:, :, r, window_inds] = True

    if add_cls_mask:
        qk[:, :, 0, :] = True
        qk[:, :, :, 0] = True

    if add_last_mask is True:
        qk[:, :, -1, :] = True
        qk[:, :, :, -1] = True

    pad_mask = get_length_mask(lengths, n)
    if causal:
        causal_mask = subsequent_mask(n).bool().to(pad_mask.device)
        joint_mask = pad_mask.unsqueeze(-1) & pad_mask.unsqueeze(1)
        joint_mask = joint_mask & causal_mask.unsqueeze(0)
    else:
        causal_mask = None
        joint_mask = pad_mask.unsqueeze(-1) & pad_mask.unsqueeze(1)

    nz = qk.masked_fill(~joint_mask.unsqueeze(1), False).nonzero(as_tuple=False)[:, -2:]
    diff = nz[:, 1] - nz[:, 0]
    sparsity_per_head = compute_sparsity(qk.float(), pad_mask, causal_mask=causal_mask)

    stats = {
        'sparsity': sparsity_per_head,
        'edge_distances': diff,
    }
    return stats


def compute_bucket_exact_fraction(
        buckets_q,
        buckets_k,
        window_inds=None,
        causal=False,
        add_cls_mask=False,
        add_last_mask=False,
        gold_p=None,
        lengths=None
):
    """
    Computes fraction of queries for which recall is 100%, which recovers the exact entmax graph

    Args:
        buckets_q: Tensor (batch, heads, n, projections)
        buckets_k: Tensor (batch, heads, n, projections)
        positive_inds: Tensor (batch, heads, n, k) indices of the keys sorted
            according to relevance to each query; may contain padding
        num_positive: Tensor (batch, heads, n) with the number of actual
            positive key per query
        lengths: Tensor (batch,) with the length of queries
        window_inds: Tensor (n, window_size) with the positions of keys attended
            by each query in all batch items and heads. The keys found here are
            added to the set found via bucketing.
    """
    batch_size = buckets_q.shape[0]
    n = buckets_q.shape[2]

    # cross all Q and K to find out when they match
    # shape is (batch, heads, query, key, projection)
    qk = buckets_q.unsqueeze(3) == buckets_k.unsqueeze(2)

    # consider matches across any projection; (batch, heads, query, key)
    qk = qk.sum(4) > 0
    # qk = qk.any(-1)

    if window_inds is not None:
        window_size = window_inds.shape[-1]
        win_mask = neighbours_mask(n, window_size).unsqueeze(0).unsqueeze(1).to(qk.device).bool()
        qk |= win_mask
        # r = torch.arange(n).view(-1, 1).to(qk.device)
        # qk[:, :, r, window_inds] = True

    if add_cls_mask:
        qk[:, :, 0, :] = True
        qk[:, :, :, 0] = True

    if add_last_mask is True:
        qk[:, :, -1, :] = True
        qk[:, :, :, -1] = True

    pad_mask = get_length_mask(lengths, n)
    causal_mask = None
    if causal:
        causal_mask = subsequent_mask(n).bool().to(pad_mask.device)

    exact_per_head = compute_exact_fraction(qk.float(), gold_p.float(), pad_mask, causal_mask=causal_mask)

    return exact_per_head


def compute_gold_sparsity(p, lengths, causal=False):
    """
    Compute the gold sparsity of the distribution `p`

    Args:
        p: tensor (batch, heads, n, n)
        lengths: tensor (batch,)
        causal: bool
    """
    max_length = p.shape[2]

    # number of non-null keys for each query
    positive_p = p > 0
    pad_mask = get_length_mask(lengths, max_length)
    causal_mask = None
    if causal:
        causal_mask = subsequent_mask(max_length).bool().to(pad_mask.device)
    sparsity_per_head = compute_sparsity(positive_p.float(), pad_mask, causal_mask=causal_mask)
    return sparsity_per_head


def compute_bucket_counts(buckets_q, buckets_k, num_buckets, mask):
    buckets = torch.arange(num_buckets).view(1, 1, 1, 1, num_buckets)
    buckets = buckets.to(buckets_k.device)
    presences_k = buckets_k.unsqueeze(-1) == buckets
    presences_q = buckets_q.unsqueeze(-1) == buckets
    mask = mask.view(mask.shape[0], 1, -1, 1, 1)
    presences_k.masked_fill_(~mask, False)
    presences_q.masked_fill_(~mask, False)
    counts_k = presences_k.sum(2)
    counts_q = presences_q.sum(2)
    counts_q_per_head = counts_q.sum(-2).sum(0)
    counts_k_per_head = counts_k.sum(-2).sum(0)
    return counts_q_per_head, counts_k_per_head


def compute_recall(pred_p, gold_p, pad_mask, causal_mask=None):
    """
    Compute the recall between pred_p and gold_p selections

    Args:
        pred_p: float tensor (batch, heads, n, n)
        gold_p: float tensor (batch, heads, n, n)
        pad_mask: bool tensor (batch, n)
        causal_mask: bool tensor  (n, n)
    """
    pairwise_mask = pad_mask.unsqueeze(-1) & pad_mask.unsqueeze(1)
    pairwise_mask = pairwise_mask.unsqueeze(1)
    if causal_mask is not None:
        pairwise_mask = pairwise_mask & causal_mask.unsqueeze(0).unsqueeze(1)

    # compute (micro-average) recall for each batch item then average
    gold_p = (gold_p > 0).masked_fill(~pairwise_mask, False)
    pred_p = (pred_p > 0).masked_fill(~pairwise_mask, False)
    matches = pred_p & gold_p
    matches_per_head = matches.sum(-1).sum(-1).float()
    total_per_head = gold_p.sum(-1).sum(-1).float()
    recall_per_head = matches_per_head / total_per_head
    recall = recall_per_head.mean(0)
    return recall


def compute_sparsity(p, pad_mask, causal_mask=None):
    """
    Compute the sparsity of the distribution `p`

    Args:
        p: float tensor (batch, heads, n, n)
        pad_mask: bool tensor (batch, n)
        causal_mask: bool tensor  (n, n)
    """
    pairwise_mask = pad_mask.unsqueeze(-1) & pad_mask.unsqueeze(1)
    pairwise_mask = pairwise_mask.unsqueeze(1)
    if causal_mask is not None:
        pairwise_mask = pairwise_mask & causal_mask.unsqueeze(0).unsqueeze(1)

    positive_p = (p > 0).masked_fill(~pairwise_mask, False)
    total_per_head = pairwise_mask.sum(-1).sum(-1).float()
    num_selected_per_head = positive_p.sum(-1).sum(-1).float()
    positive_ratio_per_head = num_selected_per_head / total_per_head
    sparsity = 1 - positive_ratio_per_head.mean(0)
    return sparsity


def compute_exact_fraction(pred_p, gold_p, pad_mask, causal_mask=None):
    """
    Compute the number of queries for which the entmax graph was recovered exact,
    i.e., when we get 100% recall due to entmax top-k property:

    a = entmax([1.3, 20.0, 19.5, 1.0]) = [0.0000, 0.6740, 0.3260, 0.0000]
    b = entmax([0.0, 20.0, 19.5, 0.0]) = [0.0000, 0.6740, 0.3260, 0.0000]

    Args:
        pred_p: float tensor (batch, heads, n, n)
        gold_p: float tensor (batch, heads, n, n)
        pad_mask: bool tensor (batch, n)
        causal_mask: bool tensor  (n, n)
    """
    pairwise_mask = pad_mask.unsqueeze(-1) & pad_mask.unsqueeze(1)
    pairwise_mask = pairwise_mask.unsqueeze(1)
    if causal_mask is not None:
        pairwise_mask = pairwise_mask & causal_mask.unsqueeze(0).unsqueeze(1)

    # compute (micro-average) recall for each batch item then average
    gold_p = (gold_p > 0).masked_fill(~pairwise_mask, False)
    pred_p = (pred_p > 0).masked_fill(~pairwise_mask, False)
    matches = pred_p & gold_p

    # get fraction of exact entmax distribution for each query vector
    # recall == 1.0 means an exact recovery of entmax dist for that vector
    matches_per_query = matches.sum(-1).float()
    total_per_query = gold_p.sum(-1).float()
    recall_per_query = matches_per_query / total_per_query
    exact_per_query = recall_per_query == 1.0
    valid_exact_per_query = exact_per_query.masked_fill(~pad_mask.unsqueeze(1), False)
    lengths = pad_mask.sum(-1).unsqueeze(-1).float()
    exact_per_head = valid_exact_per_query.sum(-1).float() / lengths
    return exact_per_head.mean(0)

    # # get fraction of exact entmax graph (queries x keys)
    # matches_per_head = matches.sum(-1).sum(-1).float()
    # total_per_head = gold_p.sum(-1).sum(-1).float()
    # recall_per_head = matches_per_head / total_per_head
    # exact_per_graph_head = (recall_per_head == 1.0).float()
    # return exact_per_graph_head.mean(0)
