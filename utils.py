import math
import random

import numpy as np
import torch
from entmax import entmax15


def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def get_window_positions(n, window, causal=False):
    """
    Create a tensor (n, window) such that the i-th row has the indices of the
    columns attended by it. E.g.,

    [[2, 0, 1],  # shifts as padding
     [0, 1, 2],
     [1, 2, 3],
     [2, 3, 4],
     ...]
    """
    half_window = window // 2

    # form indices for columns in an attention window pattern
    # e.g. [0, 1, 2], [1, 2, 3], [2, 3, 4] etc
    r = torch.arange(n).view(-1, 1)
    attended = r + torch.arange(-half_window, half_window + 1)

    # make the windows at the first and last few words the same size as in
    # the middle
    attended[attended < 0] += window
    attended[attended >= n] -= window

    if causal:
        attended = attended[:, :window//2 + 1]

    return attended


def get_length_mask(lengths, max_length=None):
    """Create a (batch_size, max_length) boolean mask
    True for true positions and False for padding"""
    if max_length is None:
        max_length = lengths.max()
    r = torch.arange(max_length).unsqueeze(0).to(lengths.device)
    mask = r < lengths.unsqueeze(-1)
    return mask


def get_mask(num_positive, length):
    r = torch.arange(length).unsqueeze(0).to(num_positive.device)
    mask = r < num_positive.unsqueeze(1)
    return mask


def get_key_mask(num_positive, length=None):
    """Mask positions after num_positions, for each token/head/sample"""
    assert num_positive.ndim == 3
    if length is None:
        length = num_positive.shape[-1]
    r = torch.arange(length).view(1, 1, 1, -1).to(num_positive.device)
    mask = r < num_positive.unsqueeze(-1)
    return mask


def neighbours_mask(size, window_size):
    """Mask for neighbour positions.
    Args:
        size (int): squared tensor size
        window_size (int): how many elements to be considered as valid around
            the ith element (including ith).
    Returns:
        torch.Tensor: (size, size)
    """
    z = torch.ones(size, size, dtype=torch.uint8)
    mask = (torch.triu(z, diagonal=1 + window_size // 2)
            + torch.tril(z, diagonal=- window_size // 2))
    return z - mask


def subsequent_mask(size):
    """Mask out subsequent positions.
    Args:
        size (int): squared tensor size
    Returns:
        torch.Tensor: (size, size)
    """
    return torch.tril(torch.ones(size, size, dtype=torch.uint8))


def unsqueeze_as(tensor, as_tensor, dim=-1):
    """Expand new dimensions based on a template tensor along `dim` axis.
    Args:
        Args:
        tensor (torch.Tensor): tensor with shape (bs, ..., d1)
        as_tensor (torch.Tensor): tensor with shape (bs, ..., n, ..., d2)
    Returns:
        torch.Tensor: (bs, ..., 1, ..., d1)
    """
    while tensor.dim() < as_tensor.dim():
        tensor = tensor.unsqueeze(dim)
    return tensor


def dot_product_and_mask(q, k, lengths=None, mask_value=-float('inf'), causal=False):
    """
    Args:
        q: tensor (batch, heads, n, dim)
        k: tensor (batch, heads, n, dim)
        lengths: tensor (batch,)
        mask_value: value for padding positions
    """
    # (batch, heads, n, n)
    dots = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5

    if lengths is None:
        return dots

    # mask out padding positions - mask is (batch, n)
    mask = get_length_mask(lengths, q.shape[2])  # add head and query dim
    if causal:
        # pad_mask = mask.unsqueeze(1).unsqueeze(-1) & mask.unsqueeze(1).unsqueeze(2)
        pad_mask = mask.unsqueeze(1).unsqueeze(2)
        sub_mask = subsequent_mask(q.shape[2]).bool().to(pad_mask.device)
        sub_mask = unsqueeze_as(sub_mask, pad_mask, dim=0)  # add batch and head dims to sub_mask
        mask = pad_mask & sub_mask
    else:
        mask = mask.unsqueeze(1).unsqueeze(2)

    dots.masked_fill_(~mask, mask_value)

    return dots


def get_ground_truth_for_tied_inputs(att_dist, lengths, causal=False):
    # fix case for concat_q_and_k:
    # if   qi -> kj
    # then qj -> ki
    n = att_dist.shape[-1]
    device = att_dist.device
    eps = att_dist[att_dist > 0].min().item() / 2

    # symmetric att_dist
    att_dist_sym = att_dist.masked_fill((att_dist > 0).transpose(-1, -2), eps)

    # mask padding and causal
    pad_mask = get_length_mask(lengths, n).unsqueeze(1).unsqueeze(2)
    if causal:
        sub_mask = subsequent_mask(n).bool().to(device)
        sub_mask = unsqueeze_as(sub_mask, pad_mask, dim=0)  # add batch and head dims to sub_mask
        mask = pad_mask & sub_mask
        att_dist_sym.masked_fill_(~mask, 0)
    else:
        att_dist_sym.masked_fill_(~pad_mask, 0)

    # number of non-null keys for each query
    num_positive_new = (att_dist_sym > 0).sum(-1)
    num_positive_new.masked_fill_(~pad_mask.squeeze(2), 0)

    # sorted indices
    sorted_k_inds_new = att_dist_sym.argsort(-1, descending=True)

    return sorted_k_inds_new, num_positive_new


def get_ground_truth(att_dist, lengths):
    """
    Args:
        att_dist: tensor (batch, heads, n, n)
        lengths: tensor (batch,)
    """
    # number of non-null keys for each query
    num_positive = (att_dist > 0).sum(-1)
    inds = att_dist.argsort(-1, descending=True)

    # zero positive counts past sentence end
    # no need to consider causal, because att_dist[causal_positions]=0
    mask = get_length_mask(lengths, att_dist.shape[-1])
    num_positive.masked_fill_(~mask.unsqueeze(1), 0)

    return inds, num_positive


def update_mean(current_mean, computed_samples, batch_mean, batch_size):
    """
    Computes an accumulated average in O(1).
    """
    assert (current_mean is None and computed_samples == 0) or \
           (current_mean is not None and computed_samples > 0), \
        '"current_mean is None" requires "computed samples==0" and vice-versa'
    if current_mean is None:
        return batch_mean
    else:
        updated_mean = (current_mean * computed_samples + batch_mean * batch_size) / (computed_samples + batch_size)
        return updated_mean


def append_tensor_cpu(original_tensor, batch):
    """Concat tensors in cpu"""
    batch = batch.cpu()
    if original_tensor is None:
        return batch
    else:
        return torch.cat([original_tensor, batch], dim=0)


def update_graph_stats(graph_stats, computed_samples, batch_graph_stats, batch_size):
    assert (graph_stats is None and computed_samples == 0) or \
           (graph_stats is not None and computed_samples > 0), \
        '"graph_stats is None" requires "computed samples==0" and vice-versa'

    batch_graph_stats['edge_distances'] = batch_graph_stats['edge_distances'].cpu()
    if graph_stats is None:
        return batch_graph_stats
    else:
        graph_stats['sparsity'] = update_mean(graph_stats['sparsity'], computed_samples,
                                              batch_graph_stats['sparsity'], batch_size)
        graph_stats['edge_distances'] = torch.cat([graph_stats['edge_distances'],
                                                   batch_graph_stats['edge_distances']], dim=0)
        return graph_stats


def blockify(q_low, k_low, lengths, att_dist=None, block_size=1):
    batch_size, num_heads, n, proj_size = q_low.shape

    # shortcuts
    b = block_size
    n_blocks = math.ceil(n / float(b))

    # add padding to be divisible by block size
    if n % b > 0:
        padding = torch.full([batch_size, num_heads, b - n % b, q_low.shape[-1]], 0, device=q_low.device)
        q_low = torch.cat([q_low, padding], dim=2)
        k_low = torch.cat([k_low, padding], dim=2)
        if att_dist is not None:
            padding = torch.full([batch_size, num_heads, b - n % b, n], 0, device=q_low.device)
            att_dist = torch.cat([att_dist, padding], dim=2)
            padding = torch.full([batch_size, num_heads, n + b - n % b, b - n % b], 0, device=q_low.device)
            att_dist = torch.cat([att_dist, padding], dim=3)

    # reshape to contiguous chunks
    # (batch, heads, n, hdim) -> (batch, heads, n_blocks, hdim * block_size)
    q_low = q_low.reshape(batch_size, num_heads, n_blocks, -1)
    # (batch, heads, n, hdim) -> (batch, heads, n_blocks, hdim * block_size)
    k_low = k_low.reshape(batch_size, num_heads, n_blocks, -1)

    # lengths will be equal to (lengths + block_size - lengths % block_size) // block_size
    lengths = torch.ceil(lengths.float() / b).long()

    # recalculate attention info for chunked inputs
    # (batch, heads, n, n) -> (batch, heads, n_blocks, n_blocks)
    # chunk_and_sum
    if att_dist is not None:
        att_dist = att_dist.reshape(batch_size, num_heads, att_dist.shape[2], n_blocks, -1).sum(-1)
        att_dist = att_dist.reshape(batch_size, num_heads, n_blocks, -1, n_blocks).sum(-2)

    return q_low, k_low, lengths, att_dist


def unblockify_attn(att_dist, block_size=1, pad_mask=None, causal_mask=None):
    # (batch, heads, n_blocks, n_blocks) -> (batch, heads, n, n)
    att_dist = att_dist.repeat_interleave(block_size, dim=-1).repeat_interleave(block_size, dim=-2)
    # mask out padding and "future" positions
    if pad_mask is not None:
        pairwise_mask = pad_mask.unsqueeze(-1) & pad_mask.unsqueeze(1)
        pairwise_mask = pairwise_mask.unsqueeze(1)
        if causal_mask is not None:
            pairwise_mask = pairwise_mask & causal_mask.unsqueeze(0).unsqueeze(1)
        att_dist.masked_fill(~pairwise_mask, 0)
    return att_dist
