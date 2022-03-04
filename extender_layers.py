import itertools
import math
import pickle
from functools import partial

import torch
from torch import Tensor
from torch import nn
from entmax import Sparsemax, Entmax15, EntmaxBisect
# from joeynmt.better_sparsemax import BetterSparsemax as Sparsemax


class ExtenderAttention(nn.Module):

    """
    Context multihead attention for all sentences in the same document:
    doc_1 sent_1 attend doc_1 sent_1
    doc_1 sent_1 attend doc_1 sent_2
    doc_1 sent_1 attend doc_1 sent_n
    ...
    doc_1 sent_n attend doc_1 sent_1
    doc_1 sent_n attend doc_1 sent_2
    doc_1 sent_n attend doc_1 sent_n
    ...
    doc_m sent_1 attend up to doc_m sent_n
    doc_m sent_n attend up to doc_m sent_n

    Args:
        num_heads: number of attention heads
        size: hidden size. It must be a multiple of the number of heads
        dropout: dropout rate
        attn_func: one of "softmax", "sparsemax", "entmax", "entmax15"
        attn_alpha: alpha parameter of entmax. If any other attention function
            is used, this is ignored
        bucket_size: size, in number of tokens, of each bucket
        kwargs: ignored; here only for compatibility with other classes
    """
    def __init__(self,
                 num_heads: int,
                 size: int,
                 projection_dim: int,
                 bucket_size: int,
                 dropout: float = 0.,
                 attn_func: str = 'entmax',
                 attn_alpha: float = 1.5,
                 **kwargs
                 ):
        super().__init__()
        assert size % num_heads == 0

        self.head_size = head_size = size // num_heads
        self.num_heads = num_heads
        self.bucket_size = bucket_size

        self.key = nn.Linear(size, num_heads * head_size)
        self.value = nn.Linear(size, num_heads * head_size)
        self.query = nn.Linear(size, num_heads * head_size)

        self.output_layer = nn.Linear(size, size)

        Entmax = partial(EntmaxBisect, alpha=attn_alpha, n_iter=30)
        attn_funcs = {"softmax": nn.Softmax,
                      "sparsemax": Sparsemax,
                      "entmax15": Entmax15,
                      "entmax": Entmax}
        assert attn_func in attn_funcs, "Unknown attention function"
        self.transform = attn_funcs[attn_func](dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.proj_q = HeadwiseLinearProjection(
            num_heads, head_size, projection_dim)
        self.proj_k = HeadwiseLinearProjection(
            num_heads, head_size, projection_dim)
        self.centroids = None

    @classmethod
    def load(cls,
             state_dict_q,
             state_dict_k,
             query,
             key,
             value,
             bucket_size: int,
             dropout: float = 0.,
             attn_func: str = 'entmax',
             attn_alpha: float = 1.5,
             centroids: Tensor = None
             ):
        """
        Load pretrained projections and create a new instance of this class.

        Projections are modular, in the sense that they can be trained
        independently of the model

        Args:
            state_dict_q: the state dict for query projections
            state_dict_k: the state dict for key projections
        """
        w = state_dict_q['w']
        num_heads, head_dim, projection_dim = w.shape
        extender_attn = cls(
            num_heads, num_heads * head_dim, projection_dim, bucket_size,
            dropout, attn_func, attn_alpha)
        extender_attn.proj_q.load_state_dict(state_dict_q)
        extender_attn.proj_k.load_state_dict(state_dict_k)
        if centroids is not None:
            extender_attn.centroids = nn.Parameter(centroids.to(w.device))

        for param in extender_attn.proj_q.parameters():
            param.requires_grad = False

        for param in extender_attn.proj_k.parameters():
            param.requires_grad = False

        extender_attn.query = query  # todo check if gradient flows here, if needed for finetune
        extender_attn.key = key
        extender_attn.value = value

        return extender_attn

    def forward(self, k, v, q, mask=None):
        """
        Computes multi-headed attention.
        B = batch size
        H = number of heads
        L = seq length
        M = context length

        Params:
            k: keys   [B, M, D]
            v: values [B, M, D]
            q: query  [B, L, D]
            mask: optional mask [B, 1, M]. True in positions with actual
                content, False in padding positions.
        Returns:
            a tuple of tensors:
            - output: [B, L, D]
            - attention: the attention distribution by head, bucket and
                projection [B, H, buckets, projections, L, M]
        """
        batch_size = k.size(0)
        num_heads = self.num_heads
        k_seq_len = k.size(1)
        v_seq_len = v.size(1)
        q_seq_len = q.size(1)

        # project the queries (q), keys (k), and values (v)
        k = self.key(k)
        v = self.value(v)
        q = self.query(q)

        # split heads
        k = k.view(batch_size, k_seq_len, num_heads, self.head_size)
        v = v.view(batch_size, v_seq_len, num_heads, self.head_size)
        q = q.view(batch_size, q_seq_len, num_heads, self.head_size)

        # reshape q, k, v to [B, num_heads, L or M, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = q.transpose(1, 2)

        if self.centroids is not None:
            context, attention = self.compute_clustered_attention(q, k, v, mask)
        else:
            context, attention = self.compute_attentions(q, k, v, mask)
        output = self.output_layer(context)

        return output, attention

    def compute_clustered_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
        original_n = q.shape[2]

        # centroids.shape is (num_heads, num_runs, num_centroids, num_projections)
        centroids = self.centroids
        num_runs = centroids.shape[1]
        num_centroids = centroids.shape[2]
        device = q.device

        # pad to a multiple of bucket_size, if necessary
        # q = self.pad_tensor(q, num_centroids)
        # k = self.pad_tensor(k, num_centroids)
        # v = self.pad_tensor(v, num_centroids)
        # mask = self.pad_mask(mask, num_centroids)

        batch_size, num_heads, q_seq_len, head_size = q.shape
        _, _, k_seq_len, _ = k.shape

        # q_low.shape is (batch_size, num_heads, q_seq_len, num_projections)
        q_low = self.proj_q(q)
        # k_low.shape is (batch_size, num_heads, k_seq_len, num_projections)
        k_low = self.proj_k(k)
        num_projections = q_low.shape[-1]

        assert centroids.shape == (num_heads, num_runs, num_centroids, num_projections)

        # add `batch` dimension
        # (batch_size, num_heads, num_runs, num_centroids, num_projections)
        expanded_centroids = centroids.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

        # add `num_runs` dimension
        # (batch_size, num_heads, 1, q_seq_len, num_projections)
        expanded_q_low = q_low.unsqueeze(2)
        # (batch_size, num_heads, 1, k_seq_len, num_projections)
        expanded_k_low = k_low.unsqueeze(2)

        # q_dists.shape is (batch, num_heads, num_runs, q_seq_len, num_centroids)
        q_dists = torch.cdist(expanded_q_low, expanded_centroids, p=2)
        # k_dists.shape is (batch, num_heads, num_runs, k_seq_len, num_centroids)
        k_dists = torch.cdist(expanded_k_low, expanded_centroids, p=2)

        # q_clustered.shape is (batch, num_heads, num_runs, q_seq_len)
        q_clustered = torch.argmin(q_dists, dim=-1)
        # k_clustered.shape is (batch, num_heads, num_runs, k_seq_len)
        k_clustered = torch.argmin(k_dists, dim=-1)

        # transpose to get `num_runs` as different hashing rounds
        # q_clustered.shape is (batch, num_heads, q_seq_len, num_runs)
        q_clustered = q_clustered.transpose(2, 3)
        # k_clustered.shape is (batch, num_heads, k_seq_len, num_runs)
        k_clustered = k_clustered.transpose(2, 3)

        # set cluster id for padding positions as `num_centroids` (ids start with 0)
        q_clustered.masked_fill_(~mask.view(batch_size, 1, q_seq_len, 1), num_centroids)
        k_clustered.masked_fill_(~mask.view(batch_size, 1, k_seq_len, 1), num_centroids)

        # we need to divide q_clustered into
        # (batch, num_heads, num_centroids, max_cluster_size_q_for_all_batch_and_heads, num_runs)

        # overall, num_runs = 1, so we dont need to do q_clustered.unsqueeze(-1) to broadcast
        # q_clustered_bin.shape is (batch, num_heads, q_seq_len, num_centroids)
        q_clustered_bin = q_clustered == torch.arange(num_centroids).to(device)
        # k_clustered_bin.shape is (batch, num_heads, k_seq_len, num_centroids)
        k_clustered_bin = k_clustered == torch.arange(num_centroids).to(device)

        # q_clustered_bin.shape is (batch, num_heads, num_centroids, q_seq_len)
        q_clustered_bin = q_clustered_bin.transpose(-1, -2).int()
        # k_clustered_bin.shape is (batch, num_heads, num_centroids, k_seq_len)
        k_clustered_bin = k_clustered_bin.transpose(-1, -2).int()

        # get the max cluster size across all batches and heads
        max_cluster_size_q = q_clustered_bin.sum(-1).max().item()
        max_cluster_size_k = k_clustered_bin.sum(-1).max().item()

        # `q_clustered_vals` contains only 0 or 1 ints (due to one hot binarization)
        q_clustered_vals, q_clustered_idxs = q_clustered_bin.sort(dim=-1, descending=True)
        k_clustered_vals, k_clustered_idxs = k_clustered_bin.sort(dim=-1, descending=True)

        # values that are 0 correspond to padding positions, so we mask them with q_seq_len - 1 (last token)
        q_clustered_idxs[~q_clustered_vals.bool()] = q_seq_len - 1
        k_clustered_idxs[~k_clustered_vals.bool()] = k_seq_len - 1

        # get 0 and 1s as masks
        mask_clustered_q = q_clustered_vals.bool()
        mask_clustered_k = k_clustered_vals.bool()

        # q_bucketed.shape is (batch, num_heads, num_centroids, max_cluster_size_q)
        q_bucketed = q_clustered_idxs[:, :, :, :max_cluster_size_q]
        # k_bucketed.shape is (batch, num_heads, num_centroids, max_cluster_size_k)
        k_bucketed = k_clustered_idxs[:, :, :, :max_cluster_size_k]
        # same shape as above
        mask_bucketed_q = mask_clustered_q[:, :, :, :max_cluster_size_q]
        mask_bucketed_k = mask_clustered_k[:, :, :, :max_cluster_size_k]
        # create pairwise mask with shape (batch, num_heads, num_centroids, max_cluster_size_q, max_cluster_size_k)
        mask_bucketed = mask_bucketed_q.unsqueeze(-1) & mask_bucketed_k.unsqueeze(-2)

        # (batch, num_heads, num_clusters * max_cluster_size)
        squished_inds_q = q_bucketed.reshape(batch_size, num_heads, -1)
        squished_inds_k = k_bucketed.reshape(batch_size, num_heads, -1)

        # keys and values are bucketed with the same buckets
        # the bucketed tensors are (batch, num_heads, num_clusters * max_cluster_size, head_size)
        bucketed_q = q.gather(2, squished_inds_q.unsqueeze(-1).expand(-1, -1, -1, head_size))
        bucketed_k = k.gather(2, squished_inds_k.unsqueeze(-1).expand(-1, -1, -1, head_size))
        bucketed_v = v.gather(2, squished_inds_k.unsqueeze(-1).expand(-1, -1, -1, head_size))

        # we now expand the squished dim into (num_centroids, max_cluster_size)
        bucketed_q = bucketed_q.view(batch_size, num_heads, num_centroids, -1, head_size)
        bucketed_k = bucketed_k.view(batch_size, num_heads, num_centroids, -1, head_size)
        bucketed_v = bucketed_v.view(batch_size, num_heads, num_centroids, -1, head_size)

        # dots are (batch, num_heads, num_centroids, max_cluster_size_q, max_cluster_size_k)
        sqrt_d = head_size ** 0.5
        dots = bucketed_q @ bucketed_k.transpose(-1, -2) / sqrt_d

        # mask the dots past key length; add `max_cluster_size_q` dim for broadcasting
        neg_inf = -9999999
        dots = dots.masked_fill(~mask_bucketed, neg_inf)  # float('-inf') will generate nans in softmax

        # att_dist is (batch, num_heads, num_centroids, max_cluster_size_q, max_cluster_size_k)
        att_dist = self.transform(dots)

        # fix the uniform numbers for padding positions
        att_dist = att_dist * mask_bucketed.float()

        # output is (batch, num_heads, num_centroids, max_cluster_size_q, head_size)
        output = torch.matmul(att_dist, bucketed_v)

        # make sure squashed indices for pad positions are higher than last valid token id
        squished_mask_q = mask_bucketed_q.reshape(batch_size, num_heads, -1)
        # squished_mask_k = mask_bucketed_k.reshape(batch_size, num_heads, -1)
        fixed_squished_inds_q = squished_inds_q.masked_fill(~squished_mask_q, q_seq_len + 1)
        # fixed_squished_inds_k = squished_inds_q.masked_fill(~squished_mask_k, k_seq_len + 1)

        # get indices of valid contextualized query vectors
        rev_inds_q = fixed_squished_inds_q.argsort(-1)
        # truncate to get only the first q_seq_len vectors -> the valid ones
        rev_inds_q = rev_inds_q[:, :, :q_seq_len]

        # squish output and gather correct vectors
        squished_output = output.view(batch_size, num_heads, -1, head_size)
        # output.shape is (batch, num_heads, q_seq_len, head_size)
        output = squished_output.gather(2, rev_inds_q.unsqueeze(-1).expand(-1, -1, -1, head_size))

        # concat heads back
        output = output.transpose(1, 2).reshape(batch_size, -1, num_heads * head_size)

        # get stats
        att_stats = compute_graph_stats(q_clustered, k_clustered, num_centroids=num_centroids, mask=mask)

        return output, att_stats

    def pad_tensor(self, x, num_centroids):
        """
        Insert zeros to get a number of timesteps that is multiple of num_centroids.
        Later we will divide into pieces of num_centroids x num_max_elements_in_a_centroid

        :param x: Tensor of shape (batch, heads, n, head_size)
        :param num_centroids: int, number of clusters in kmeans
        :return:
        """
        batch_size, num_heads, original_n, head_size = x.shape
        modulo = original_n % num_centroids
        if modulo != 0:
            diff = num_centroids - modulo
            padding = torch.zeros(batch_size, num_heads, diff, head_size, device=x.device)
            x = torch.cat([x, padding], dim=2)
        return x

    def pad_mask(self, mask, num_centroids):
        """
        Same as `pad_tensor` but for the mask.
        todo: we need to take special care for the decoder self-attn casual mask

        :param mask: Tensor of shape (batch, 1 or n, n)
        :param num_centroids: int, number of clusters in kmeans
        :return:
        """
        batch_size, _, original_n = mask.shape
        modulo = original_n % num_centroids
        if modulo != 0:
            diff = num_centroids - modulo
            mask_padding = torch.zeros(batch_size, 1, diff, dtype=torch.bool, device=mask.device)
            mask = torch.cat([mask, mask_padding], dim=2)
        return mask

    def compute_attentions(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
        """
        Split queries, keys and values into bucket.

        IMPORTANT: THE CURRENT IMPLEMENTATION ASSUMES SELF-ATTENTION,
        SO LENGTH_K == LENGTH_Q

        Queries are [batch, heads, length, head_dim]
        Keys are [batch, heads, context_length, head_dim]
        Values are [batch, heads, context_length, head_dim]

        Args:
            q: tensor with shape (batch, heads, length, head_dim)
            k: tensor with shape (batch, heads, length, head_dim)
            v: tensor with shape (batch, heads, length, head_dim)
            mask: tensor (B, 1, L)
        """
        batch_size, num_heads, original_n, head_size = q.shape

        # pad to a multiple of bucket_size, if necessary
        modulo = original_n % self.bucket_size
        if modulo != 0:
            # pad
            shape = [batch_size, num_heads, self.bucket_size - modulo,
                     head_size]
            padding = torch.zeros(shape, device=q.device)
            q = torch.cat([q, padding], dim=2)
            k = torch.cat([k, padding], dim=2)
            v = torch.cat([v, padding], dim=2)

            mask_padding = torch.zeros(
                [batch_size, 1, self.bucket_size - modulo], dtype=torch.bool,
                device=q.device)
            mask = torch.cat([mask, mask_padding], dim=2)

        # n possibly changed after padding
        n = q.shape[2]
        q_low = self.proj_q(q)
        k_low = self.proj_k(k)
        num_projections = q_low.shape[-1]

        # fill the padding positions with inf so that they will be pushed to the
        # last bucket, not spread across all of them
        q_low.masked_fill_(~mask.view(batch_size, 1, n, 1), float('inf'))
        k_low.masked_fill_(~mask.view(batch_size, 1, n, 1), float('inf'))

        # each round will have all tokens sorted by value
        inds_q = q_low.argsort(2)
        inds_kv = k_low.argsort(2)
        rev_inds_q = inds_q.argsort(2)

        # (batch, heads, n * projections, 1)
        squished_inds_q = inds_q.view(batch_size, num_heads, -1, 1)
        squished_inds_kv = inds_kv.view(batch_size, num_heads, -1, 1)

        # keys and values are bucketed with the same buckets
        # the bucketed tensors are (batch, heads, n * projections, dim)
        bucketed_q = q.gather(2, squished_inds_q.expand(-1, -1, -1, head_size))
        bucketed_k = k.gather(2, squished_inds_kv.expand(-1, -1, -1, head_size))
        bucketed_v = v.gather(2, squished_inds_kv.expand(-1, -1, -1, head_size))

        # we now expand the squished dim into
        # [num_buckets, bucket_size, num_projections]
        shape = [batch_size, num_heads, -1, self.bucket_size,
                 num_projections, head_size]
        bucketed_q = bucketed_q.view(shape)
        bucketed_k = bucketed_k.view(shape)
        bucketed_v = bucketed_v.view(shape)

        # (batch, heads, buckets, projections, bucket_size, dim)
        bucketed_q.transpose_(-3, -2)
        bucketed_k.transpose_(-3, -2)
        bucketed_v.transpose_(-3, -2)

        # dots are (batch, heads, num_buckets, projections, bucket, bucket)
        c = head_size ** 0.5
        dots = bucketed_q @ bucketed_k.transpose(-1, -2) / c

        # mask the dots past key length
        mask6d = mask.view(batch_size, 1, -1, 1, 1, self.bucket_size)
        dots.masked_fill_(~mask6d, -float('inf'))

        # the current implementation of entmax fails when all inputs are -inf
        # and this happens when a bucket has only padding. In that case, let's
        # unpad those buckets -- they will be ignored later on anyway
        only_padding = mask6d.sum(-1) == 0
        dots.masked_fill_(only_padding.unsqueeze(-1), 0)

        # aggregate over all projections with a softmax over all projections
        # first, compute the partition function for each projection
        # log_z is (batch, heads, buckets, proj, bucket_size, 1)
        log_z = torch.logsumexp(dots, -1, keepdim=True)

        # coefficients to weight each projection; coeff.sum(3) == 1
        # shape is (batch, heads, buckets, projections, bucket_size, 1)
        projection_coeff = torch.softmax(log_z, dim=3)

        # att is (batch, heads, buckets, projections, bucket_size, dim)
        att_dist = self.transform(dots) * projection_coeff
        att = att_dist @ bucketed_v

        # each projection used a different sorting of queries and keys
        # let's unsort the attentions to the original ordering
        # make rev_inds (batch, heads, projections, n, 1)
        rev_inds_q = rev_inds_q.transpose(2, 3).unsqueeze(-1)

        # (batch, heads, projections, n, dim)
        backsorted_att = self._unbucket_and_sort(att, rev_inds_q)

        # now sum over projections (they are already weighted)
        # (batch, heads, n, dim)
        output = backsorted_att.sum(2)

        # concatenate heads and remove the bucket padding
        output = output.transpose(1, 2).reshape(batch_size, n, -1)
        output = output[:, :original_n]

        b_q = squished_inds_q.view(batch_size, num_heads, n, num_projections)
        b_kv = squished_inds_kv.view(batch_size, num_heads, n, num_projections)
        att_dist = compute_graph_stats(b_q, b_kv, mask=mask)

        return output, att_dist

    def _unbucket_and_sort(self, data: Tensor, indices: Tensor) -> Tensor:
        """
        Auxiliary function to unbucket bucketed data (attention distributions
        or attended values) and sort them according to indices.

        Args:
            data: tensor with shape (batch, heads, buckets, projections,
                bucket_size, dim); the last dim is irrelevant
            indices: tensor with integer indices to sort each projection round
                (batch, heads, projections, n, 1)
        """
        batch_size, num_heads, _, num_projections, _, dim = data.shape

        # unbucket to (batch, heads, projections, n, dim)
        data = data.transpose(2, 3)
        data = data.reshape(batch_size, num_heads, num_projections, -1, dim)

        # (batch, heads, projections, n, dim)
        sorted_data = data.gather(3, indices.expand(-1, -1, -1, -1, dim))

        return sorted_data


class HeadwiseSkipProjection(nn.Module):
    """
    Just do f(x) = x
    but also declare some params to not break the entire code.

    :param num_heads: number of attention heads
    :param head_dim: hidden dimension size of each head
    :param proj_dim: dimensionality to project queries and keys to; it
        corresponds to a number of hashing rounds.
    """

    def __init__(self, num_heads: int, head_dim: int, proj_dim: int):
        super(HeadwiseSkipProjection, self).__init__()
        self.num_heads = num_heads
        self.proj_dim = proj_dim
        self.w = nn.Parameter(torch.Tensor(num_heads, head_dim, proj_dim))
        self.b = nn.Parameter(torch.Tensor(num_heads, 1, proj_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # same initialization as pytorch Linear layer
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.b.shape[1])
        nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        """:param x: (batch, num_heads, length, head_dim)"""
        return x

    def extra_repr(self):
        return 'heads={}, in_features={}, out_features={}'.format(
            self.w.shape[0], self.w.shape[1], self.w.shape[2])


class HeadwiseLinearProjection(nn.Module):
    """
    Class to perform efficient matrix multiplication for projecting the queries
    and keys of each head to a lower dimensionality, such that each head has
    its own set of weights.

    :param num_heads: number of attention heads
    :param head_dim: hidden dimension size of each head
    :param proj_dim: dimensionality to project queries and keys to; it
        corresponds to a number of hashing rounds.
    """

    def __init__(self, num_heads: int, head_dim: int, proj_dim: int):
        super(HeadwiseLinearProjection, self).__init__()
        self.num_heads = num_heads
        self.proj_dim = proj_dim
        self.w = nn.Parameter(torch.Tensor(num_heads, head_dim, proj_dim))
        self.b = nn.Parameter(torch.Tensor(num_heads, 1, proj_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # same initialization as pytorch Linear layer
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.b.shape[1])
        nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        """:param x: (batch, num_heads, length, head_dim)"""
        return torch.einsum('bhld,hdp->bhlp', x, self.w) + self.b

    def extra_repr(self):
        return 'heads={}, in_features={}, out_features={}'.format(
            self.w.shape[0], self.w.shape[1], self.w.shape[2])


class HeadwiseMLPProjection(nn.Module):
    """
    Class to perform efficient matrix multiplication for projecting the queries
    and keys of each head to a lower dimensionality, such that each head has
    its own set of weights.

    :param num_heads: number of attention heads
    :param head_dim: hidden dimension size of each head
    :param proj_dim: dimensionality to project queries and keys to; it
        corresponds to a number of hashing rounds.
    """

    def __init__(
            self, num_heads: int, head_dim: int, proj_dim: int,
            hidden_size: int):
        super(HeadwiseMLPProjection, self).__init__()
        self.num_heads = num_heads
        self.proj_dim = proj_dim
        self.w1 = nn.Parameter(torch.Tensor(num_heads, head_dim, hidden_size))
        self.b1 = nn.Parameter(torch.Tensor(num_heads, 1, hidden_size))
        self.w2 = nn.Parameter(torch.Tensor(num_heads, hidden_size, proj_dim))
        self.b2 = nn.Parameter(torch.Tensor(num_heads, 1, proj_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # same initialization as pytorch Linear layer
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        bound1 = 1 / math.sqrt(self.b1.shape[1])
        bound2 = 1 / math.sqrt(self.b2.shape[1])
        nn.init.uniform_(self.b1, -bound1, bound1)
        nn.init.uniform_(self.b2, -bound2, bound2)

    def forward(self, x):
        """:param x: (batch, num_heads, length, head_dim)"""
        h = torch.einsum('bhld,hdp->bhlp', x, self.w1) + self.b1
        h = torch.tanh(h)
        h = torch.einsum('bhld,hdp->bhlp', h, self.w2) + self.b2
        return h

    def extra_repr(self):
        return 'heads={}, in_features={}, hidden={}, out_features={}'.format(
            self.w1.shape[0], self.w1.shape[1], self.w1.shape[2],
            self.w2.shape[2])


def compute_graph_stats(
        buckets_q: Tensor, buckets_k: Tensor, num_centroids: int = None,
        mask: Tensor = None, window_inds: Tensor = None):
    """
    Compute the graph sparsity across all projection rounds; i.e., as if queries
    could look at any key it shared at least one bucket with.

    Args:
        buckets_q: Tensor (batch, heads, n, projections)
        buckets_k: Tensor (batch, heads, n, projections)
        mask: Tensor (batch, 1, n)
        window_inds: Tensor (n, window_size) with the positions attended
           inside a window around each query
    """
    batch_size, num_heads, n, num_rounds = buckets_q.shape
    device = buckets_q.device

    # cross all Q and K to find out when they match
    # shape is (batch, heads, query, key, projection)
    qk = buckets_q.unsqueeze(3) == buckets_k.unsqueeze(2)

    # consider matches across any projection; (batch, heads, query, key)
    qk = qk.sum(4) > 0

    if window_inds is not None:
        r = torch.arange(n).view(-1, 1).to(qk.device)
        qk[:, :, r, window_inds] = True

    qk.masked_fill_(~mask.view(batch_size, 1, n, 1), False)
    qk.masked_fill_(~mask.view(batch_size, 1, 1, n), False)

    lengths = mask.int().sum(-1).squeeze()
    links_per_head = qk.sum(3).sum(2)
    mean_links = links_per_head.float().mean(1)
    sparsity = 1 - (mean_links / lengths ** 2)

    # compute entropies
    if num_centroids is not None:
        entropies = {'q_avg': [], 'k_avg': [], 'q_min': [], 'k_min': []}
        q_counts = (buckets_q.unsqueeze(-1) == torch.arange(num_centroids).view(1, 1, 1, 1, -1).to(device)).sum(dim=2)
        k_counts = (buckets_k.unsqueeze(-1) == torch.arange(num_centroids).view(1, 1, 1, 1, -1).to(device)).sum(dim=2)
        q_cluster_probs = q_counts.float() / lengths.view(-1, 1, 1, 1)
        k_cluster_probs = k_counts.float() / lengths.view(-1, 1, 1, 1)

        change_log_base = torch.log(torch.tensor(num_centroids, dtype=torch.float))
        q_logs, k_logs = torch.log(q_cluster_probs), torch.log(k_cluster_probs)
        q_logs[q_logs == float('-inf')], k_logs[k_logs == float('-inf')] = 0, 0
        q_cluster_ent = -1 * (q_cluster_probs * q_logs / change_log_base).sum(-1)
        k_cluster_ent = -1 * (k_cluster_probs * k_logs / change_log_base).sum(-1)

        entropies['q_avg'] = q_cluster_ent.view(batch_size, -1).sum(-1) / (num_heads * num_rounds)
        entropies['k_avg'] = k_cluster_ent.view(batch_size, -1).sum(-1) / (num_heads * num_rounds)
        entropies['q_min'] = q_cluster_ent.view(batch_size, -1).min(-1)[0]
        entropies['k_min'] = k_cluster_ent.view(batch_size, -1).min(-1)[0]
    else:
        entropies = None

    return sparsity, entropies


def load_projectors(path):
    if not torch.cuda.is_available():
        map_location = torch.device('cpu')
    else:
        map_location = None
    return torch.load(path, map_location=map_location)


def get_layer_projectors(loaded_projs, layer):
    l_projs = loaded_projs[layer]
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
        raise TypeError("unknown projector")
    proj_q.load_state_dict(proj_q_dict)
    proj_k.load_state_dict(proj_k_dict)
    if torch.cuda.is_available():
        proj_q.cuda()
        proj_k.cuda()
    return proj_q, proj_k
