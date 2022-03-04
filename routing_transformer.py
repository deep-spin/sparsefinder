# from: https://github.com/lucidrains/routing-transformer/blob/master/routing_transformer/routing_transformer.py

import os
import math
from inspect import isfunction

import torch
from torch import nn
import torch.nn.functional as F


# constants
KMEAN_INIT_ITERS = 10


# helper functions
def exists(val):
    return val is not None


def default(x, d):
    if not exists(x):
        return d if not isfunction(d) else d()
    return x


def to(t):
    return {"device": t.device, "dtype": t.dtype}


def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]


def is_empty(t):
    return t.nelement() == 0


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(2, expand_dim(indices, -1, last_dim))


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]


def ema(old, new, decay):
    if not exists(old):
        return new
    return old * decay + new * (1 - decay)


def ema_inplace(moving_avg, new, decay):
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def similarity(x, means):
    return torch.einsum("bhld,hcd->bhlc", x, means)


def dists_and_buckets(x, means):
    dists = similarity(x, means)
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets


def batched_bincount(index, num_classes, dim=-1):
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out


def update_kmeans_on_backwards(module):
    module.kmean_modules = find_modules(module, Kmeans)
    def hook(_, grad_in, grad_out):
        for m in module.kmean_modules:
            m.update()
    return module.register_backward_hook(hook)


def kmeans_iter(x, means, buckets=None):
    b, h, l, d, dtype, num_clusters = *x.shape, x.dtype, means.shape[1]

    if not exists(buckets):
        _, buckets = dists_and_buckets(x, means)

    bins = batched_bincount(buckets, num_clusters).sum(0, keepdim=True)
    zero_mask = bins.long() == 0

    means_ = buckets.new_zeros(b, h, num_clusters, d, dtype=dtype)
    means_.scatter_add_(-2, expand_dim(buckets, -1, d), x)
    means_ = F.normalize(means_.sum(0, keepdim=True), dim=-1).type(dtype)

    means = torch.where(zero_mask.unsqueeze(-1), means, means_)
    means = means.squeeze(0)
    return means


def distribution(dists, window_size):
    _, topk_indices = dists.topk(k=window_size, dim=-2)
    indices = topk_indices.transpose(-2, -1)
    return indices.reshape(*indices.size()[:2], -1)


class Kmeans(nn.Module):
    def __init__(self, num_heads, head_dim, num_clusters, ema_decay=0.999, commitment=1e-4):
        super().__init__()
        self.commitment = commitment
        self.ema_decay = ema_decay

        self.register_buffer("means", torch.randn(num_heads, num_clusters, head_dim))
        self.register_buffer("initted", torch.tensor(False))
        self.num_new_means = 0
        self.new_means = None

    @torch.no_grad()
    def load_means(self, means):
        self.means.data.copy_(means)
        self.initted.data.copy_(torch.tensor(True))

    @torch.no_grad()
    def init(self, x):
        if self.initted:
            return
        _, h, _, d, device, dtype = *x.shape, x.device, x.dtype

        num_clusters = self.means.shape[1]

        means = x.transpose(0, 1).contiguous().view(h, -1, d)
        num_samples = means.shape[1]

        if num_samples >= num_clusters:
            indices = torch.randperm(num_samples, device=device)[:num_clusters]
        else:
            indices = torch.randint(0, num_samples, (num_clusters,), device=device)

        means = means[:, indices]

        for _ in range(KMEAN_INIT_ITERS):
            means = kmeans_iter(x, means)

        self.num_new_means = 0
        self.means.data.copy_(means)
        self.initted.data.copy_(torch.tensor(True))

    @torch.no_grad()
    def update(self, new_means=None):
        new_means = default(new_means, self.new_means)
        assert exists(new_means), "new kmeans has not been supplied"
        ema_inplace(self.means, new_means, self.ema_decay)

        del self.new_means
        self.new_means = None
        self.num_new_means = 0

    def forward(self, x, update_means=False):
        self.init(x)

        b, dtype = x.shape[0], x.dtype
        means = self.means.type(dtype)
        x = F.normalize(x, 2, dim=-1).type(dtype)

        with torch.no_grad():
            dists, buckets = dists_and_buckets(x, means)

        routed_means = batched_index_select(expand_dim(means, 0, b), buckets)
        loss = F.mse_loss(x, routed_means) * self.commitment

        if update_means:
            with torch.no_grad():
                means = kmeans_iter(x, means, buckets)
            self.new_means = ema(self.new_means, means, self.num_new_means / (self.num_new_means + 1))
            self.num_new_means += 1

        return dists, loss


class KmeansAttention(nn.Module):
    def __init__(
        self,
        num_clusters,
        window_size,
        num_heads,
        head_dim,
        ema_decay=0.999,
        commitment=1e-4,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_clusters = num_clusters
        self.head_dim = head_dim
        self.window_size = window_size
        self.kmeans = Kmeans(num_heads, head_dim, num_clusters, ema_decay, commitment)

    def update_kmeans(self):
        self.kmeans.update()

    def forward(self, q, k, window_size=None, update_kmeans=False):
        b, h, t, d = q.shape
        kv_t = k.shape[2]
        wsz = self.window_size if window_size is None else window_size
        nc = self.num_clusters

        kv_wsz = wsz
        wsz = min(wsz, t)
        kv_wsz = min(kv_wsz, kv_t)

        dists, aux_loss = self.kmeans(torch.cat((q, k), dim=2), update_kmeans)
        q_dists, k_dists = split_at_index(2, t, dists)
        indices = distribution(q_dists, wsz)
        kv_indices = distribution(k_dists, kv_wsz)

        qi = indices.view(b, h, nc, wsz)
        ki = kv_indices.view(b, h, nc, wsz)

        qm = qi.unsqueeze(-1) == torch.arange(t).view(1, 1, 1, 1, -1).to(qi.device)
        qm = qm.sum(-2).transpose(-1, -2)
        qc = qm.argmax(-1).masked_fill((qm == 0).all(-1), -1)
        mqnull = qc == -1

        km = ki.unsqueeze(-1) == torch.arange(t).view(1, 1, 1, 1, -1).to(ki.device)
        km = km.sum(-2).transpose(-1, -2)
        kc = km.argmax(-1).masked_fill((km == 0).all(-1), -1)
        mknull = kc == -1

        matches = qc.unsqueeze(-1) == kc.unsqueeze(-2)
        mmnull = mqnull.unsqueeze(-1) & mknull.unsqueeze(-2)

        return matches.masked_fill(mmnull, False)


def load_kmeans_obj(path_to_kmeans_obj, device=None):
    # routing_transformer_(src)_(trg)_(num_clusters)_(window_size)_(num_heads)_(head_size)_(layer_id)_km.pt
    parts = path_to_kmeans_obj.split('_')
    num_clusters = int(parts[4])
    window_size = int(parts[5]) if parts[5] != 'None' else None
    num_heads = int(parts[6])
    head_size = int(parts[7])
    km = KmeansAttention(
        num_clusters,
        window_size,
        num_heads,
        head_size,
    )
    km_state_dict = torch.load(path_to_kmeans_obj, map_location=lambda storage, loc: storage)
    km.load_state_dict(km_state_dict)
    return km.to(device)


def save_kmeans_obj(kmeans_obj, layer_id, lp='en_de'):
    if not os.path.exists('centroids'):
        os.mkdir('centroids')
    fpath = 'centroids/routing_transformer_{}_{}_{}_{}_{}_{}_km.pt'.format(
        lp,
        kmeans_obj.num_clusters,
        kmeans_obj.window_size,
        kmeans_obj.num_heads,
        kmeans_obj.head_dim,
        layer_id,
    )
    torch.save(kmeans_obj.state_dict(), fpath)


def routing_simulated_attention(q, k, kmeans_attn_obj, topk_window_size=None):
    num_clusters = kmeans_attn_obj.num_clusters
    seq_len = q.shape[-2]
    window_size = seq_len // num_clusters + 1 if topk_window_size is None else topk_window_size
    m = kmeans_attn_obj(q, k, window_size=window_size, update_kmeans=False)
    return m.bool()


def learn_routing_centroids(
        dataset,
        proj_q,
        proj_k,
        layer,
        num_clusters,
        topk_window_size,
        concat_q_and_k=False
):
    km = KmeansAttention(
        num_clusters,
        topk_window_size,
        dataset.num_heads,
        proj_q.proj_dim,
    )
    km = km.cuda()
    for q, k, _ in dataset.get_train_batch(layer):
        if concat_q_and_k:
            q_low = k_low = proj_q(torch.cat([q, k], dim=-1))
        else:
            q_low, k_low = proj_q(q), proj_k(k)
        seq_len = q_low.shape[-2]
        window_size = seq_len // num_clusters + 1 if topk_window_size is None else topk_window_size
        km(q_low, k_low, window_size=window_size, update_kmeans=True)
        km.update_kmeans()
    return km


if __name__ == "__main__":
    from utils import get_length_mask, subsequent_mask

    L = 21
    q = torch.randn(4, 2, L, 25)
    k = torch.randn(4, 2, L, 25)
    v = torch.randn(4, 2, L, 25)
    lengths = torch.tensor([L - 2, L - 2, L - 1, L]).int()
    mask = get_length_mask(lengths)

    km = KmeansAttention(
        num_clusters=2,
        window_size=None,
        num_heads=2,
        head_dim=25,
    )
    x = km.forward(q, k, window_size=L//2 + 1, update_kmeans=True)
    print(x.int())
    print(x.shape)
    print(x.float().mean())

    save_kmeans_obj(km, 0)
    fpath = 'kmeans/routing_transformer_{}_{}_{}_{}_{}_km.pt'.format(2, None, 2, 25, 0)
    new_km = load_kmeans_obj(
        fpath,
        device=q.device
    )
    y = routing_simulated_attention(q, k, new_km)
    print(y.int())
    print(y.shape)
    print(y.float().mean())
    print(torch.allclose(x.float(), y.float()))
    os.remove(fpath)
