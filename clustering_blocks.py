import os
import time

import torch
import numpy as np

from multi_kmeans import MultiKmeans
from routing_transformer import KmeansAttention
from utils import blockify


class KmeansManager:

    def __init__(self, args, base_dir='centroids'):
        super().__init__()
        self.args = args
        self.base_dir = base_dir
        self.fname = self.get_fname()
        self.current_layer = None
        self.centroids = []
        self.loaded = False

    def get_fname(self):
        base_folder = os.path.join(self.base_dir, 'skip-proj') if self.args.skip_projectors else self.base_dir
        fname = '{}/{}_{}r_{}s_{}n_{}bs_{}{}.pt'.format(
            base_folder,
            os.path.basename(self.args.data),
            self.args.rounds,  # projected vectors size
            self.args.num_clusters,  # how many clusters
            self.args.cluster_rounds,  # how many runs
            self.args.block_size,
            'shared' if self.args.share_projectors else 'indep',
            '_concat' if self.args.concat_q_and_k else ''
        )
        return fname

    def load_if_exists(self):
        if not os.path.exists(self.fname):
            print('Skipping load. {} does not exist'.format(self.fname))
        else:
            # return a list of centroids (one centroid per layer)
            print('Loading centroids from: ', self.fname)
            self.centroids = torch.load(self.fname, map_location=lambda storage, loc: storage)
            self.loaded = True

    def save(self):
        os.makedirs(os.path.dirname(self.fname), exist_ok=True)
        print('Saving centroids to: ', self.fname)
        torch.save(self.centroids, self.fname)

    @staticmethod
    def convert_sklearn_centroids_to_torch_tensor(clusters_per_head):
        # centroids_l.shape is (num_heads, num_runs, num_clusters, projection_size)
        list_clusters_per_head = [h.cluster_centers_ for h in clusters_per_head]
        centroids_l = np.stack(list_clusters_per_head)
        return torch.from_numpy(centroids_l)

    def _get_clustering_data(self, dataset, layer, proj_q, proj_k, split="train"):
        if split == "train":
            get_batch_fn = dataset.get_train_batch
        else:
            get_batch_fn = dataset.get_eval_batch
        data = [[] for _ in range(self.args.num_heads)]
        block_size = self.args.block_size
        with torch.no_grad():
            for q, k, length in get_batch_fn(layer):
                batch_size, num_heads, _, _ = q.shape
                # (bs, nh, seq_len, d) -> (bs, nh, seq_len, r)
                if self.args.concat_q_and_k:
                    q_low = k_low = proj_q(torch.cat([q, k], dim=-1))
                else:
                    q_low = proj_q(q)
                    k_low = proj_k(k)
                if block_size > 1:
                    q_low, k_low, length, _ = blockify(q_low, k_low, length, att_dist=None, block_size=block_size)
                # (bs, nh, seq_len, r) -> (nh, bs, seq_len, r)
                q_low = q_low.transpose(0, 1)
                k_low = k_low.transpose(0, 1)
                # (seq_len) -> (1, seq_len) -> (bs, seq_len)
                ar = torch.arange(q_low.shape[-2], device=q.device)
                ar = ar.unsqueeze(0).expand(batch_size, -1)
                ix = ar < length.unsqueeze(1)
                for h in range(num_heads):
                    # (nh, bs, seq_len, r) -> (nh, bs*var_seq_len, r)
                    q_low_vectors = q_low[h, ix]
                    data[h].append(q_low_vectors.cpu().detach())
                    k_low_vectors = k_low[h, ix]
                    data[h].append(k_low_vectors.cpu().detach())
            # if not concat_q_and_k: (nh, num_queries + num_keys, r)
            # if concat_q_and_k: (nh, num_queries, r)
            data = torch.stack([torch.cat(head) for head in data])
        data = data.cpu().numpy()
        return data

    def learn(self, dataset, layer, proj_q, proj_k):
        print('Training kmeans for layer {}'.format(layer))
        data = self._get_clustering_data(dataset, layer, proj_q, proj_k, split="train")
        heads, _, _ = data.shape
        clusters_per_head = []
        time_start = time.perf_counter()
        for h in range(heads):
            kmeans = MultiKmeans(n_clusters=self.args.num_clusters, rounds=self.args.cluster_rounds)
            kmeans.fit(data[h])
            clusters_per_head.append(kmeans)
        print('Done! Took {:.2f}s'.format(time.perf_counter() - time_start))
        centroids_l = KmeansManager.convert_sklearn_centroids_to_torch_tensor(clusters_per_head)
        self.centroids.append(centroids_l)

    def predict(self, x, layer=None, top_clusters=1):
        centroids = self.centroids[self.current_layer] if layer is None else self.centroids[layer]
        expanded_centroids = centroids.unsqueeze(0).expand(x.shape[0], -1, -1, -1, -1).to(x.device)
        expanded_x_low = x.unsqueeze(2)
        x_diffs_sq = (expanded_x_low.unsqueeze(-2).double() - expanded_centroids.unsqueeze(-3).double())**2
        x_dists = torch.sum(x_diffs_sq, dim=-1)
        _, x_clusters = torch.topk(x_dists, top_clusters, dim=-1, largest=False)
        x_clusters = x_clusters.squeeze(2)
        return x_clusters


class RoutingManager:
    def __init__(self, args, base_dir='centroids', topk_window_size=None):
        super().__init__()
        self.args = args
        self.base_dir = base_dir
        self.fname = self.get_fname()
        self.current_layer = None
        self.topk_window_size = topk_window_size
        self.centroids = []
        self.loaded = False

    def get_fname(self):
        lp = 'en_fr' if 'en-fr' in self.args.data else 'en_de'
        if 'roberta' in self.args.data:
            lp = 'entmax_roberta'
        fname_base = '{}/routing_transformer_{}_{}_{}_{}_{}_{}bs_{}_km.pt'.format(
            self.base_dir,
            lp,
            self.args.num_clusters,
            self.args.window_size,
            self.args.num_heads,
            self.args.rounds,
            self.args.block_size,
            'LAYERID'
        )
        return fname_base

    def load_if_exists(self):
        if not os.path.exists(self.fname.replace('LAYERID', str(self.args.num_layers-1))):
            print('Skipping load. {} does not exist'.format(self.fname))
        else:
            for i in range(self.args.num_layers):
                fname = self.fname.replace('LAYERID', str(i))
                print('Loading centroids from: ', fname)
                km = KmeansAttention(self.args.num_clusters, self.topk_window_size, self.args.num_heads, self.args.rounds)
                km_state_dict = torch.load(fname, map_location=lambda storage, loc: storage)
                km.load_state_dict(km_state_dict)
                km.eval()
                self.centroids.append(km)
            self.loaded = True

    def save(self):
        os.makedirs(os.path.dirname(self.fname), exist_ok=True)
        for i in range(self.args.num_layers):
            fname = self.fname.replace('LAYERID', str(i))
            print('Saving centroids to: ', fname)
            torch.save(self.centroids[i].state_dict(), fname)

    def learn(self, dataset, layer, proj_q, proj_k):
        print('Training routing for layer {}'.format(layer))
        km = KmeansAttention(self.args.num_clusters, self.topk_window_size, self.args.num_heads, self.args.rounds)
        km = km.cuda()
        km.train()
        for q, k, l in dataset.get_train_batch(layer):
            if self.args.concat_q_and_k:
                q_low = k_low = proj_q(torch.cat([q, k], dim=-1))
            else:
                q_low, k_low = proj_q(q), proj_k(k)
            if self.args.block_size > 1:
                q_low, k_low, l, _ = blockify(q_low, k_low, l, att_dist=None, block_size=self.args.block_size)
            seq_len = q_low.shape[-2]
            num_clusters = km.num_clusters
            window_size = seq_len // num_clusters + 1 if self.topk_window_size is None else self.topk_window_size
            km(q_low, k_low, window_size=window_size, update_kmeans=True)
            km.update_kmeans()
        return km

    def predict(self, q, k, layer=None, top_clusters=1):
        km = self.centroids[self.current_layer] if layer is None else self.centroids[layer]
        num_clusters = km.num_clusters
        seq_len = q.shape[-2]
        window_size = seq_len // num_clusters + 1 if self.topk_window_size is None else self.topk_window_size
        m = km(q, k, window_size=window_size, update_kmeans=False)
        return m.bool()


class RoutingExtendedManager:

    def __init__(self, args, base_dir='centroids', topk_window_size=None):
        super().__init__()
        self.args = args
        self.base_dir = base_dir
        self.fname = self.get_fname()
        self.current_layer = None
        self.topk_window_size = topk_window_size
        self.centroids = []
        self.loaded = False

    def get_fname(self):
        base_folder = os.path.join(self.base_dir, 'skip-proj') if self.args.skip_projectors else self.base_dir
        fname = '{}/{}_{}r_{}s_{}n_{}bs_{}{}.pt'.format(
            base_folder,
            os.path.basename(self.args.data),
            self.args.rounds,  # projected vectors size
            self.args.num_clusters,  # how many clusters
            self.args.cluster_rounds,  # how many runs
            self.args.block_size,
            'shared' if self.args.share_projectors else 'indep',
            '_concat' if self.args.concat_q_and_k else ''
        )
        return fname

    def load_if_exists(self):
        if not os.path.exists(self.fname):
            print('Skipping load. {} does not exist'.format(self.fname))
        else:
            # return a list of centroids (one centroid per layer)
            print('Loading centroids from: ', self.fname)
            centroids = torch.load(self.fname, map_location=lambda storage, loc: storage)
            for i in range(self.args.num_layers):
                c = centroids[i][:, 0]  # get just the first run (multi-round kmeans)
                km = KmeansAttention(self.args.num_clusters, self.topk_window_size, self.args.num_heads, self.args.rounds)
                km = km.cuda()
                km.eval()
                km.kmeans.load_means(c.cuda())
                self.centroids.append(km)
            self.loaded = True

    def predict(self, q, k, layer=None, top_clusters=1):
        km = self.centroids[self.current_layer] if layer is None else self.centroids[layer]
        num_clusters = km.num_clusters
        seq_len = q.shape[-2]
        window_size = seq_len // num_clusters + 1 if self.topk_window_size is None else self.topk_window_size
        m = km(q, k, window_size=window_size, update_kmeans=False)
        return m.bool()
