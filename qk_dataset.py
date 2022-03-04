import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from utils import blockify


def pad_list_of_tensors(list_of_tensors):
    # the contents of the list are tensors of different lengths
    # with shape (layer, head, length, dim)
    # this will return a batch of shape (len(list), layer, head, max_lengths_in_list, dim)
    x = [item.transpose(0, -2) for item in list_of_tensors]
    batch = pad_sequence(x, batch_first=True).transpose(1, -2)
    return batch


def load_texts(fname):
    texts = []
    with open(fname, 'r', encoding='utf8') as f:
        for line in f:
            texts.append(line.strip().split())
    return texts


class QKDataset(object):
    """
    Create a dataset with real query and key vectors.

    Each batch this class returns has Q and K from the heads in a given layer
    within an actual batch. Further batches will return the other layers and
    eventually a new actual batch.
    """

    def __init__(
        self,
        path,
        batch_size=16,
        to_cuda=True,
        rescale_queries=False,
        max_size=10000,
        shard_train=True,
        texts_path=None
    ):
        data = torch.load(path)
        q = data['q']
        k = data['k']
        lengths = data['length_src']
        texts = None
        if texts_path is not None:
            texts = load_texts(texts_path)

        # cut data if needed
        if len(q) > max_size:
            print('Cutting data size from {} to {}'.format(len(q), max_size))
            q = q[:max_size]
            k = k[:max_size]
            lengths = lengths[:max_size]

        if not shard_train:
            q = pad_list_of_tensors(q)
            k = pad_list_of_tensors(k)

        if 'roberta' in path:
            val_path = path.replace('kqs_enc-attn.pt', 'kqs_validation_enc-attn.pt')
            val_data = torch.load(val_path)
            val_q = val_data['q']
            val_k = val_data['k']
            val_lengths = val_data['length_src']
            self.num_train = len(q)
            self.num_eval = len(val_q)
            self.train_q = q
            self.train_k = k
            self.train_length = lengths
            self.eval_q = pad_list_of_tensors(val_q)
            self.eval_k = pad_list_of_tensors(val_k)
            self.eval_length = val_lengths
            self.train_texts = texts if texts is not None else None
            self.eval_texts = texts if texts is not None else None

        else:
            # leave 10% of the data for evaluating and the rest for training
            self.num_eval = int(0.1 * len(q))
            self.num_train = len(q) - self.num_eval
            # q and k are lists or tensors of shape (num_items, layer, head, length, dim)
            self.train_q = q[:self.num_train]
            self.train_k = k[:self.num_train]
            self.train_length = lengths[:self.num_train]
            self.eval_q = pad_list_of_tensors(q[self.num_train:])
            self.eval_k = pad_list_of_tensors(k[self.num_train:])
            self.eval_length = lengths[self.num_train:]
            self.train_texts = texts[:self.num_train] if texts is not None else None
            self.eval_texts = texts[self.num_train:] if texts is not None else None

        self.to_cuda = to_cuda
        self.rescale_queries = rescale_queries
        self.d = q[0].shape[-1]
        self.num_layers = q[0].shape[0]
        self.num_heads = q[0].shape[1]
        self.batch_size = batch_size
        self.shard_train = shard_train
        self.print_stats()

    def print_stats(self):
        print('rescale queries:', self.rescale_queries)
        print('head size:', self.d)
        print('num layers:', self.num_layers)
        print('num heads:', self.num_heads)
        print('batch size:', self.batch_size)
        print('train sents:', self.num_train)
        print('train tokens:', self.train_length.sum().item())
        print('avg sent len:', self.train_length.float().mean().item())
        print('std sent len:', self.train_length.float().std().item())
        print('train batches:', self.num_train // self.batch_size)
        print('eval sents:', self.num_eval)
        print('eval tokens:', self.eval_length.sum().item())
        print('eval avg sent len:', self.eval_length.float().mean().item())
        print('eval std sent len:', self.eval_length.float().std().item())
        print('eval batches:', self.num_eval // self.batch_size)

    def get_train_batch(self, layer: int):
        """
        Returns:
            A tuple of three tensors:
            q and k (batch, num_heads, max_length, dim)
            lengths (batch,)
        """
        for batch_start in range(0, self.num_train, self.batch_size):
            batch_end = batch_start + self.batch_size  # last batch may be smaller
            q = self.train_q[batch_start:batch_end]
            k = self.train_k[batch_start:batch_end]
            if self.shard_train:
                q = pad_list_of_tensors(q)
                k = pad_list_of_tensors(k)
            q = q[:, layer]
            k = k[:, layer]
            length = self.train_length[batch_start:batch_end]
            if self.to_cuda and torch.cuda.is_available():
                q = q.cuda()
                k = k.cuda()
                length = length.cuda()
            if self.rescale_queries:
                q = q * q.shape[-1] ** 0.5  # multiply by sqrt(d) to fix fairseq attention
            if self.train_texts is not None:
                yield q, k, length, self.train_texts[batch_start:batch_end]
            else:
                yield q, k, length

    def get_eval_batch(self, layer: int):
        """
        Returns:
            A tuple of three tensors:
            q and k (batch, num_heads, max_length, dim)
            lengths (batch,)
        """
        for batch_start in range(0, self.num_eval, self.batch_size):
            batch_end = batch_start + self.batch_size  # last batch may be smaller
            q = self.eval_q[batch_start:batch_end, layer]
            k = self.eval_k[batch_start:batch_end, layer]
            length = self.eval_length[batch_start:batch_end]
            if self.to_cuda and torch.cuda.is_available():
                q = q.cuda()
                k = k.cuda()
                length = length.cuda()
            if self.rescale_queries:
                q = q * q.shape[-1] ** 0.5  # multiply by sqrt(d) to fix fairseq attention
            if self.eval_texts is not None:
                yield q, k, length, self.eval_texts[batch_start:batch_end]
            else:
                yield q, k, length

    def get_clustering_data(self, layer, proj_q, proj_k, dataset="train", concat_q_and_k=False, block_size=1):
        if dataset == "train":
            get_batch_fn = self.get_train_batch
        elif dataset == "valid":
            get_batch_fn = self.get_eval_batch
        else:
            raise ValueError
        data = [[] for _ in range(self.num_heads)]
        with torch.no_grad():
            for q, k, length in get_batch_fn(layer):
                batch_size, num_heads, seq_len, _ = q.shape
                # (bs, nh, seq_len, d) -> (bs, nh, seq_len, r)
                if concat_q_and_k:
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
