# from: https://github.com/huggingface/transformers/blob/master/src/transformers/models/reformer/modeling_reformer.py

import torch
from torch import nn


class LSHSelfAttention(nn.Module):
    def __init__(
            self,
            lsh_attn_chunk_length,
            num_hashes,
            num_buckets,
            hash_seed,
            max_position_embeddings,
            num_attention_heads,
            attention_head_size,
            hidden_size
    ):
        super().__init__()

        self.chunk_length = lsh_attn_chunk_length
        self.num_hashes = num_hashes
        self.num_buckets = num_buckets
        self.hash_seed = hash_seed
        self.max_position_embeddings = max_position_embeddings

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.hidden_size = hidden_size

        # save mask value here. Need fp32 and fp16 mask values
        self.register_buffer("self_mask_value_float16", torch.tensor(-1e3))
        self.register_buffer("self_mask_value_float32", torch.tensor(-1e5))
        self.register_buffer("mask_value_float16", torch.tensor(-1e4))
        self.register_buffer("mask_value_float32", torch.tensor(-1e9))

    def forward(self, hidden_states, attention_mask=None):
        sequence_length = hidden_states.shape[1]

        # num hashes can optionally be overwritten by user
        num_hashes = self.num_hashes

        # project hidden_states to query_key and value
        query_vectors = None
        query_key_vectors = hidden_states

        # if query key is not already split
        query_key_vectors = self._split_hidden_size_dim(
            query_key_vectors, self.num_attention_heads, self.attention_head_size
        )

        # set `num_buckets` on the fly, recommended way to do it
        if self.num_buckets is None:
            self._set_num_buckets(sequence_length)

        # hash query key vectors into buckets
        _, buckets = self._hash_vectors(query_key_vectors, num_hashes, attention_mask)
        buckets = buckets.transpose(-1, -2)
        m = buckets.unsqueeze(-2) == buckets.unsqueeze(-3)
        return m.any(-1)

    def _split_hidden_size_dim(self, x, num_attn_heads, attn_head_size):
        """
        splits hidden_size dim into attn_head_size and num_attn_heads
        """
        new_x_shape = x.size()[:-1] + (num_attn_heads, attn_head_size)
        x = x.view(*new_x_shape)
        return x.transpose(2, 1)

    def _set_num_buckets(self, sequence_length):
        # `num_buckets` should be set to 2 * sequence_length // chunk_length as recommended in paper
        num_buckets_pow_2 = (2 * (sequence_length // self.chunk_length)).bit_length() - 1
        # make sure buckets are power of 2
        num_buckets = 2 ** num_buckets_pow_2

        # factorize `num_buckets` if `num_buckets` becomes too large
        num_buckets_limit = 2 * max(
            int((self.max_position_embeddings // self.chunk_length) ** (0.5)),
            self.chunk_length,
        )
        if num_buckets > num_buckets_limit:
            num_buckets = [2 ** (num_buckets_pow_2 // 2), 2 ** (num_buckets_pow_2 - num_buckets_pow_2 // 2)]

        # set num buckets in config to be properly saved
        self.num_buckets = num_buckets

    def _hash_vectors(self, vectors, num_hashes, attention_mask, increase_num_buckets=False):
        batch_size = vectors.shape[0]

        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        if isinstance(self.num_buckets, int):
            assert (
                    self.num_buckets % 2 == 0
            ), f"There should be an even number of bucktes, but `self.num_bucktes`: {self.num_buckets}"
            rotation_size = self.num_buckets
            num_buckets = self.num_buckets
        else:
            # Factorize the hash if self.num_buckets is a list or tuple
            rotation_size, num_buckets = 0, 1
            for bucket_factor in self.num_buckets:
                assert (
                        bucket_factor % 2 == 0
                ), f"The number of buckets should be even, but `num_bucket`: {bucket_factor}"
                rotation_size = rotation_size + bucket_factor
                num_buckets = num_buckets * bucket_factor

        # remove gradient
        vectors = vectors.detach()

        if self.hash_seed is not None:
            # for determinism
            torch.manual_seed(self.hash_seed)

        rotations_shape = (self.num_attention_heads, vectors.shape[-1], num_hashes, rotation_size // 2)
        # create a random self.attention_head_size x num_hashes x num_buckets/2
        random_rotations = torch.randn(rotations_shape, device=vectors.device, dtype=vectors.dtype)
        # Output dim: Batch_Size x Num_Attn_Heads x Num_Hashes x Seq_Len x Num_Buckets/2
        rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)

        if isinstance(self.num_buckets, int) or len(self.num_buckets) == 1:
            rotated_vectors = torch.cat([rotated_vectors, -rotated_vectors], dim=-1)
            buckets = torch.argmax(rotated_vectors, dim=-1)
        else:
            # Get the buckets for them and combine.
            buckets, cur_sum, cur_product = None, 0, 1
            for bucket_factor in self.num_buckets:
                rotated_vectors_factor = rotated_vectors[..., cur_sum : cur_sum + (bucket_factor // 2)]
                cur_sum = cur_sum + bucket_factor // 2
                rotated_vectors_factor = torch.cat([rotated_vectors_factor, -rotated_vectors_factor], dim=-1)
                if buckets is None:
                    buckets = torch.argmax(rotated_vectors_factor, dim=-1)
                else:
                    buckets = buckets + (cur_product * torch.argmax(rotated_vectors_factor, dim=-1))

                cur_product = cur_product * bucket_factor

        if attention_mask is not None and (attention_mask.sum().item() < batch_size * attention_mask.shape[-1]):
            # add an extra bucket for padding tokens only
            num_buckets = num_buckets + 1
            # assign padding tokens extra bucket
            buckets_mask = attention_mask.to(torch.uint8)[:, None, None, :].expand(buckets.shape)
            buckets = torch.where(
                buckets_mask, buckets, torch.tensor(num_buckets - 1, dtype=torch.long, device=buckets.device)
            )
        elif increase_num_buckets:
            num_buckets = num_buckets + 1

        # buckets is now (Batch_size x Num_Attn_Heads x Num_Hashes x Seq_Len).
        # Next we add offsets so that bucket numbers from different hashing rounds don't overlap.
        offsets = torch.arange(num_hashes, device=vectors.device)
        offsets = (offsets * num_buckets).view((1, 1, -1, 1))

        # expand to batch size and num attention heads
        offsets = offsets.expand((batch_size, self.num_attention_heads) + offsets.shape[-2:])
        offset_buckets = (buckets + offsets).flatten(start_dim=2, end_dim=3)

        return offset_buckets, buckets


def reformer_simulated_attention(qk_vectors, lsh_attn_chunk_length=None, num_buckets=None, num_hashes=1, mask=None):
    bs, num_attention_heads, seq_len, head_size = qk_vectors.shape
    input_vectors = qk_vectors.transpose(1, 2).reshape(bs, seq_len, num_attention_heads*head_size)
    sim = LSHSelfAttention(
        lsh_attn_chunk_length=lsh_attn_chunk_length,
        num_hashes=num_hashes,
        num_buckets=num_buckets,
        hash_seed=torch.randint(0, 1000, size=()).item(),
        max_position_embeddings=3000,
        num_attention_heads=num_attention_heads,
        attention_head_size=head_size,
        hidden_size=num_attention_heads*head_size
    )
    matches = sim(input_vectors, attention_mask=mask).int()
    if mask is not None:
        matches = matches.masked_fill(~mask.unsqueeze(1).unsqueeze(2), 0)
    return matches


if __name__ == '__main__':
    from utils import get_length_mask, subsequent_maskw

    qk = torch.randn(4, 2, 5, 8)
    L = qk.shape[2]
    lengths = torch.tensor([L-2, L-2, L-1, L]).int()
    mask = get_length_mask(lengths)

    m = reformer_simulated_attention(qk, lsh_attn_chunk_length=2, num_buckets=None, num_hashes=1, mask=mask)
    print(m)
    print(m.shape)
    print(m.float().mean())
