# Copyright 2020 The BigBird Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BigBird Attention Layers."""

import torch
import numpy as np


def get_single_block_row_attention(block_id,
                                   to_start_block_id,
                                   to_end_block_id,
                                   num_rand_blocks,
                                   window_block_left=1,
                                   window_block_right=1,
                                   global_block_left=1,
                                   global_block_right=1):
    """For a single row block get random row attention.

    Args:
      block_id: int. block id of row.
      to_start_block_id: int. random attention coloum start id.
      to_end_block_id: int. random attention coloum end id.
      num_rand_blocks: int. number of random blocks to be selected.
      window_block_left: int. number of blocks of window to left of a block.
      window_block_right: int. number of blocks of window to right of a block.
      global_block_left: int. Number of blocks globally used to the left.
      global_block_right: int. Number of blocks globally used to the right.

    Returns:
      row containing the random attention vector of size num_rand_blocks.
    """

    # list of to_blocks from which to choose random attention
    to_block_list = np.arange(to_start_block_id, to_end_block_id,
                              dtype=np.int32)
    # permute the blocks
    perm_block = np.random.permutation(to_block_list)
    # print(perm_block)

    # illegal blocks for the current block id, using window
    illegal_blocks = list(
        range(block_id - window_block_left, block_id + window_block_right + 1))

    # Add blocks at the start and at the end
    illegal_blocks.extend(list(range(global_block_left)))
    illegal_blocks.extend(
        list(range(to_end_block_id - global_block_right, to_end_block_id)))

    # The second from_block cannot choose random attention on second last to_block
    if block_id == 1:
        illegal_blocks.append(to_end_block_id-2)

    # The second last from_block cannot choose random attention on second to_block
    if block_id == to_end_block_id - 2:
        illegal_blocks.append(1)

    selected_random_blokcs = []

    for i in range(to_end_block_id - to_start_block_id):
        if perm_block[i] not in illegal_blocks:
            selected_random_blokcs.append(perm_block[i])
        if len(selected_random_blokcs) == num_rand_blocks:
            break
    return np.array(selected_random_blokcs, dtype=np.int32)


def bigbird_block_rand_mask_with_head(from_seq_length,
                                      to_seq_length,
                                      from_block_size,
                                      to_block_size,
                                      num_heads,
                                      plan_from_length,
                                      plan_num_rand_blocks,
                                      window_block_left=1,
                                      window_block_right=1,
                                      global_block_top=1,
                                      global_block_bottom=1,
                                      global_block_left=1,
                                      global_block_right=1):
    """Create adjacency list of random attention.

    Args:
      from_seq_length: int. length of from sequence.
      to_seq_length: int. length of to sequence.
      from_block_size: int. size of block in from sequence.
      to_block_size: int. size of block in to sequence.
      num_heads: int. total number of heads.
      plan_from_length: list. plan from lenght where num_rand are choosen from.
      plan_num_rand_blocks: list. number of rand blocks within the plan.
      window_block_left: int. number of blocks of window to left of a block.
      window_block_right: int. number of blocks of window to right of a block.
      global_block_top: int. number of blocks at the top.
      global_block_bottom: int. number of blocks at the bottom.
      global_block_left: int. Number of blocks globally used to the left.
      global_block_right: int. Number of blocks globally used to the right.

    Returns:
      adjacency list of size num_head where each element is of size
      from_seq_length//from_block_size-2 by num_rand_blocks
    """
    assert from_seq_length//from_block_size == to_seq_length//to_block_size, \
        "Error the number of blocks needs to be same!"

    assert from_seq_length in plan_from_length, \
        "Error from sequence length not in plan!"

    # Total number of blocks in the mmask
    num_blocks = from_seq_length//from_block_size
    # Number of blocks per plan
    plan_block_length = np.array(plan_from_length) // from_block_size
    # till when to follow plan
    max_plan_idx = plan_from_length.index(from_seq_length)
    # Random Attention adjajency list
    rand_attn = [np.zeros((num_blocks,
                           np.sum(plan_num_rand_blocks[:max_plan_idx+1])),
                          dtype=np.int32) for i in range(num_heads)]

    # We will go iteratively over the plan blocks and pick random number of
    # Attention blocks from the legally allowed blocks
    for plan_idx in range(max_plan_idx+1):
        rnd_r_cnt = 0
        if plan_idx > 0:
            # set the row for all from_blocks starting from 0 to
            # plan_block_length[plan_idx-1]
            # column indx start fromm plan_block_length[plan_idx-1] and ends at
            # plan_block_length[plan_idx]
            if plan_num_rand_blocks[plan_idx] > 0:
                rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
                curr_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx+1]))
                for blk_rw_idx in range(global_block_top,
                                        plan_block_length[plan_idx-1]):
                    for h in range(num_heads):
                        # print("head", h, "blk_rw_idx", blk_rw_idx)
                        rand_attn[h][blk_rw_idx,
                        rnd_r_cnt:curr_r_cnt] = get_single_block_row_attention(
                            block_id=blk_rw_idx,
                            to_start_block_id=plan_block_length[plan_idx - 1],
                            to_end_block_id=plan_block_length[plan_idx],
                            num_rand_blocks=plan_num_rand_blocks[plan_idx],
                            window_block_left=window_block_left,
                            window_block_right=window_block_right,
                            global_block_left=global_block_left,
                            global_block_right=global_block_right)

            for pl_id in range(plan_idx):
                if plan_num_rand_blocks[pl_id] == 0:
                    continue
                for blk_rw_idx in range(plan_block_length[plan_idx-1],
                                        plan_block_length[plan_idx]):
                    rnd_r_cnt = 0
                    to_start_block_id = 0
                    if pl_id > 0:
                        rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id]))
                        to_start_block_id = plan_block_length[pl_id-1]
                    curr_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id+1]))
                    for h in range(num_heads):
                        # print("head", h, "blk_rw_idx", blk_rw_idx)
                        rand_attn[h][blk_rw_idx,
                        rnd_r_cnt:curr_r_cnt] = get_single_block_row_attention(
                            block_id=blk_rw_idx,
                            to_start_block_id=to_start_block_id,
                            to_end_block_id=plan_block_length[pl_id],
                            num_rand_blocks=plan_num_rand_blocks[pl_id],
                            window_block_left=window_block_left,
                            window_block_right=window_block_right,
                            global_block_left=global_block_left,
                            global_block_right=global_block_right)

        if plan_num_rand_blocks[plan_idx] == 0:
            continue
        # print("Start from here")
        curr_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx+1]))
        from_start_block_id = global_block_top
        to_start_block_id = 0
        if plan_idx > 0:
            rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
            from_start_block_id = plan_block_length[plan_idx-1]
            to_start_block_id = plan_block_length[plan_idx-1]

        for blk_rw_idx in range(from_start_block_id, plan_block_length[plan_idx]):
            for h in range(num_heads):
                # print("head", h, "blk_rw_idx", blk_rw_idx)
                rand_attn[h][blk_rw_idx,
                rnd_r_cnt:curr_r_cnt] = get_single_block_row_attention(
                    block_id=blk_rw_idx,
                    to_start_block_id=to_start_block_id,
                    to_end_block_id=plan_block_length[plan_idx],
                    num_rand_blocks=plan_num_rand_blocks[plan_idx],
                    window_block_left=window_block_left,
                    window_block_right=window_block_right,
                    global_block_left=global_block_left,
                    global_block_right=global_block_right)

    for nh in range(num_heads):
        rand_attn[nh] = rand_attn[nh][global_block_top:num_blocks -
                                                       global_block_bottom, :]
    return rand_attn


def get_rand_attn_plan(from_seq_length, from_block_size, num_rand_blocks):
    """Gives the plan of where to put random attention.

    Args:
      from_seq_length: int. length of from sequence.
      from_block_size: int. size of block in from sequence.
      num_rand_blocks: int. Number of random chunks per row.

    Returns:
      plan_from_length: ending location of from block
      plan_num_rand_blocks: number of random ending location for each block
    """
    # general plan
    plan_from_length = []
    plan_num_rand_blocks = []
    if (2*num_rand_blocks + 5) < (from_seq_length // from_block_size):
        plan_from_length.append(int((2*num_rand_blocks + 5)*from_block_size))
        plan_num_rand_blocks.append(num_rand_blocks)
        plan_from_length.append(from_seq_length)
        plan_num_rand_blocks.append(0)
    elif (num_rand_blocks + 5) < (from_seq_length // from_block_size):
        plan_from_length.append(int((num_rand_blocks + 5)*from_block_size))
        plan_num_rand_blocks.append(num_rand_blocks//2)
        plan_from_length.append(from_seq_length)
        plan_num_rand_blocks.append(num_rand_blocks - (num_rand_blocks//2))
    else:
        plan_from_length.append(from_seq_length)
        plan_num_rand_blocks.append(num_rand_blocks)

    return plan_from_length, plan_num_rand_blocks


def bigbird_block_rand_mask(from_seq_length,
                            to_seq_length,
                            from_block_size,
                            to_block_size,
                            num_rand_blocks,
                            last_idx=-1):
    """Create adjacency list of random attention.

    Args:
      from_seq_length: int. length of from sequence.
      to_seq_length: int. length of to sequence.
      from_block_size: int. size of block in from sequence.
      to_block_size: int. size of block in to sequence.
      num_rand_blocks: int. Number of random chunks per row.
      last_idx: if -1 then num_rand_blocks blocks chosen anywhere in to sequence,
        if positive then num_rand_blocks blocks choosen only upto last_idx.

    Returns:
      adjacency list of size from_seq_length//from_block_size-2 by num_rand_blocks
    """
    assert from_seq_length//from_block_size == to_seq_length//to_block_size, \
        "Error the number of blocks needs to be same!"

    rand_attn = np.zeros(
        (from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32)
    middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
    last = to_seq_length // to_block_size - 1
    if last_idx > (2 * to_block_size):
        last = (last_idx // to_block_size) - 1

    r = num_rand_blocks  # shorthand
    for i in range(1, from_seq_length // from_block_size-1):
        start = i-2
        end = i
        if i == 1:
            rand_attn[i-1, :] = np.random.permutation(middle_seq[2:last])[:r]
        elif i == 2:
            rand_attn[i-1, :] = np.random.permutation(middle_seq[3:last])[:r]
        elif i == from_seq_length // from_block_size - 3:
            rand_attn[i-1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -3: should have been sliced till last-3
        elif i == from_seq_length // from_block_size - 2:
            rand_attn[i-1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -4: should have been sliced till last-4
        else:
            if start > last:
                start = last
                rand_attn[i-1, :] = np.random.permutation(middle_seq[:start])[:r]
            elif (end+1) == last:
                rand_attn[i-1, :] = np.random.permutation(middle_seq[:start])[:r]
            else:
                rand_attn[i-1, :] = np.random.permutation(
                    np.concatenate((middle_seq[:start], middle_seq[end+1:last])))[:r]
    return rand_attn


def full_bigbird_mask(from_seq_length,
                      to_seq_length,
                      from_block_size,
                      to_block_size,
                      num_rand_blocks,
                      rand_attn=None,
                      focus=1024,
                      max_seq_len=4096):
    """Calculate BigBird attention pattern as a full dense matrix.

    Args:
      from_seq_length: int. length of from sequence.
      to_seq_length: int. length of to sequence.
      from_block_size: int. size of block in from sequence.
      to_block_size: int. size of block in to sequence.
      num_rand_blocks: int. Number of random chunks per row.
      rand_attn: adjajency matrix for random attention.
      focus: pick random mask within focus

    Returns:
      attention mask matrix of shape [from_seq_length, to_seq_length]
    """
    if rand_attn is None:
        rand_attn = bigbird_block_rand_mask(max_seq_len, max_seq_len,
                                            from_block_size, to_block_size,
                                            num_rand_blocks, focus)

    attn_mask = np.zeros((max_seq_len, max_seq_len), dtype=np.int32)
    for i in range(1, (max_seq_len // from_block_size) - 1):
        attn_mask[(i) * from_block_size:(i + 1) * from_block_size,
        (i - 1) * to_block_size:(i + 2) * to_block_size] = 1
        for j in rand_attn[i - 1, :]:
            attn_mask[i * from_block_size:(i + 1) * from_block_size,
            j * to_block_size:(j + 1) * to_block_size] = 1

    attn_mask[:from_block_size, :] = 1
    attn_mask[:, :to_block_size] = 1
    attn_mask[:, -to_block_size:] = 1
    attn_mask[-from_block_size:, :] = 1
    clipped_attn_mask = attn_mask[:from_seq_length, :to_seq_length]
    return np.array(clipped_attn_mask, dtype=bool)


def create_rand_mask_from_inputs(from_blocked_mask,
                                 to_blocked_mask,
                                 rand_attn,
                                 num_attention_heads,
                                 num_rand_blocks,
                                 batch_size,
                                 from_seq_length,
                                 from_block_size):
    """Create 3D attention mask from a 2D tensor mask.

    Args:
      from_blocked_mask: 2D Tensor of shape [batch_size,
        from_seq_length//from_block_size, from_block_size].
      to_blocked_mask: int32 Tensor of shape [batch_size,
        to_seq_length//to_block_size, to_block_size].
      rand_attn: [batch_size, num_attention_heads,
        from_seq_length//from_block_size-2, num_rand_blocks]
      num_attention_heads: int. Number of attention heads.
      num_rand_blocks: int. Number of random chunks per row.
      batch_size: int. Batch size for computation.
      from_seq_length: int. length of from sequence.
      from_block_size: int. size of block in from sequence.

    Returns:
      float Tensor of shape [batch_size, num_attention_heads,
                             from_seq_length//from_block_size-2,
                             from_block_size, num_rand_blocks*to_block_size].
    """
    num_windows = from_seq_length // from_block_size - 2
    rand_mask = tf.reshape(
        tf.gather(to_blocked_mask, rand_attn, batch_dims=1), [
            batch_size, num_attention_heads, num_windows,
            num_rand_blocks * from_block_size
        ])
    rand_mask = tf.einsum("BLQ,BHLK->BHLQK", from_blocked_mask[:, 1:-1],
                          rand_mask)
    return rand_mask


def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
    """Create 3D attention mask from a 2D tensor mask.

    Args:
      from_blocked_mask: 2D Tensor of shape [batch_size,
        from_seq_length//from_block_size, from_block_size].
      to_blocked_mask: int32 Tensor of shape [batch_size,
        to_seq_length//to_block_size, to_block_size].

    Returns:
      float Tensor of shape [batch_size, 1, from_seq_length//from_block_size-4,
                             from_block_size,  3*to_block_size].
    """
    exp_blocked_to_pad = tf.concat(
        [to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2],
         to_blocked_mask[:, 3:-1]], 2)
    band_mask = tf.einsum("BLQ,BLK->BLQK",
                          tf.cast(from_blocked_mask[:, 2:-2], tf.float32),
                          tf.cast(exp_blocked_to_pad, tf.float32))
    band_mask = tf.expand_dims(band_mask, 1)
    return band_mask


def create_attention_mask_from_input_mask(from_mask, to_mask):
    """Create attention mask from a 2D tensor mask.

    Args:
      from_mask: int32 Tensor of shape [batch_size, from_seq_length].
      to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
      int32 Tensor of shape [batch_size, 1, from_seq_length, to_seq_length].
    """
    mask = tf.einsum("BF,BT->BFT", from_mask, to_mask)

    # expand to create a slot for heads.
    mask = tf.expand_dims(mask, 1)

    return mask


def bigbird_simulated_attention(attention_mask,
                                num_attention_heads,
                                num_rand_blocks,
                                from_seq_length,
                                to_seq_length,
                                from_block_size,
                                to_block_size,
                                max_seq_len=4096,
                                seed=None):
    """BigBird attention calculation using masks in quadratic time.

    Args:
      query_layer: float Tensor of shape [batch_size, num_attention_heads,
        from_seq_length, size_per_head]
      key_layer: float Tensor of shape [batch_size, num_attention_heads,
        to_seq_length, size_per_head]
      value_layer: float Tensor of shape [batch_size, num_attention_heads,
        to_seq_length, size_per_head]
      attention_mask: int32 Tensor of shape [batch_size,
        from_seq_length, to_seq_length]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions in
        the mask that are 0, and will be unchanged for positions that are 1.
      num_attention_heads: int. Number of attention heads.
      num_rand_blocks: int. Number of random chunks per row.
      size_per_head: int. Size of each attention head.
      from_seq_length: int. length of from sequence.
      to_seq_length: int. length of to sequence.
      from_block_size: int. size of block in from sequence.
      to_block_size: int. size of block in to sequence.
      seed: (Optional) int. Reandom seed for generating random mask.

    Returns:
      float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
        size_per_head].
    """

    if seed:
        np.random.seed(seed)

    plan_from_length, plan_num_rand_blocks = get_rand_attn_plan(
        from_seq_length, from_block_size, num_rand_blocks)

    rand_attn = bigbird_block_rand_mask_with_head(
        from_seq_length=from_seq_length,
        to_seq_length=to_seq_length,
        from_block_size=from_block_size,
        to_block_size=to_block_size,
        num_heads=num_attention_heads,
        plan_from_length=plan_from_length,
        plan_num_rand_blocks=plan_num_rand_blocks)
    temp_mask = [
        full_bigbird_mask(
            from_seq_length,
            to_seq_length,
            from_block_size,
            to_block_size,
            num_rand_blocks,
            rand_attn=rand_attn[i],
            focus=1024,
            max_seq_len=max_seq_len)
        for i in range(num_attention_heads)
    ]
    temp_mask = np.stack(temp_mask, axis=0)
    temp_mask = np.array(temp_mask, dtype=bool)

    rand_block_mask = torch.from_numpy(temp_mask).int().unsqueeze(0)
    if attention_mask is not None:
        attention_mask = torch.min(torch.from_numpy(attention_mask).int(), rand_block_mask)
    else:
        attention_mask = rand_block_mask

    return attention_mask


if __name__ == '__main__':

    def get_length_mask(lengths, max_length=None):
        if max_length is None:
            max_length = lengths.max()
        r = torch.arange(max_length).unsqueeze(0).to(lengths.device)
        mask = r < lengths.unsqueeze(-1)
        return mask

    lengths = torch.tensor([5, 6, 8, 8])
    mask = get_length_mask(lengths)
    mask = mask.unsqueeze(-1) & mask.unsqueeze(1)
    mask = mask.int().unsqueeze(1).numpy()
    x = bigbird_simulated_attention(
        mask,
        num_attention_heads=8,
        num_rand_blocks=3,
        # from_seq_length: int. length of from sequence.
        from_seq_length=mask.shape[-2],
        # to_seq_length: int. length of to sequence.
        to_seq_length=mask.shape[-1],
        # from_block_size: int. size of block in from sequence.
        from_block_size=1,
        # to_block_size: int. size of block in to sequence.
        to_block_size=1,
        max_seq_len=lengths.max().item()
    )
    print(x)

    for L in range(1,200):
        import math
        mask = np.ones((1, 16, L, L))
        num_rand_blocks = L // 8 if L <= 18 else 6
        # num_rand_blocks = min(6, L//8)
        print(L, num_rand_blocks)
        x = bigbird_simulated_attention(
            mask,
            num_attention_heads=16,
            num_rand_blocks=num_rand_blocks,
            from_seq_length=mask.shape[-2],
            to_seq_length=mask.shape[-1],
            from_block_size=1,
            to_block_size=1,
            max_seq_len=mask.shape[-1]
        )
    print(x)