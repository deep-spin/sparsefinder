import torch

from utils import neighbours_mask


def get_global_mask(bs, seq_len, min_globals, max_globals, device=None):
    mask = torch.zeros(bs, seq_len, seq_len, dtype=torch.int, device=device)
    # inds = torch.randint(seq_len, size=(bs, max_globals))
    # mask = mask.scatter(1, inds, 1)
    for i in range(bs):
        n = torch.randint(min_globals, max_globals, size=()).item() if min_globals < max_globals else max_globals
        inds = torch.randint(seq_len, size=(n,))
        mask[i, inds, :] = 1
        mask[i, :, inds] = 1
    return mask.bool()


def longformer_simulated_attention(
        attention_mask,
        window_size,
        dilation=0,
        min_globals_per_sample=2,
        max_globals_per_sample=4
):
    bs, seq_len = attention_mask.shape
    device = attention_mask.device
    window_mask = neighbours_mask(seq_len, window_size).to(device).bool()
    if dilation > 0:
        # todo: implement dilated window mask
        pass
    global_mask = torch.zeros_like(attention_mask).unsqueeze(0)
    if max_globals_per_sample > 0:
        assert min_globals_per_sample < max_globals_per_sample
        global_mask = get_global_mask(bs, seq_len, min_globals_per_sample, max_globals_per_sample, device)
    new_mask = attention_mask.unsqueeze(1) & (window_mask.unsqueeze(0) | global_mask)
    return new_mask


if __name__ == '__main__':
    from utils import get_length_mask, subsequent_mask

    L = 10
    lengths = torch.tensor([L-2, L-2, L-1, L]).int()
    mask = get_length_mask(lengths)

    x = longformer_simulated_attention(
        mask,
        window_size=5,
        dilation=0,
        min_globals_per_sample=2,
        max_globals_per_sample=4
    )
    print(x)
    print(x.shape)