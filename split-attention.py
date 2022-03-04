# -*- coding: utf-8 -*-

"""
Script to split Q and K vectors by length

This expects data as saved by extract-att.py.
"""

import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', help='.pt file with Q, K and lengths')
    parser.add_argument('output', help='Base name and path for the output')
    args = parser.parse_args()

    length_bins = [(10, 20), (20, 30), (30, 40), (40, int(1e6))]
    # length_bins = [(10, 30), (10, 40), (10, int(1e6)), (20, 40), (20, int(1e6)), (30, int(1e6))]
    binned_q = {bin_: [] for bin_ in length_bins}
    binned_k = {bin_: [] for bin_ in length_bins}
    binned_lengths_src = {bin_: [] for bin_ in length_bins}
    binned_lengths_trg = {bin_: [] for bin_ in length_bins}

    # split the data into dictionaries mapping bins to a list of tensors that
    # fall within that length
    data = torch.load(args.data)

    # each of them is a list of batches
    all_q = data['q']
    all_k = data['k']
    all_lenghts_src = data['length_src']
    if '-enc.pt' in args.data:
        all_lenghts_trg = all_lenghts_src
    else:
        all_lenghts_trg = data['length_trg']
    del data  # this is big in memory

    for q, k, lengths_src, lengths_trg in zip(
            all_q, all_k, all_lenghts_src, all_lenghts_trg):
        # q and k are (batch, layer, head, length, dim)

        for low, high in length_bins:
            inds = torch.logical_and(low <= lengths_src, lengths_src < high)
            bin_q = q[inds]
            bin_k = k[inds]
            bin_lengths_src = lengths_src[inds]
            bin_lengths_trg = lengths_trg[inds]

            for i, (length_src, length_trg) in enumerate(
                    zip(bin_lengths_src, bin_lengths_trg)):
                binned_lengths_src[(low, high)].append(length_src)
                binned_lengths_trg[(low, high)].append(length_trg)
                slice_q = bin_q[i, :, :, :length_trg]
                slice_k = bin_k[i, :, :, :length_src]

                binned_q[(low, high)].append(slice_q)
                binned_k[(low, high)].append(slice_k)

    del all_q, all_k, all_lenghts_src, all_lenghts_trg
    for bin_ in binned_q.keys():
        q = binned_q[bin_]
        k = binned_k[bin_]
        lengths_src = torch.stack(binned_lengths_src[bin_])
        lengths_trg = torch.stack(binned_lengths_trg[bin_])
        data = {'q': q, 'k': k, 'length_src': lengths_src,
                'length_trg': lengths_trg}
        name = args.output + '-{}-{}.pt'.format(*bin_)
        torch.save(data, name)
