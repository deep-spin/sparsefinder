# cp centroids/kqs_enc-attn.pt_4r_20s_1n_shared.pickle /mnt/data-zeus2/mtreviso/longformer_stuff/entmax-roberta-maia/centroids/
# cp centroids/kqs_enc-attn.pt_4r_16s_1n_shared.pickle /mnt/data-zeus2/mtreviso/longformer_stuff/entmax-roberta-maia/centroids/
# python3 convert_centroids_routing_all.py
# cp centroids/routing_transformer_entmax_roberta_16_None_12_4_all_km.pt /mnt/data-zeus2/mtreviso/longformer_stuff/entmax-roberta-maia/centroids/
# cp centroids/routing_transformer_entmax_roberta_20_None_12_4_all_km.pt /mnt/data-zeus2/mtreviso/longformer_stuff/entmax-roberta-maia/centroids/

import torch

rounds = 4
num_heads = 12
num_layers = 12
# for nc in [2, 4, 6, 8, 10, 12]:
for nc in [16, 20]:
    centroids = []
    for li in range(num_layers):
        fname = "centroids/routing_transformer_entmax_roberta_{}_None_{}_{}_{}_km.pt".format(nc, num_heads, rounds, li)
        centroids_li = torch.load(fname, map_location=lambda storage, loc: storage)
        centroids.append(centroids_li)
    new_fname = "centroids/routing_transformer_entmax_roberta_{}_None_{}_{}_all_km.pt".format(nc, num_heads, rounds)
    print('Saving torch centroids to: {}'.format(new_fname))
    torch.save(centroids, new_fname)



