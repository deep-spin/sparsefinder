import torch

from multi_kmeans import MultiKmeans


def learn_clusters_for_head(n_clusters, rounds, data_head):
    kmeans = MultiKmeans(n_clusters=n_clusters, rounds=rounds).fit(data_head)
    return kmeans.fit(data_head)


def learn_clusters(dataset, layer, proj_q, proj_k, n_clusters=16, rounds=1, concat_q_and_k=False):
    data = dataset.get_clustering_data(layer, proj_q, proj_k, dataset="train", concat_q_and_k=concat_q_and_k)
    heads, elems, dim = data.shape
    kmeans_heads = []
    for h in range(heads):
        kmeans = learn_clusters_for_head(n_clusters, rounds, data[h])
        kmeans_heads.append(kmeans)
    return kmeans_heads


def predict_clusters_for_head(cluster, x_head, top_clusters=1):
    batch_size, seq_len, dim = x_head.shape
    device = x_head.device
    x_head = x_head.reshape(batch_size*seq_len, dim).cpu().detach().numpy()
    preds = cluster.predict(x_head, top_clusters=top_clusters)
    # shape: (rounds, batch*seq_len) if top_clusters==1 else (rounds*k, batch*seq_len)
    preds = torch.from_numpy(preds).to(device)
    adjust_rounds = cluster.kmeans[0].n_clusters if top_clusters > 1 else 1
    preds = preds.transpose(1, 0).reshape(batch_size, seq_len, cluster.rounds * adjust_rounds)
    return preds


def predict_clusters(x, clusters_per_head, top_clusters=1):
    if isinstance(clusters_per_head, torch.Tensor):
        bsz = x.shape[0]
        expanded_centroids = clusters_per_head.unsqueeze(0).expand(bsz, -1, -1, -1, -1).to(x.device)
        expanded_x_low = x.unsqueeze(2)
        x_diffs_sq = (expanded_x_low.unsqueeze(-2).double() - expanded_centroids.unsqueeze(-3).double())**2
        x_dists = torch.sum(x_diffs_sq, dim=-1)
        _, x_clusters = torch.topk(x_dists, top_clusters, dim=-1, largest=False)
        x_clusters = x_clusters.squeeze(2)
    else:
        x_clusters = []
        for head in range(len(clusters_per_head)):
            cluster = clusters_per_head[head]
            x_head = x[:, head, :, :]
            preds = predict_clusters_for_head(cluster, x_head, top_clusters=top_clusters)
            x_clusters.append(preds)
        x_clusters = torch.stack(x_clusters).transpose(0, 1)
    return x_clusters
