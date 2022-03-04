import itertools
from collections import OrderedDict
from copy import deepcopy

from qk_dataset import QKDataset
from quantization import main, get_arg_parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    print('Loading data...')
    dataset = QKDataset(args.data, args.batch, rescale_queries=args.rescale_queries)

    # define runs (order of keys in each dict determines the execution order)
    runs = [
        # OrderedDict({
        #     # baseline
        #     'grouping': ['dist'],
        #     'dist': ['cos'],
        #     'window': [0, 1, 3, 5, 7, 9, 11, 15, 19, 23, 27],
        #     'dist_t': [0.0],
        # }),
        # OrderedDict({
        #     # distance-based
        #     'grouping': ['dist'],
        #     'dist': ['cos'],
        #     'window': [None, 1, 3, 5, 7, 9, 11, 15, 19, 23, 27],
        #     'dist_t': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        # }),
        # OrderedDict({
        #     # distance-based
        #     'grouping': ['dist'],
        #     'dist': ['l2'],
        #     'window': [None, 1, 3, 5, 7, 9, 11, 15, 19, 23, 27],
        #     'dist_t': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        # }),
        # OrderedDict({
        #     # entmax
        #     'grouping': ['dist'],
        #     'dist': ['dotproduct_entmax'],
        #     'window': [None, 1, 3, 5, 7, 9, 11, 15, 19, 23, 27],
        #     'dist_t': [1.2, 1.4, 1.6, 1.8, 2.0],
        # }),
        # OrderedDict({
        #     # fixed and dynamic buckets
        #     'grouping': ['quantize'],
        #     'same_size': [True, False],  # fixed vs dynamic
        #     'window': [None, 1, 3, 5, 7, 9, 11, 15, 19, 23, 27],
        #     'bucket_size': [4, 8, 12, 16, 20],
        # }),
        # OrderedDict({
        #     # kmeans with topk=1 and topk=2
        #     'grouping': ['kmeans'],
        #     'top_clusters': [1, 2],
        #     'window': [None, 1, 3, 5, 7, 9, 11, 15, 19, 23, 27],
        #     'bucket_size': [2, 4, 8, 12, 16, 20],
        #     # 'bucket_size': [2, 4, 6, 8, 10, 12],  # faster and similar plots
        # }),
        # OrderedDict({
        #     'grouping': ['dist'],
        #     'dist': ['bigbird'],
        #     'window': [None, 1, 3, 5, 7, 9, 11, 15, 19, 23, 27],
        #     'dist_t': [1, 2, 3, 4, 5],
        # }),
        # OrderedDict({
        #     'grouping': ['dist'],
        #     'dist': ['longformer'],
        #     'window': [None, 1, 3, 5, 7, 9, 11, 15, 19, 23, 27],
        #     'dist_t': [4, 8, 12, 16, 20],
        # }),
        # OrderedDict({
        #     'grouping': ['dist'],
        #     'dist': ['openai_sparse'],
        #     'window': [None, 1, 3, 5, 7, 9, 11, 15, 19, 23, 27],
        #     'dist_t': [4, 8, 12, 16, 20],
        # }),
        # OrderedDict({
        #     'grouping': ['dist'],
        #     'dist': ['reformer'],
        #     'window': [None, 1, 3, 5, 7, 9, 11, 15, 19, 23, 27],
        #     'dist_t': [2, 4, 6, 8, 10, 14, 18, 22, 26, 30],
        # }),
        # OrderedDict({
        #     'grouping': ['dist'],
        #     'dist': ['reformer_rounds'],
        #     'window': [None, 1, 3, 5, 7, 9, 11, 15, 19, 23, 27],
        #     'dist_t': [2, 4, 6, 8, 10, 14, 18, 22, 26, 30],
        # }),
        # OrderedDict({
        #     'grouping': ['dist'],
        #     'dist': ['routing'],
        #     'window': [None, 1, 3, 5, 7, 9, 11, 15, 19, 23, 27],
        #     'bucket_size': [2, 4, 6, 8, 10, 12],  # num_clusters
        #     'top_clusters': [0],  # balanced clusters with size of seq_len // num_clusters + 1
        # }),
        # OrderedDict({
        #     'grouping': ['dist'],
        #     'dist': ['routing_trained'],
        #     'window': [None, 1, 3, 5, 7, 9, 11, 15, 19, 23, 27],
        #     'bucket_size': [2, 4, 6, 8, 10, 12],  # num_clusters
        #     'top_clusters': [0],  # balanced clusters with size of seq_len // num_clusters + 1
        # }),
        # OrderedDict({
        #     'grouping': ['dist'],
        #     'dist': ['routing_extender'],
        #     'window': [None, 1, 3, 5, 7, 9, 11, 15, 19, 23, 27],
        #     'bucket_size': [2, 4, 6, 8, 10, 12],  # num_clusters
        #     'top_clusters': [0],  # balanced clusters with size of seq_len // num_clusters + 1
        # }),
        OrderedDict({
            'grouping': ['dist'],
            'dist': ['routing_trained'],
            'window': [None],
            'bucket_size': [16, 20],  # num_clusters
            'top_clusters': [0],  # balanced clusters with size of seq_len // num_clusters + 1
        }),
    ]

    for od_run in runs:
        arg_names = list(od_run.keys())
        arg_values = list(od_run.values())
        for args_comb_values in itertools.product(*arg_values):
            # copy args to not overwrite default args
            new_args = deepcopy(args)
            # set new args
            for arg_name, arg_value in zip(arg_names, args_comb_values):
                setattr(new_args, arg_name, arg_value)
            # make sure dist_t is a list for having backwards compatibility with argparse result
            if isinstance(new_args.dist_t, float) or isinstance(new_args.dist_t, int):
                new_args.dist_t = [float(new_args.dist_t)]
            # make sure dist is not set for non-dist methods
            if 'dist' not in arg_names:
                new_args.dist = None
            # call main from quantization.py
            main(new_args, dataset)
