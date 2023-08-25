import numpy as np
import pandas as pd
from utils.data import fetching_run, get_data, get_run_path, saving_run
from utils.argparser import build_parser
from IPython import embed
import sys

args = build_parser().parse_args(tuple(sys.argv[1:]))
args.budget= "high"
_, _, _, idx= get_data(args)

def get_toy_path(separable, std, algorithm, sd=1, tsh= 0.25, eps=0.95, cover_threshold=0.5):
    root= "/Users/victoriabarenne/thesis_experiments/toy_runs/"
    if algorithm=="adpc":
        path_to_run= root + f"toy_{separable}_{std}/100_4_{tsh}_False_gamma0.5_pessimistic_{sd}"
    elif algorithm=="pc":
        path_to_run= root + f"toy_{separable}_{std}/100_{tsh}_{sd}"
    elif algorithm in ["random", "full"]:
        path_to_run= root + f"toy_{separable}_{std}/100_{sd}"
    elif algorithm =="coverpc":
        path_to_run= root + f"toy_{separable}_{std}/100_4_0.95_False_eps{eps}_cover{cover_threshold}_{sd}"

    return path_to_run


n_seeds= 5
args.tsh= 0.95
for separable in ["linear", "not"]:
    args.separable= separable
    stds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] if separable=="linear" else [0.2, 0.3, 0.4, 0.5, 0.6]
    for std in stds:
        args.std= std
        for sd in range(1, n_seeds+1):
            args.sd= sd
            run_path = get_toy_path(args.separable, args.std, args.algorithm, sd=args.sd, tsh=args.tsh, eps=args.eps,
                                    cover_threshold=args.cover_threshold)
            scores, queries, radiuses, degrees, options, covers= fetching_run("adpc", run_path)
            print(run_path)
            scores= scores[:144]
            queries= queries[:5000]
            radiuses= radiuses[:,:144]
            degrees= degrees[:5000]
            options= options[:5000]
            covers= covers[:144]
            saving_run("adpc", run_path,
                       scores, queries, radiuses, degrees, options, covers)

