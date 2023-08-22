import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def build_parser():
    parser = argparse.ArgumentParser()
    ### parameters to set for all runs ###
    parser.add_argument('--dataset', type=str, required=True,
                        help='"cifar10" or "cifar100" or "toy"')
    parser.add_argument('--run', type=str, required=False, default="runs",
                        help='folder name to save the run')
    parser.add_argument('--sd', type=int, default=None, required=True,
                        help="run seed")
    parser.add_argument('--n_epochs', type=int, required=True,
                        help="number of training epochs for extracted features")
    parser.add_argument('--tsh', type=float, default=0.95, required=True,
                        help="purity threshold")
    parser.add_argument('--hard_thresholding', type=str2bool, default=False)
    parser.add_argument('--radius', type=float, required=False,
                        help="initial radius, if not set will be initialized using kmeans and the purity threshold")
    parser.add_argument('--budget', type=str, required=False, default="normal",
                        help="'low', 'high', 'full'")
    parser.add_argument('--running_cluster', type= str2bool, default=False)
    ### parameters to set for cifar experiments ###
    parser.add_argument('--algorithm', type=str, required=True,
                        help="Wether to run 'benchmark', 'adpc', 'pc' or 'coverpc'")
    parser.add_argument('--warm_start', type=str2bool, default=False)

    ### extra parameters to set for the varistions of the ADPC method ###
    parser.add_argument('--gamma', type=float, required=False, default=0.5,
                        help="from 0 (pc: not reducing the balls) to 0.5 (reducing the balls minimally so that their intersection is empty: usual adpc")
    parser.add_argument('--reduction_method', type=str, required=False, default="pessimistic",
                        help="how to reduce the balls: can be 'pessimistic' or 'mix' ")

    ### extra parameters to set for the coverpc method ###
    parser.add_argument('--cover_threshold', type=float, required=False, default=0.5,
                        help="cover threshold")
    parser.add_argument('--eps', type=float, required=False, default=0.95,
                        help="reduction factor for radius")

    ### extra parameters for the PartialADPC method ###

    ### parameters for toy dataset runs ###
    parser.add_argument('--separable', type=str, required=False, default="not",
                        help='Whether to make problem separable')  # others are "linear" and "nonlinear"
    parser.add_argument('--std', type=float, required=False,
                        help="std of clusters")
    parser.add_argument('--n_points', type=int, default=1000,
                        help="number of points per class")

    ### default parameters for all runs ####
    parser.add_argument('--gauss', type=int, required=False, default=4,
                        help='norm for the gaussian weighting')
    parser.add_argument('--savefreq', type=float, default=100,
                        help="how often to save the progress")
    parser.add_argument('--K', type=int, default=5,
                        help="number of neighbours for radius update using Knn average")

    return parser

