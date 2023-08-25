import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from activelearners import ProbCoverSampler_Faiss, RandomSampler
from clustering import MyKMeans
import sys
import os
import pandas as pd
import time
import warnings
from utils.reduction_methods import get_cover
from utils.data import get_data, get_run_path, saving_run, fetching_run
from utils.argparser import build_parser
from models import ClassifierNN
warnings.filterwarnings(action='ignore', category=FutureWarning)

def simulate_random(run_path, dataset, dataset_test, args, eval_points):
    dataset.restart()
    scores, aucs= [], []
    learner = RandomSampler(dataset)
    model = ClassifierNN(1, dataset, args.dataset, running_cluster=args.running_cluster)
    x_test, y_test = dataset_test.get_all_data()

    while len(dataset.queries)<eval_points.max():
        learner.query(1)
        # Train a 1-NN classifier and get the test accuracy
        if len(dataset.queries) in eval_points:
            x_train, y_train = dataset.get_labeled_data()
            model.fit_all(x_train, y_train)
            perf= model.score_accuracy(x_test, y_test)
            if args.running_cluster and args.dataset=="chexpert":
                auc= model.score_auc(x_test, y_test)
                aucs.append(auc)
            scores.append(perf)
            print(len(dataset.queries), perf)
    saving_run("random", run_path, scores, dataset.queries, aucs=aucs)

def simulate_PC(run_path, algorithm: "str", dataset, dataset_test, args, eval_points, plot=False):
    assert(algorithm in ["pc" ,"adpc", "partialadpc", "coverpc"])
    dataset.restart()

    scores, degrees, options, covers, aucs= [], [], [], [], []
    radiuses = np.empty(shape=(dataset.n_points, 0))
    n_classes = len(np.unique(dataset.y))
    clustering = MyKMeans(dataset, n_classes)
    current_threshold= 0.1 # for partialadpc
    x_test, y_test = dataset_test.get_all_data()
    model= ClassifierNN(1, dataset, args.dataset, running_cluster=args.running_cluster)

    # Initialize the radius as in original Probcover
    learner = ProbCoverSampler_Faiss(dataset,
                                     args.tsh,
                                     clustering,
                                     args.algorithm,
                                     radius= args.radius,
                                     hard_thresholding=args.hard_thresholding)

    while len(dataset.queries)< eval_points.max():
        # Query new points
        if algorithm == "pc":
            deg, opt = learner.query(1, args)
        elif algorithm=="adpc":
            deg, opt = learner.adpc_query(1, args)
        elif algorithm== "coverpc":
            deg, opt = learner.coverpc_query(1, args)
        elif algorithm == "partialadpc":
            deg, opt, current_threshold = learner.partialadpc_query(1, args, current_threshold)

        degrees.append(deg)
        options.append(opt)

        if plot and len(dataset.queries)%5==0 and len(dataset.queries)<=100:
            dataset.plot_al(True) 
  
        if len(dataset.queries) in eval_points:
            cover= get_cover(algorithm, dataset, learner.lims_ref, learner.D_ref, learner.I_ref)
            covers.append(cover)
            x_train, y_train = dataset.get_labeled_data()
            model.fit_all(x_train, y_train)
            perf= model.score_accuracy(x_test, y_test)
            if args.running_cluster and args.dataset=="chexpert":
                auc= model.score_auc(x_test, y_test)
                aucs.append(auc)
            print(len(dataset.queries), dataset.queries[-1], cover*100, perf)
            scores.append(perf)
            radiuses= np.concatenate((radiuses, dataset.radiuses.reshape(-1,1)), axis=1)

        if len(dataset.queries)%args.savefreq==0:
            saving_run(algorithm, run_path, scores, dataset.queries, radiuses, degrees, options, covers, aucs=aucs)

    saving_run(algorithm, run_path, scores, dataset.queries, radiuses, degrees, options, covers, aucs=aucs)


def simulate_fullsupervised(run_path, length, dataset, dataset_test):
    x_train, y_train = dataset.get_all_data()
    x_test, y_test = dataset_test.get_all_data()

    model = ClassifierNN(1, dataset, args.dataset, running_cluster=args.running_cluster)
    model.fit_all(x_train, y_train)
    perf = model.score_accuracy(x_test, y_test)
    if args.running_cluster and args.dataset=="chexpert":
        auc= model.score_auc(x_test, y_test)
        pd.DataFrame({'full_auc': np.repeat(auc, length)}).to_csv(f'{run_path}/full_auc.csv')
    pd.DataFrame({'full_scores': np.repeat(perf, length)}).to_csv(f'{run_path}/full.csv')

if __name__ == "__main__":
    args = build_parser().parse_args(tuple(sys.argv[1:]))

    if args.sd is not None:
        np.random.seed(args.sd)

    # get the training and testing data
    dataset, dataset_test, run_path, eval_points = get_data(args)

    if not os.path.exists(run_path):
        os.makedirs(run_path)

    ####
    from IPython import embed
    embed()

    import faiss
    d = 2048  # dimension
    nlist=1000
    k=2
    rad= 0.4
    #old way
    # index_bis = faiss.IndexFlatL2(d)
    # index_bis.add(dataset.x.astype('float32'))
    # D_bis, I_bis = index_bis.search(dataset_test.x.astype("float32"), k)
    # lims_bis, D_bis, I_bis = index_bis.range_search(dataset_test.x.astype('float32'), 0.3 ** 2)  # because faiss uses squared L2 error
    t1=time.time()
    quantizer = faiss.IndexFlatL2(d)  # the other index
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    # here we specify METRIC_L2, by default it performs inner-product search
    assert not index.is_trained
    index.train(dataset.x.astype('float32'))
    assert index.is_trained
    t2= time.time()
    print(f"Index trained in {t2-t1}")

    index.add(dataset.x.astype('float32')[:3,:])  # add may be a bit slower as well
    t3= time.time()
    print(f"Data added in {t3-t2}")

    index.nprobe=1
    # D, I = index.search(dataset_test.x.astype('float32'), k)  # actual search
    lims, D, I = index.range_search(dataset_test.x.astype('float32'), rad**2)
    print(I[-5:])  # neighbors of the 5 last queries
    t4 = time.time()
    print(f"Data searched with 1 probe in {t4 - t3}")

    index.nprobe = 10000 # The tradeoff between speed and accuracy is set via the nprobe parameter. Higher-> more accurate
    DD, II = index.search(dataset_test.x.astype('float32'), k)  # actual search
    lllims, DDD, III = index.range_search(dataset_test.x.astype('float32'), rad**2)

    print(np.sum(II!=I))
    t5 = time.time()
    print(f"Data searched with 100 probes in {t5 - t4}")

    from IPython import embed
    embed()

    ####




    # simulate the runs
    if args.algorithm=="benchmark":
        print("Running the benchmarks")
        start_time=time.time()
        simulate_random(run_path, dataset, dataset_test, args, eval_points)
        random_time= time.time()
        print(f"Random ran in {(random_time-start_time)/60} minutes")
        simulate_fullsupervised(run_path, len(eval_points), dataset, dataset_test)
    elif args.algorithm in ["pc", "adpc", "partialadpc", "coverpc"]:
        print(f"Running {args.algorithm} only")
        start_time= time.time()
        simulate_PC(run_path, args.algorithm, dataset, dataset_test, args, eval_points)
        pc_time= time.time()
        print(f"{args.algorithm} ran in {(pc_time-start_time)/60} minutes")
    print("done")

