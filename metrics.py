import pandas as pd
import faiss
import numpy as np
from IPython import embed

def kl_divergence(target_p, predicted_q):
    # target_p is of shape N x C
    # predicted_q is of shape 1 x C
    # we would like the kl divergence of shape N x 1
    assert(target_p.shape[1]==predicted_q.shape[1])
    assert(predicted_q.shape[0]==1)

    # Checking absolute continuity
    idx= np.where(predicted_q==0)[0]
    if len(idx)>0:
        assert(np.all(target_p[:,idx]==0))

    # done in two steps to avoid divide by zero error
    # https://stackoverflow.com/questions/21610198/runtimewarning-divide-by-zero-encountered-in-log
    temp= np.where(target_p==0, 100, target_p)
    kl= np.where(temp<=1, -(temp*np.log(predicted_q/temp)), 0)

    return kl.sum(axis=1)