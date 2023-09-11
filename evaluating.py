import pandas as pd
import numpy as np
from utils.data import get_data, fetching_run
from models import ClassifierNN
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from utils.argparser import build_parser
import sys
from datasets import CHEXPERT_remedis
# args= pd.Series({
#     "dataset":"chexpert",
#     "algorithm":"pc",
#     "sd":1,
#     "gamma":0.5,
#     "tsh":0.95,
#     "hard_thresholding":"False",
#     "separable":"not",
#     "n_epochs":1000,
#     "running_cluster":"True",
#     "run":"chexpert_runs",
#     "budget":"low",
#     })

def get_auc_knn(K, x_train, y_train, x_test, y_test, weights="uniform"):
    if len(x_train)<K:
        auc=None
    else:
        model= KNeighborsClassifier(n_neighbors=K, weights=weights)
        #weights "uniform" or "distance"
        model.fit(x_train, y_train)
        y_pred= model.predict(x_test)
        auc= tf.keras.metrics.AUC(multi_label=True)(y_test, y_pred).numpy()
    return auc

def get_auc_continuous_knn(K, x_train, y_train, x_test, y_test, weights="uniform"):
    if len(x_train)<K:
        auc=None
    else:
        model= KNeighborsClassifier(n_neighbors=K, weights=weights)
        #weights "uniform" or "distance"
        model.fit(x_train, y_train)
        model.classes_= [np.array([0., 1.]),
                         np.array([0., 1.]),
                         np.array([0., 1.]),
                         np.array([0., 1.]),
                         np.array([0., 1.])]
        y_pred = model.predict_proba(x_test)
        y_pred = np.array(y_pred)[:,:,1]
        y_pred = np.moveaxis(y_pred, 0, 1)
        auc= tf.keras.metrics.AUC(multi_label=True)(y_test, y_pred).numpy()
    return auc
    

def get_auc_mlp(x_train, y_train, x_valid, y_valid, x_test, y_test, lr_init=0.001, n_epochs=100, patience=10):
    SHUFFLE_BUFFER_SIZE=128
    BATCH_SIZE=64
    
    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    ds_valid = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))

    ds_train = ds_train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    ds_test = ds_test.batch(BATCH_SIZE)
    ds_valid = ds_valid.batch(BATCH_SIZE)

    model = tf.keras.Sequential([
        # tf.keras.layers.Dense(1024, activation='relu'),
        # tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5),
        tf.keras.layers.Activation('sigmoid'),
    ])

    model.build([None, 2048])
    
    # https://stackoverflow.com/questions/62350538/tf2-2-loading-a-saved-model-from-tensorflow-hub-failed-with-attributeerror
    optimizer= tf.keras.optimizers.Adam(learning_rate= lr_init)    
    model.compile(optimizer= optimizer, 
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics = [tf.keras.metrics.AUC(from_logits=False, multi_label= True)])

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                patience=patience, 
                                                baseline=None, 
                                                start_from_epoch= 10)
    
    # Fitting the model
    history_train = model.fit(ds_train,
                              verbose=0, 
                              epochs=n_epochs,
                              validation_data= ds_valid,
                              validation_freq= 1,
                              callbacks=[callback], 
                             )
    history_train= pd.DataFrame.from_dict(history_train.history)

    #Evaluating the model
    history_test = model.evaluate(ds_test, verbose=0)

    return history_train, history_test
    
    
if __name__ == "__main__":
    args = build_parser().parse_args(tuple(sys.argv[1:]))

    if args.sd is not None:
        np.random.seed(args.sd)
    
    dataset, dataset_test, run_path, idx = get_data(args)
    dataset_valid = CHEXPERT_remedis(type="valid", cluster=args.running_cluster)
    scores_true, queries, radiuses, degrees, options, covers= fetching_run(args.algorithm, run_path)

    x_test, y_test = dataset_test.get_all_data()
    x_valid, y_valid = dataset_test.get_all_data()
    aucs= pd.DataFrame(columns= ["5_NN", "20_NN", "100_NN", "5_NN_continuous", "20_NN_continuous", "100_NN_continuous", "mlp"])

    if args.algorithm=="full":
        x_train, y_train = dataset.get_all_data()
        auc5=  get_auc_knn(5, x_train, y_train, x_test, y_test, weights="distance")
        auc20=  get_auc_knn(20, x_train, y_train, x_test, y_test, weights="distance")
        auc100=  get_auc_knn(100, x_train, y_train, x_test, y_test, weights="distance")
        
        auc5_cont=  get_auc_continuous_knn(5, x_train, y_train, x_test, y_test, weights="distance")
        auc20_cont=  get_auc_continuous_knn(20, x_train, y_train, x_test, y_test, weights="distance")
        auc100_cont=  get_auc_continuous_knn(100, x_train, y_train, x_test, y_test, weights="distance")
        print(auc5, auc20, auc100, auc5_cont, auc20_cont, auc100_cont)
        _, history_test= get_auc_mlp(x_train, y_train, x_valid, y_valid, x_test, y_test, lr_init=0.001, n_epochs=500)
        new_row = {"5_NN":auc5, "20_NN":auc20, "100_NN":auc100, 
                   "5_NN_continuous":auc5_cont, "20_NN_continuous":auc20_cont, "100_NN_continuous":auc100_cont, "mlp": history_test[-1]}
        aucs = pd.concat([aucs, pd.DataFrame([new_row])], ignore_index=True)
        aucs.to_csv(run_path+"/evaluation_full.csv")
    else:
        for i in range(len(idx)):  
            dataset.restart()
            dataset.observe(queries[:idx[i]])
            x_train, y_train = dataset.get_labeled_data()
            auc5=  get_auc_knn(5, x_train, y_train, x_test, y_test, weights="distance")
            auc20=  get_auc_knn(20, x_train, y_train, x_test, y_test, weights="distance")
            auc100=  get_auc_knn(100, x_train, y_train, x_test, y_test, weights="distance")
            
            auc5_cont=  get_auc_continuous_knn(5, x_train, y_train, x_test, y_test, weights="distance")
            auc20_cont=  get_auc_continuous_knn(20, x_train, y_train, x_test, y_test, weights="distance")
            auc100_cont=  get_auc_continuous_knn(100, x_train, y_train, x_test, y_test, weights="distance")
        
            _, history_test= get_auc_mlp(x_train, y_train, x_valid, y_valid, x_test, y_test, lr_init=0.001, n_epochs=500)
            new_row = {"5_NN":auc5, "20_NN":auc20, "100_NN":auc100, 
                       "5_NN_continuous":auc5_cont, "20_NN_continuous":auc20_cont, "100_NN_continuous":auc100_cont, "mlp": history_test[-1]}
            aucs = pd.concat([aucs, pd.DataFrame([new_row])], ignore_index=True)
            if i>200:
                aucs.to_csv(run_path+"/evaluation.csv")
    
