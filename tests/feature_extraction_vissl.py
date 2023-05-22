from torch.utils.data import DataLoader
from vissl.models import build_model
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from classy_vision.generic.util import load_checkpoint
from vissl.utils.checkpoint import init_model_from_consolidated_weights
import numpy as np
import torchvision
from torchvision.transforms import Resize, Compose, ToTensor
import torch
from torch.utils.data import Dataset
from IPython import embed
import os
import argparse
import sys
import torch.nn as nn

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



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

    parser.add_argument('--dataset', type= str, required=True,
                        help='"cifar10" or "cifar100"')

    parser.add_argument('--trunk', type= str2bool, default= True,
                        help= 'Whether to extract the features from the "trunk" or the "head"')

    parser.add_argument('--n_epochs', type=int, required=True,
                        help='number of pre-training epochs for the simclr model') # 100, 200, 400, 800, 1000

    parser.add_argument('--batch_size', type=int, default= 128,
                        help='batch size for the dataloaders')
    return parser

class custom_subset(Dataset):
    def __init__(self, dataset, indices):
      self.data= dataset.data[indices.astype(int)]
      self.targets = np.array(dataset.targets)[indices.astype(int)]
    def __getitem__(self, idx):
      return (self.data[idx], self.targets[idx])
    def __len__(self):
        return len(self.targets)



if __name__ == "__main__":
    args = build_parser().parse_args(tuple(sys.argv[1:]))

    transform= Compose([ToTensor(), Resize(224)])
    if args.dataset=="cifar10":
        cifar_traindata = torchvision.datasets.CIFAR10(root="./.", transform= transform, train= True , download=True)
        cifar_testdata = torchvision.datasets.CIFAR10(root="./.", transform= transform, train= False , download=True)
    elif args.dataset=="cifar100":
        cifar_traindata = torchvision.datasets.CIFAR100(root=".", transform= transform, train= True , download=True)
        cifar_testdata = torchvision.datasets.CIFAR100(root=".", transform= transform, train= False , download=True)

    dataset_train= custom_subset(cifar_traindata, np.arange(len(cifar_traindata)))
    dataset_test= custom_subset(cifar_testdata, np.arange(len(cifar_testdata)))

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)


    run_path= f"./data_test/{args.dataset}/{args.n_epochs}epochs"
    if args.trunk==False:
        run_path = f"./data_test/{args.dataset}/{args.n_epochs}epochs/head"

    if not os.path.exists(run_path):
        os.makedirs(run_path)

    np.save(f"{run_path}/train_targets", dataset_train.targets.reshape(-1,1))
    np.save(f"{run_path}/test_targets", dataset_test.targets.reshape(-1,1))


    cfg = [
        'config=pretrain/simclr/simclr_8node_resnet.yaml',
        f'config.MODEL.WEIGHTS_INIT.PARAMS_FILE=./models/resnet50_simclr_{args.n_epochs}ep.torch',
        # Specify path for the model weights.
        'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True',  # Turn on model evaluation mode.
        'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=True',  # Freeze trunk.
        # 'config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY=True', # Extract the trunk features, as opposed to the HEAD. -> features of size 2048
        f'config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY={args.trunk}',
        # Extract the trunk features, as opposed to the HEAD. -> features of size 128
        'config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=True',  # Do not flatten features.
        # 'config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP=[["res5avg", ["Identity", []]]]'
        # Extract only the res5avg features.
    ]

    # Compose the hydra configuration.
    cfg = compose_hydra_configuration(cfg)
    _, cfg = convert_to_attrdict(cfg)

    # Build the model
    model = build_model(cfg.MODEL, cfg.OPTIMIZER)
    print(cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE)
    # Load the checkpoint weights.
    weights = load_checkpoint(checkpoint_path=cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE)

    # Initialize the model with the simclr model weights.
    init_model_from_consolidated_weights(
        config=cfg,
        model=model,
        state_dict=weights,
        state_dict_key_name="classy_state_dict",
        skip_layers=[],  # Use this if you do not want to load all layers
    )

    print("Weights have loaded")
    model.eval()
    embed()
    if args.trunk:
        extracted_features_train = np.empty(shape=(0, 2048))
        extracted_features_test = np.empty(shape=(0, 2048))

    else:
        extracted_features_train= np.empty(shape=(0, 128))
        extracted_features_test= np.empty(shape=(0, 128))


    # Extracting the features for the testing data
    for id, (img, target) in enumerate(test_loader):
        img = img.permute((0, 3, 1, 2))
        features = model(img.to(torch.float32))
        extracted_features_test = np.append(extracted_features_test, features[0].detach(), axis=0)
        print(id, extracted_features_test.shape)

    # Extracting the features for the training data
    for id, (img, target) in enumerate(train_loader):
        img = img.permute((0, 3, 1, 2))
        features = model(img.to(torch.float32))
        extracted_features_train = np.append(extracted_features_train, features[0].detach(), axis=0)
        print(id, extracted_features_train.shape)

    np.save(f"{run_path}/train_features", extracted_features_train)
    np.save(f"{run_path}/test_features", extracted_features_test)





