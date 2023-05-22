from torch.utils.data import DataLoader
import numpy as np
import torchvision
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, CenterCrop
import torch
from torch.utils.data import Dataset
from IPython import embed
import os
import argparse
import sys
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models import resnet18
from models import ContrastiveModel


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
    mean= [0.4914, 0.4822, 0.4465] if args.dataset=="cifar10" else [0.5071, 0.4867, 0.4408]
    std= [0.2023, 0.1994, 0.2010] if args.dataset=="cifar100" else [0.2675, 0.2565, 0.2761]
    crop_size= 32
    transform = Compose([CenterCrop(crop_size),
                         ToTensor(),
                         Normalize(mean=mean, std=std)])


    if args.dataset=="cifar10":
        dataset_train = torchvision.datasets.CIFAR10(root="./.", transform= transform, train= True , download=True)
        dataset_test = torchvision.datasets.CIFAR10(root="./.", transform= transform, train= False , download=True)
    elif args.dataset=="cifar100":
        dataset_train = torchvision.datasets.CIFAR100(root=".", transform= transform, train= True , download=True)
        dataset_test = torchvision.datasets.CIFAR100(root=".", transform= transform, train= False , download=True)


    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    run_path= f"./data/new_transform/normalized/{args.dataset}/{args.n_epochs}epochs"

    if not os.path.exists(run_path):
        os.makedirs(run_path)
    np.save(f"{run_path}/train_targets", np.array(dataset_train.targets).reshape(-1,1))
    np.save(f"{run_path}/test_targets", np.array(dataset_test.targets).reshape(-1,1))

    #Build the model
    backbone = resnet18()
    model_kwargs =  {'head': 'mlp', 'features_dim': 128}
    model = ContrastiveModel(backbone, **model_kwargs)
    model.load_state_dict(torch.load(f"/Users/victoriabarenne/Documents/DoctorLoop/models/scan-{args.dataset}/resnet18_simclr_{args.n_epochs}epochs.pth.tar", map_location="cpu"))
    print("Weights have loaded")
    model.eval()

    if not os.path.exists(f"{run_path}/test_features.npy"):
        print(f"{run_path}/test_features.npy does not exist")
        # Extracting the features for the testing data
        extracted_features_test = np.empty(shape=(0, 512))
        for id, (img, target) in enumerate(test_loader):
            features = model.backbone(img.to(torch.float32))
            extracted_features_test = np.append(extracted_features_test, features.detach(), axis=0)
            print(id, extracted_features_test.shape)
        np.save(f"{run_path}/test_features", extracted_features_test)

    if not os.path.exists(f"{run_path}/train_features.npy"):
        print(f"{run_path}/train_features.npy does not exist")
        extracted_features_train = np.empty(shape=(0, 512))
        # Extracting the features for the training data
        for id, (img, target) in enumerate(train_loader):
            features = model.backbone(img.to(torch.float32))
            extracted_features_train = np.append(extracted_features_train, features.detach(), axis=0)
            print(id, extracted_features_train.shape)
        np.save(f"{run_path}/train_features", extracted_features_train)

    # normalizing the features
    if not os.path.exists(f"{run_path}/train_features_normalized.npy") or not os.path.exists(f"{run_path}/test_features_normalized.npy"):
        train_features= np.load(f"{run_path}/train_features.npy")
        test_features= np.load(f"{run_path}/test_features.npy")

        train_features= F.normalize(torch.from_numpy(train_features), dim=1)
        test_features= F.normalize(torch.from_numpy(test_features), dim=1)

        np.save(f"{run_path}/train_features_normalized", train_features)
        np.save(f"{run_path}/test_features_normalized", test_features)

