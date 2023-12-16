import argparse
import numpy as np
import pickle
import sys
from pathlib import Path
from torchvision import datasets
from sklearn.model_selection import StratifiedKFold
import scipy.io as sio

from arguments import parse_option
from utils import reproductibility


def download_dataset(dataset_dir='./data/', dataset='cifar10'):
    if dataset == 'cifar10':
        datasets.CIFAR10(root=dataset_dir, download=True)
    elif dataset == 'svhn':
        datasets.SVHN(root=dataset_dir, split='train', download=True)
        datasets.SVHN(root=dataset_dir, split='test', download=True)
    else:
        raise ValueError(dataset)

def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()
    return data

def load_cifar10(data_dir='./data/'):
    X_train = None
    y_train = []

    for i in range(1, 6):
        data_dic = unpickle(
            data_dir/"cifar-10-batches-py/data_batch_{}".format(i))
        if i == 1:
            X_train = data_dic['data']
        else:
            X_train = np.vstack((X_train, data_dic['data']))
        y_train += data_dic['labels']

    test_data_dic = unpickle(data_dir/"cifar-10-batches-py/test_batch")
    X_test = test_data_dic['data']
    X_test = X_test.reshape(len(X_test), 3, 32, 32)
    y_test = np.array(test_data_dic['labels'])
    X_train = X_train.reshape((len(X_train), 3, 32, 32))
    y_train = np.array(y_train)

    train_img = X_train.transpose((0, 2, 3, 1))
    train_label = y_train
    test_img = X_test.transpose((0, 2, 3, 1))
    test_label = y_test

    return train_img, train_label, test_img, test_label


def load_svhn(data_dir='./dataset/'):
    train_data = sio.loadmat(data_dir/'train_32x32.mat')
    x_train = train_data['X']
    x_train = x_train.transpose((3, 0, 1, 2))
    y_train = train_data['y'].reshape(-1)
    y_train[y_train == 10] = 0

    test_data = sio.loadmat(data_dir/'test_32x32.mat')
    x_test = test_data['X']
    x_test = x_test.transpose((3, 0, 1, 2))
    y_test = test_data['y'].reshape(-1)
    y_test[y_test == 10] = 0

    return x_train, y_train, x_test, y_test

def get_label_proportion(num_bags=100, num_classes=10):
    proportion = np.random.rand(num_bags, num_classes)
    proportion /= proportion.sum(axis=1, keepdims=True)

    return proportion

def get_N_label_proportion(proportion, num_instances, num_classes):
    N = np.zeros(proportion.shape)
    for i in range(len(proportion)):
        p = proportion[i]
        for c in range(len(p)):
            if (c+1) != num_classes:
                num_c = int(np.round(num_instances*p[c]))
                if sum(N[i])+num_c >= num_instances:
                    num_c = int(num_instances-sum(N[i]))
            else:
                num_c = int(num_instances-sum(N[i]))

            N[i][c] = int(num_c)
        np.random.shuffle(N[i])
    return N

def get_index(label, proportion_N, replace=True):
    # make index
    idx = np.arange(len(label))
    idx_c = []
    for c in range(label.max()+1):
        x = idx[label[idx] == c]
        np.random.shuffle(x)
        idx_c.append(x)

    bags_idx = []
    for n in range(len(proportion_N)):
        bag_idx = []
        for c in range(label.max()+1):
            sample_c_index = idx_c[c][: int(proportion_N[n][c])]
            bag_idx.extend(sample_c_index)
            if replace:
                np.random.shuffle(idx_c[c])
            else:
                idx_c[c] = idx_c[c][int(proportion_N[n][c]):]

        np.random.shuffle(bag_idx)
        bags_idx.append(bag_idx)
        
        assert len(bag_idx) == sum(proportion_N[0]); 'Not enough data.'

    return np.array(bags_idx)

def make_bag(dataset_dir='./data/', dataset='cifar10', num_classes=10, num_instances=1024, num_bags=100, replace=True):
    if dataset == 'cifar10':
        data, label, test_data, test_label = load_cifar10(dataset_dir)
    elif dataset == 'svhn':
        data, label, test_data, test_label = load_svhn(dataset_dir)
    else:
        raise ValueError(dataset)
    
    # make poroportion
    proportion = get_label_proportion(num_bags, num_classes)
    proportion_N = get_N_label_proportion(proportion, num_instances, num_classes)
    bags_idx = get_index(label, proportion_N, replace)

    bags, labels = data[bags_idx], label[bags_idx]
    proportions = proportion_N / num_instances
    
    np.save(f'{dataset_dir}/{dataset}_bags_{num_instances}_{num_bags}', bags)
    np.save(f'{dataset_dir}/{dataset}_label_{num_instances}_{num_bags}', labels)
    np.save(f'{dataset_dir}/{dataset}_proportions_{num_instances}_{num_bags}', proportions)

    np.save(f'{dataset_dir}/{dataset}_test_data', test_data)
    np.save(f'{dataset_dir}/{dataset}_test_label', test_label)
    
if __name__ == '__main__':
    args = parse_option()

    reproductibility(seed=args.seed)

    print('downloading dataset ...')
    download_dataset(dataset_dir=args.dataset_dir, dataset=args.dataset)

    print('making bag ...')
    make_bag(dataset_dir=args.dataset_dir, 
             dataset=args.dataset, 
             num_classes=args.num_classes, 
             num_instances=args.num_instances, 
             num_bags=args.num_bags)
