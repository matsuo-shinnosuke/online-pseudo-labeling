from load_svhn import load_svhn
from load_cifar10 import load_cifar10
import random
import numpy as np
import matplotlib.pyplot as plt


def get_label_proportion(num_bags=100, num_classes=10):
    random.seed(0)
    label_proportion = np.random.rand(num_bags, num_classes)
    label_proportion /= label_proportion.sum(axis=1, keepdims=True)
    return label_proportion


def create_bags_index(index, label, label_proportion, num_bags=1000, num_instances=64):
    random.seed(0)
    num_classes = label_proportion.shape[1]

    bags_index = []
    for n in range(num_bags):
        bag_index = []
        for c in range(num_classes):
            # c_index = np.where(label == c)[0]
            c_index = index[label[index] == c]
            if (c+1) != num_classes:
                num_c = int(num_instances*label_proportion[n][c])
            else:
                num_c = num_instances-len(bag_index)

            sample_c_index = random.sample(list(c_index), num_c)
            bag_index.extend(sample_c_index)

        random.shuffle(bag_index)
        bags_index.append(bag_index)

    bags_index = np.array(bags_index)
    # bags_index.shape => (num_bags, num_instances)

    return bags_index


if __name__ == '__main__':
    # num_bags_list = [1600, 800, 400, 200, 100, 50, 25]
    num_bags_list = [100] * 7
    num_instances_list = [64, 128, 256, 512, 1024, 2048, 4096]
    # num_bags_list = [1600]
    # num_instances_list = [64]

    dataset = 'cifar10'
    for num_bags, num_instances in zip(num_bags_list, num_instances_list):
        print(num_bags, num_instances)
        print(dataset)
        if dataset == 'svhn':
            train_data, train_label, _, _ = load_svhn(dataset_dir='./data/')
        if dataset == 'cifar10':
            train_data, train_label, _, _ = load_cifar10(dataset_dir='./data/')
        num_classes = train_label.max()+1

        N_train = len(train_data)
        index = np.arange(N_train)
        np.random.seed(0)
        np.random.shuffle(index)
        train_index = index[: int(N_train*0.7)]
        val_index = index[int(N_train*0.7):]

        label_proportion = get_label_proportion(num_bags, num_classes)
        bags_index = create_bags_index(
            train_index, train_label, label_proportion, num_bags, num_instances)
        print(bags_index.shape, label_proportion.shape)
        np.save('./dataset/%s/bias-%d-%d-train_index.npy' %
                (dataset, num_bags, num_instances), bags_index)

        label_proportion = get_label_proportion(
            num_bags=num_bags, num_classes=10)
        bags_index = create_bags_index(
            val_index, train_label, label_proportion, num_bags=num_bags, num_instances=num_instances)
        print(bags_index.shape, label_proportion.shape)
        np.save('./dataset/%s/bias-%d-%d-val_index.npy' %
                (dataset, num_bags, num_instances), bags_index)
