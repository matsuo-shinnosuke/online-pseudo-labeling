from utils import make_folder
from load_svhn import load_svhn
from load_cifar10 import load_cifar10
import random
import numpy as np
import matplotlib.pyplot as plt
from tslearn.datasets import UCR_UEA_datasets
from sktime.datasets import load_from_tsfile_to_dataframe
from sktime.datasets import load_UCR_UEA_dataset
from sktime.datatypes._panel._convert import from_nested_to_2d_array
import pandas as pd


def get_label_proportion(num_bags=100, num_classes=10):
    label_proportion = np.random.rand(num_bags, num_classes)
    label_proportion /= label_proportion.sum(axis=1, keepdims=True)
    return label_proportion


def create_bags_index(index, label, label_proportion, num_bags=1000, num_instances=64):
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

            sample_c_index = random.choices(list(c_index), k=num_c)
            bag_index.extend(sample_c_index)

        random.shuffle(bag_index)
        bags_index.append(bag_index)

    bags_index = np.array(bags_index)
    # bags_index.shape => (num_bags, num_instances)

    return bags_index


if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    num_bags_list = [1600, 800, 400, 200, 100, 50, 25]
    num_instances_list = [64, 128, 256, 512, 1024, 2048, 4096]
    # num_bags_list = [100] * 7
    # num_bags_list = [1600]
    # num_instances_list = [64]
    # num_bags_list = [512, 256, 128, 64, 32, 16, 8]
    # num_instances_list = [64, 128, 256, 512, 1024, 2048, 4096]

    # dataset = 'cifar10'
    # for num_bags, num_instances in zip(num_bags_list, num_instances_list):
    #     print(num_bags, num_instances)
    #     print(dataset)
    #     if dataset == 'svhn':
    #         train_data, train_label, _, _ = load_svhn(dataset_dir='./data/')
    #     if dataset == 'cifar10':
    #         train_data, train_label, _, _ = load_cifar10(dataset_dir='./data/')
    #     num_classes = train_label.max()+1

    #     N_train = len(train_data)
    #     index = np.arange(N_train)
    #     np.random.seed(0)
    #     np.random.shuffle(index)
    #     train_index = index[: int(N_train*0.7)]
    #     val_index = index[int(N_train*0.7):]

    #     label_proportion = get_label_proportion(num_bags, num_classes)
    #     bags_index = create_bags_index(
    #         train_index, train_label, label_proportion, num_bags, num_instances)
    #     print(bags_index.shape, label_proportion.shape)
    #     np.save('./dataset/%s/bias-%d-%d-train_index.npy' %
    #             (dataset, num_bags, num_instances), bags_index)

    #     label_proportion = get_label_proportion(
    #         num_bags=num_bags, num_classes=10)
    #     bags_index = create_bags_index(
    #         val_index, train_label, label_proportion, num_bags=num_bags, num_instances=num_instances)
    #     print(bags_index.shape, label_proportion.shape)
    #     np.save('./dataset/%s/bias-%d-%d-val_index.npy' %
    #             (dataset, num_bags, num_instances), bags_index)

    # # UCR
    # dataset = ['InsectWingbeat', 'ElectricDevices'][1]

    # make_folder('./dataset/%s/' % dataset)
    # print(dataset)

    # train_data, train_label, test_data, test_label = UCR_UEA_datasets().load_dataset(dataset)
    # data = np.concatenate([train_data, test_data])
    # label = np.concatenate([train_label, test_label])-1
    # print(label)
    # num_classes = label.max()+1
    # print(data.shape, label.shape, num_classes)

    # N = len(data)
    # index = np.arange(N)
    # np.random.shuffle(index)
    # train_index = index[: int(N*0.4)]
    # val_index = index[int(N*0.4): int(N*0.8)]
    # test_index = index[int(N*0.8):]

    # for num_bags, num_instances in zip(num_bags_list[::-1], num_instances_list[::-1]):

    #     print(num_bags, num_instances)

    #     label_proportion = get_label_proportion(num_bags, num_classes)
    #     bags_index = create_bags_index(
    #         train_index, label, label_proportion, num_bags, num_instances)
    #     print(bags_index.shape, label_proportion.shape)
    #     np.save('./dataset/%s/bias-%d-train_index.npy' %
    #             (dataset, num_instances), bags_index)

    #     label_proportion = get_label_proportion(num_bags, num_classes)
    #     bags_index = create_bags_index(
    #         val_index, label, label_proportion, num_bags=num_bags, num_instances=num_instances)
    #     print(bags_index.shape, label_proportion.shape)
    #     np.save('./dataset/%s/bias-%d-val_index.npy' %
    #             (dataset, num_instances), bags_index)

    # np.save('./dataset/%s/test_data.npy' % dataset, data[test_index])
    # np.save('./dataset/%s/test_label.npy' % dataset, label[test_index])

    # UCR
    dataset = 'SpokenArabicDigits'

    make_folder('./dataset/%s/' % dataset)
    print(dataset)

    train_data, train_label = load_from_tsfile_to_dataframe(
        './data/%s/%sEq_TRAIN.ts' % (dataset, dataset)
    )
    test_data, test_label = load_from_tsfile_to_dataframe(
        './data/%s/%sEq_TEST.ts' % (dataset, dataset)
    )

    train_data = from_nested_to_2d_array(train_data)
    train_data = np.array(train_data).reshape(-1, 13, 65).transpose(0, 2, 1)
    train_label = np.array(train_label).astype(int)-1

    test_data = from_nested_to_2d_array(test_data)
    test_data = np.array(test_data).reshape(-1, 13, 65).transpose(0, 2, 1)
    test_label = np.array(test_label).astype(int)-1

    print(train_label)
    num_classes = train_label.max()+1
    print(train_data.shape, train_label.shape, num_classes)

    N = len(train_data)
    index = np.arange(N)
    np.random.shuffle(index)
    train_index = index[: int(N*0.5)]
    val_index = index[int(N*0.5):]

    for num_bags, num_instances in zip(num_bags_list[::-1], num_instances_list[::-1]):

        print(num_bags, num_instances)

        label_proportion = get_label_proportion(num_bags, num_classes)
        bags_index = create_bags_index(
            train_index, train_label, label_proportion, num_bags, num_instances)
        print(bags_index.shape, label_proportion.shape)
        np.save('./dataset/%s/bias-%d-train_index.npy' %
                (dataset, num_instances), bags_index)

        label_proportion = get_label_proportion(num_bags, num_classes)
        bags_index = create_bags_index(
            val_index, train_label, label_proportion, num_bags=num_bags, num_instances=num_instances)
        print(bags_index.shape, label_proportion.shape)
        np.save('./dataset/%s/bias-%d-val_index.npy' %
                (dataset, num_instances), bags_index)

    np.save('./dataset/%s/test_data.npy' % dataset, test_data)
    np.save('./dataset/%s/test_label.npy' % dataset, test_label)

    # dataset = 'ECG'

    # make_folder('./dataset/%s/' % dataset)
    # print(dataset)

    # train_df = pd.read_csv(
    #     './data/%s/mitbih_train.csv' % dataset, header=None)
    # train_data = train_df.iloc[:, :-1].to_numpy().reshape(-1, 187, 1)
    # train_data = (train_data - train_data.mean()) / train_data.std()
    # train_label = train_df.iloc[:, -1].to_numpy()
    # train_label = train_label.astype('int')
    # print(train_label)
    # num_classes = train_label.max()+1
    # print(train_data.shape, train_label.shape, num_classes)

    # test_df = pd.read_csv(
    #     './data/%s/mitbih_test.csv' % dataset, header=None)
    # test_data = test_df.iloc[:, :-1].to_numpy().reshape(-1, 187, 1)
    # test_data = (test_data - test_data.mean()) / test_data.std()
    # test_label = test_df.iloc[:, -1].to_numpy()
    # test_label = test_label.astype('int')
    # print(test_data.shape, test_label.shape)

    # N = len(train_data)
    # index = np.arange(N)
    # np.random.shuffle(index)
    # train_index = index[: int(N*0.5)]
    # val_index = index[int(N*0.5):]

    # for num_bags, num_instances in zip(num_bags_list[::-1], num_instances_list[::-1]):

    #     print(num_bags, num_instances)

    #     label_proportion = get_label_proportion(num_bags, num_classes)
    #     bags_index = create_bags_index(
    #         train_index, train_label, label_proportion, num_bags, num_instances)
    #     print(bags_index.shape, label_proportion.shape)
    #     np.save('./dataset/%s/bias-%d-train_index.npy' %
    #             (dataset, num_instances), bags_index)

    #     label_proportion = get_label_proportion(num_bags, num_classes)
    #     bags_index = create_bags_index(
    #         val_index, train_label, label_proportion, num_bags=num_bags, num_instances=num_instances)
    #     print(bags_index.shape, label_proportion.shape)
    #     np.save('./dataset/%s/bias-%d-val_index.npy' %
    #             (dataset, num_instances), bags_index)

    # np.save('./dataset/%s/test_data.npy' % dataset, test_data)
    # np.save('./dataset/%s/test_label.npy' % dataset, test_label)
