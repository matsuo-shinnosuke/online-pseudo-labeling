import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.len = self.data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(data)
        label = self.label[idx]
        label = torch.tensor(label).long()
        return data, label


class DatasetPseudo(torch.utils.data.Dataset):
    def __init__(self, data, label, pseudo_label):
        (n, b, c, w, h) = data.shape
        self.data = data.reshape(b*n, c, w, h)
        self.label = label.reshape(-1)
        self.pseudo_label = pseudo_label.reshape(-1)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(data)
        label = self.label[idx]
        label = torch.tensor(label).long()
        pseudo_label = self.pseudo_label[idx]
        pseudo_label = torch.tensor(pseudo_label).long()
        return data, label, pseudo_label
    
class DatasetBag(torch.utils.data.Dataset):
    def __init__(self, bags, labels, proportions):
        self.bags = bags
        self.labels = labels
        self.proportions = proportions
        self.nun_classes = (self.labels).max()+1

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        (b, w, h, c) = bag.shape
        trans_bag = torch.zeros((b, c, w, h))
        for i in range(b):
            trans_bag[i] = self.transform(bag[i])
        bag = trans_bag

        label = self.labels[idx]
        label = torch.tensor(label).long()

        proportion = np.eye(self.nun_classes)[label].mean(axis=0)
        proportion = torch.tensor(proportion).float()

        return bag, label, proportion


def set_loader(args):
    bags = np.load(f'{args.dataset_dir}/{args.dataset}_bags_{args.num_instances}_{args.num_bags}.npy')
    labels = np.load(f'{args.dataset_dir}/{args.dataset}_label_{args.num_instances}_{args.num_bags}.npy')
    proportions = np.load(f'{args.dataset_dir}/{args.dataset}_proportions_{args.num_instances}_{args.num_bags}.npy')
    
    test_data = np.load(f'{args.dataset_dir}/{args.dataset}_test_data.npy')
    test_label = np.load(f'{args.dataset_dir}/{args.dataset}_test_label.npy')

    train_bags, val_bags, train_labels, val_labels, train_proportions, val_proportions = train_test_split(
        bags, labels, proportions, test_size=args.val_size, random_state=args.seed)

    val_dataset = DatasetBag(
        bags=val_bags, labels=val_labels, proportions=val_proportions)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.mini_batch,
        shuffle=False,  num_workers=args.num_workers)

    test_dataset = Dataset(data=test_data, label=test_label)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False,  num_workers=args.num_workers)

    return train_bags, train_labels, train_proportions, val_loader, test_loader