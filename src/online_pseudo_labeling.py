import numpy as np
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn.functional as F
from mip import *

from loader import DatasetBag

class OnlinePseudoLabeling():
    def __init__(self, 
                 train_bags, 
                 train_labels,
                 train_proportions, 
                 num_instances,
                 num_classes,
                 device,
                 sigma=1,
                 eta=5,
                 is_cumul=1,
                 is_pertur=1,
                 ):
        
        self.train_bags = train_bags
        self.train_proportions = train_proportions

        dataset = DatasetBag(bags=train_bags, labels=train_labels, proportions=train_proportions)
        self.loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False)

        self.num_bags = len(self.train_bags)
        self.num_instances = num_instances
        self.num_classes = num_classes
        self.sigma = sigma
        self.eta = eta
        self.is_cumul = is_cumul
        self.is_pertur = is_pertur
        self.device = device

        self.k = (self.train_proportions * self.num_instances).astype(int)
        self.pseudo_label = []
        for N in self.k:
            x = []
            for i, n in enumerate(N):
                x.extend([i]*n)
            np.random.shuffle(x)
            self.pseudo_label.append(x)
        self.pseudo_label = np.array(self.pseudo_label)

        self.confidence = np.zeros(
            (self.num_bags, self.num_instances, self.num_classes))
        self.loss = np.zeros(
            (self.num_bags, self.num_instances, self.num_classes))
        self.cumulative_loss = np.zeros(
            (self.num_bags, self.num_instances, self.num_classes))
        self.total_loss = np.zeros(
            (self.num_bags, self.num_instances, self.num_classes))

    def cal_loss(self, model):
        model.eval()
        with torch.no_grad():
            for i, (data, _, _) in enumerate(self.loader):
                data = data[0].to(self.device)
                y = model(data)
                confidence = F.softmax(y, dim=1).cpu().detach().numpy()
                self.confidence[i] = np.array(confidence)

        loss = self.reward(self.confidence, self.pseudo_label) # Eq.(2)
        # loss = 1-self.confidence  # Eq.(8)
        
        self.loss = loss

    def decision_pseudo_labeling(self):
        if self.is_pertur:
            perturbation = np.random.normal(
                0, self.sigma, (self.num_bags, self.num_instances, self.num_classes))
            self.cumulative_loss += self.loss
            self.total_loss = self.cumulative_loss + self.eta*perturbation
        else:
            self.cumulative_loss += self.loss
            self.total_loss = self.cumulative_loss

        for idx in tqdm(range(self.num_bags), leave=False):
            total_loss = self.total_loss[idx].reshape(-1)
            loss = self.loss[idx].reshape(-1)
            k = self.k[idx]

            m = Model()
            d = m.add_var_tensor((self.num_instances*self.num_classes, ),
                                 'd', var_type=BINARY)
            if self.is_cumul:
                m.objective = minimize(
                    xsum(x*y for x, y in zip(total_loss, d)))
            else:
                m.objective = minimize(
                    xsum(x*y for x, y in zip(loss, d)))

            for i in range(self.num_instances):
                m += xsum(d[i*self.num_classes: (i+1)*self.num_classes]) == 1
            for i in range(self.num_classes):
                m += xsum(d[i::self.num_classes]) == k[i]

            m.verbose = 0
            m.optimize()
            d = np.array(d.astype(float)).reshape(
                self.num_instances, self.num_classes)        
            self.pseudo_label[idx] = d.argmax(1)

        return self.pseudo_label

    def reward(self, confidence, label):
        confidence = confidence.reshape(-1, self.num_classes)
        label = label.reshape(-1)
        label_one_hot = np.identity(self.num_classes)[label]
        reward = np.zeros(label_one_hot.shape)
        reward[label_one_hot == 1] = (1-confidence)[label_one_hot == 1]
        reward[label_one_hot == 0] = (
            (1/self.num_classes) - confidence)[label_one_hot == 0]
        reward = reward.reshape(
            self.num_bags, self.num_instances, self.num_classes)

        return reward
    