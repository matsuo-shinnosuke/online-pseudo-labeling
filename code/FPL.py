# Follow-the-Perturbed-Leader (FPL)
import random
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from mip import *
# from pulp import *
import torchvision.transforms as transforms
from PIL import Image


class FPL_each_bag:
    def __init__(self, loader, cfg):
        np.random.seed(0)

        self.loader = loader

        self.num_bags = len(loader)
        self.num_instances = cfg.num_instances
        self.num_classes = cfg.dataset.num_classes
        self.sigma = cfg.fpl.sigma
        self.eta = cfg.fpl.eta
        self.loss_f = cfg.fpl.loss_f
        self.is_op = cfg.is_op
        self.is_pertur = cfg.fpl.is_pertur

        self.cfg = cfg

        k = []
        for _, label, _ in self.loader:
            k.append(np.eye(self.num_classes)[label[0]].sum(axis=0))
        self.k = np.array(k)

        d = []
        for _, label, _ in self.loader:
            label = label[0].numpy()
            np.random.shuffle(label)
            d.append(label)
        self.d = np.array(d)

        self.confidence = np.zeros(
            (self.num_bags, self.num_instances, self.num_classes))
        self.theta = np.zeros(
            (self.num_bags, self.num_instances, self.num_classes))
        self.cumulative_loss = np.zeros(
            (self.num_bags, self.num_instances, self.num_classes))
        self.total_loss = np.zeros(
            (self.num_bags, self.num_instances, self.num_classes))

    def update_theta(self, model):
        model.eval()
        with torch.no_grad():
            for i, (data, _, _) in enumerate(self.loader):
                data = data[0].to(self.cfg.device)
                y = model(data)
                confidence = F.softmax(y, dim=1).cpu().detach().numpy()
                self.confidence[i] = np.array(confidence)

        if self.loss_f == 'simple_confidence':
            loss = 1-self.confidence
        elif self.loss_f == 'reward1':
            loss = self.reward1(self.confidence, self.d)
        elif self.loss_f == 'reward2':
            loss = self.reward2(self.confidence, self.d)
        else:
            print('Error: No loss function!')

        self.theta = loss

    def update_d(self):
        if self.is_pertur:
            perturbation = np.random.normal(
                0, self.sigma, (self.num_bags, self.num_instances, self.num_classes))
            self.cumulative_loss += self.theta
            self.total_loss = self.cumulative_loss + self.eta*perturbation
        else:
            self.cumulative_loss += self.theta
            self.total_loss = self.cumulative_loss

        for idx in tqdm(range(self.num_bags), leave=False):
            total_loss = self.total_loss[idx].reshape(-1)
            theta = self.theta[idx].reshape(-1)
            k = self.k[idx]

            m = Model()
            d = m.add_var_tensor((self.num_instances*self.num_classes, ),
                                 'd', var_type=BINARY)
            if self.is_op:
                m.objective = minimize(
                    xsum(x*y for x, y in zip(total_loss, d)))
            else:
                m.objective = minimize(
                    xsum(x*y for x, y in zip(theta, d)))

            for i in range(self.num_instances):
                m += xsum(d[i*self.num_classes: (i+1)*self.num_classes]) == 1
            for i in range(self.num_classes):
                m += xsum(d[i::self.num_classes]) == k[i]

            m.verbose = 0
            m.optimize()
            d = np.array(d.astype(float)).reshape(
                self.num_instances, self.num_classes)
            self.d[idx] = d.argmax(1)

            # total_loss = total_loss.reshape(self.num_instances, self.num_classes)
            # total_loss = total_loss[d.astype(bool)]
            # for c in range(self.num_classes):
            #     c_index = np.where(self.d[idx] == c)[0]
            #     c_loss = total_loss[self.d[idx] == c]
            #     c_sorted_index = c_index[np.argsort(c_loss)]
            #     c_not_used_index = c_sorted_index[self.k[idx][c]:]
            #     self.d[idx][c_not_used_index] = -1

    def reward1(self, confidence, label):
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

    def reward2(self, confidence, label):
        confidence = confidence.reshape(-1, self.num_classes)
        label = label.reshape(-1)
        label_one_hot = np.identity(self.num_classes)[label]
        reward = np.zeros(label_one_hot.shape)
        reward[label_one_hot == 1] = (1-confidence)[label_one_hot == 1]
        reward[label_one_hot == 0] = (confidence.max(
            axis=1, keepdims=1) - confidence)[label_one_hot == 0]
        reward = reward.reshape(
            self.num_bags, self.num_instances, self.num_classes)

        return reward
