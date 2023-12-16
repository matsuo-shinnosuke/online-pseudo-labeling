from math import nan
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import time
import gc
from sklearn.metrics import confusion_matrix
import torchvision.transforms as transforms
from load_cifar10 import load_cifar10
from load_svhn import load_svhn
from loader import set_loader
from utils import Dataset, fix_seed, make_folder, get_rampup_weight, cal_OP_PC_mIoU, save_confusion_matrix
from losses import PiModelLoss, ProportionLoss, VATLoss
from online_pseudo_labeling import FPL_each_bag as FPL
from PIL import Image

log = logging.getLogger(__name__)


class DatasetBag(torch.utils.data.Dataset):
    def __init__(self, data, label, index):
        self.data = data
        self.label = label
        self.index = index
        self.nun_classes = max(label).max()+1

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
        self.len = len(self.index)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = self.data[self.index[idx]]
        (b, w, h, c) = data.shape
        trans_data = torch.zeros((b, c, w, h))
        for i in range(b):
            trans_data[i] = self.transform(data[i])
        data = trans_data

        label = self.label[self.index[idx]]
        label = torch.tensor(label).long()

        proportion = np.eye(self.nun_classes)[label].mean(axis=0)
        proportion = torch.tensor(proportion).float()

        return data, label, proportion

class DatasetFPL(torch.utils.data.Dataset):
    def __init__(self, data, label, pseudo_label):
        (n, b, c, w, h) = data.shape
        self.data = data.reshape(b*n, c, w, h)
        self.label = label.reshape(-1)
        self.pseudo_label = pseudo_label
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
    


def evaluation(model, loader, cfg):
    model.eval()
    l1_function = ProportionLoss(metric=cfg.val_metric)
    gt, pred = [], []
    l1_list = []
    with torch.no_grad():
        for data, label, proportion in tqdm(loader, leave=False):
            (b, n, c, w, h) = data.size()
            data = data.reshape(-1, c, w, h)
            label = label.reshape(-1)

            data, proportion = data.to(cfg.device), proportion.to(cfg.device)
            y = model(data)

            gt.extend(label.cpu().detach().numpy())
            pred.extend(y.argmax(1).cpu().detach().numpy())

            confidence = F.softmax(y, dim=1)
            confidence = confidence.reshape(b, n, -1)
            pred_prop = confidence.mean(dim=1)
            l1 = l1_function(pred_prop, proportion)
            l1_list.append(l1.cpu().detach().numpy())

    l1 = np.array(l1_list).mean()
    acc = np.array(np.array(gt) == np.array(pred)).mean()
    cm = confusion_matrix(y_true=gt, y_pred=pred, normalize='true')

    return l1, acc, cm


@ hydra.main(config_path='config', config_name='config-fpl')
def main(cfg: DictConfig) -> None:

    # file name
    cwd = hydra.utils.get_original_cwd()
    result_path = cwd + cfg.result_path
    make_folder(result_path)
    result_path += 'wsi-fpl/'
    make_folder(result_path)
    result_path += '%s-' % str(cfg.dataset.name)
    result_path += '%s-' % str(cfg.num_instances)
    result_path += '%s-' % cfg.fpl.loss_f
    result_path += '-op_%d' % cfg.is_op
    result_path += '-pertur_%d' % cfg.fpl.is_pertur
    result_path += '-lr_%s' % cfg.lr
    result_path += '-eta_%s' % str(cfg.fpl.eta)
    result_path += '-pretrain_%d' % cfg.is_pretrained
    result_path += '-seed_%d' % cfg.seed
    result_path += '/'
    make_folder(result_path)
    make_folder(result_path+'model/')
    make_folder(result_path+'theta/')
    make_folder(result_path+'accum_loss/')
    make_folder(result_path+'p_label/')

    fh = logging.FileHandler(result_path+'exec.log')
    log.addHandler(fh)
    log.info(OmegaConf.to_yaml(cfg))
    log.info('cwd:%s' % cwd)

    train_data, train_label, train_loader, val_loader, test_loader = set_loader()

    #  FPL
    fpl = FPL(loader=train_loader, cfg=cfg)

    # define model, criterion and optimizer
    fix_seed(cfg.seed)
    if cfg.model == 'resnet50':
        model = resnet50(pretrained=cfg.is_pretrained)
    elif cfg.model == 'resnet18':
        # model = resnet18(weights='IMAGENET1K_V1')
        model = resnet18(pretrained=cfg.is_pretrained)
    else:
        log.info('No model!')
    model.fc = nn.Linear(model.fc.in_features, cfg.dataset.num_classes)
    model = model.to(cfg.device)

    # weight = 1 / np.array(list(fpl.k.values())).sum(axis=0)
    # weight /= weight.sum()
    # weight = torch.tensor(weight).float().to(cfg.device)
    # loss_function = nn.CrossEntropyLoss(weight=weight)
    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    fix_seed(cfg.seed)
    pseudo_label_acces = []
    flip_p_label_ratioes = []
    train_acces, val_acces, test_acces = [], [], []
    train_p_acces = []
    train_losses, val_losses, test_losses = [], [], []
    best_validation_loss = np.inf
    final_acc = 0
    for epoch in range(cfg.num_epochs):

        if epoch % cfg.n_training == 0:
            # ##### fpl ######
            s_time = time.time()
            fpl.update_theta(model)
            fpl.update_d()

        #     ##### analysis ######
        #     label_list = train_label[train_index.reshape(-1)]
        #     p_label_list = fpl.d.reshape(-1)
        #     pseudo_label_acc = np.array(
        #         np.array(label_list) == np.array(p_label_list)).mean()
        #     pseudo_label_cm = confusion_matrix(
        #         y_true=label_list, y_pred=p_label_list, normalize='true')
        #     pseudo_label_acces.append(pseudo_label_acc)

        #     # calculate flip_pseudo_label_ratio
        #     p_label = fpl.d.reshape(-1)
        #     if epoch == 0:
        #         flip_p_label_ratio = nan
        #         flip_p_label_ratioes.append(flip_p_label_ratio)
        #         temp_p_label = p_label.copy()
        #     else:
        #         flip_p_label_ratio = (p_label != temp_p_label).mean()
        #         flip_p_label_ratioes.append(flip_p_label_ratio)
        #         temp_p_label = p_label.copy()

        #     e_time = time.time()
        #     log.info('[Epoch: %d/%d (%ds)] pseudo_label flip:  %.4f, acc: %.4f' %
        #              (epoch+1, cfg.num_epochs, e_time-s_time, flip_p_label_ratio, pseudo_label_acc))
        # else:
        #     pseudo_label_acces.append(None)
        #     flip_p_label_ratioes.append(None)

        ##### train ######
        # s_time = time.time()
        # if cfg.pseudo_ratio != 1:
        #     used_train_index = []
        #     for index in train_index:
        #         if fpl.d[index[0]][index[1]] != -1:
        #             used_train_index.append(index)
        # else:
        #     used_train_index = train_index

        train_fpl_dataset = DatasetFPL(
            data=train_data,
            label=train_label,
            pseudo_label=fpl.d.reshape(-1))
        train_fpl_loader = torch.utils.data.DataLoader(
            train_fpl_dataset, batch_size=cfg.batch_size,
            shuffle=True,  num_workers=cfg.num_workers)

        model.train()
        losses = []
        pred_list, label_list, p_label_list = [], [], []
        for data, label, p_label in tqdm(train_fpl_loader, leave=False):
            data, label, p_label = data.to(cfg.device), label.to(
                cfg.device), p_label.to(cfg.device)
            y = model(data)
            loss = loss_function(y, p_label)
            # loss = loss_function(y, label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            pred_list.extend(y.argmax(1).cpu().detach().numpy())
            label_list.extend(label.cpu().detach().numpy())
            if cfg.is_soft:
                p_label = p_label.argmax(1)
            p_label_list.extend(p_label.cpu().detach().numpy())

        train_loss = np.array(losses).mean()
        pred, label = np.array(pred_list), np.array(label_list)
        p_label = np.array(p_label_list)

        train_pseudo_acc = np.array(np.array(p_label) == np.array(pred)).mean()
        train_pseudo_cm = confusion_matrix(
            y_true=p_label, y_pred=pred, normalize='true')

        train_acc = np.array(np.array(label) == np.array(pred)).mean()
        train_cm = confusion_matrix(y_true=label, y_pred=pred)

        train_losses.append(train_loss)
        train_acces.append(train_acc)
        train_p_acces.append(train_pseudo_acc)

        e_time = time.time()
        log.info('[Epoch: %d/%d (%ds)] train_loss: %.4f, p_acc: %.4f, acc: %.4f' %
                 (epoch+1, cfg.num_epochs, e_time-s_time, train_loss,
                  train_pseudo_acc, train_acc))

        ################# validation ####################
        s_time = time.time()
        val_l1, val_acc, val_cm = evaluation(model, val_loader, cfg)
        e_time = time.time()
        val_losses.append(val_l1)
        val_acces.append(val_acc)
        log.info('[Epoch: %d/%d (%ds)] val l1: %.4f, acc: %.4f' %
                 (epoch+1, cfg.num_epochs, e_time-s_time, val_l1, val_acc))

        ################## test ###################
        s_time = time.time()
        # test_l1, test_acc, test_cm = evaluation(model, test_loader, cfg)
        model.eval()
        gt, pred = [], []
        with torch.no_grad():
            for data, label in tqdm(test_loader, leave=False):
                data = data.to(cfg.device)
                y = model(data)

                gt.extend(label.cpu().detach().numpy())
                pred.extend(y.argmax(1).cpu().detach().numpy())

        test_acc = np.array(np.array(gt) == np.array(pred)).mean()
        test_cm = confusion_matrix(y_true=gt, y_pred=pred, normalize='true')

        e_time = time.time()
        test_acces.append(test_acc)
        log.info('[Epoch: %d/%d (%ds)] test acc: %.4f' %
                 (epoch+1, cfg.num_epochs, e_time-s_time, test_acc))
        log.info('====================================================')

    #     if val_l1 < best_validation_loss:
    #         torch.save(model.state_dict(), result_path + 'best_model.pth')
    #         save_confusion_matrix(cm=pseudo_label_cm, path=result_path+'cm_pseudo_label.png',
    #                               title='epoch: %d, label acc: %.4f' % (epoch+1, pseudo_label_acc))
    #         save_confusion_matrix(cm=train_pseudo_cm, path=result_path+'cm_train_pseudo.png',
    #                               title='epoch: %d, acc: %.4f' % (epoch+1, train_pseudo_acc))

    #         save_confusion_matrix(cm=train_cm, path=result_path+'cm_train.png',
    #                               title='epoch: %d, train acc: %.4f' % (epoch+1, train_acc))
    #         save_confusion_matrix(cm=val_cm, path=result_path+'cm_val.png',
    #                               title='epoch: %d, val l1: %.4f, acc: %.4f' % (epoch+1, val_l1, val_acc))
    #         save_confusion_matrix(cm=test_cm, path=result_path+'cm_test.png',
    #                               title='epoch: %d, test acc: %.4f' % (epoch+1, test_acc))

    #         best_validation_loss = val_l1
    #         final_acc = test_acc

    #     # if (epoch+1) % 10 == 0:
    #     torch.save(model.state_dict(), result_path +
    #                'model/%d.pth' % (epoch+1))
    #     with open(result_path+'theta/%d.pkl' % (epoch+1), "wb") as tf:
    #         pickle.dump(fpl.theta, tf)
    #     with open(result_path+'accum_loss/%d.pkl' % (epoch+1), "wb") as tf:
    #         pickle.dump(fpl.total_loss, tf)
    #     with open(result_path+'p_label/%d.pkl' % (epoch+1), "wb") as tf:
    #         pickle.dump(fpl.d, tf)

    #     np.save(result_path+'train_acc', train_acces)
    #     np.save(result_path+'val_acc', val_acces)
    #     np.save(result_path+'test_acc', test_acces)
    #     plt.plot(train_acces, label='train_acc')
    #     plt.plot(val_acces, label='val_acc')
    #     plt.plot(test_acces, label='test_acc')

    #     np.save(result_path+'flip_pseudo_label_ratio', flip_p_label_ratioes)
    #     np.save(result_path+'pseudo_label_mIoU', pseudo_label_acces)
    #     mask = np.arange(0, epoch+1, cfg.n_training)
    #     plt.plot(mask, np.array(flip_p_label_ratioes)[mask],
    #              label='flip_pseudo_label_ratio')
    #     plt.plot(mask, np.array(pseudo_label_acces)
    #              [mask], label='pseudo_label_acc')
    #     plt.legend()
    #     plt.ylim(0, 1)
    #     plt.xlabel('epoch')
    #     plt.ylabel('acc')
    #     plt.savefig(result_path+'curve_acc.png')
    #     plt.close()

    #     np.save(result_path+'train_loss', train_losses)
    #     np.save(result_path+'val_loss', val_losses)
    #     np.save(result_path+'test_loss', test_losses)
    #     plt.plot(train_losses, label='train_loss')
    #     plt.plot(val_losses, label='val_loss')
    #     plt.plot(test_losses, label='test_loss')
    #     plt.legend()
    #     plt.xlabel('epoch')
    #     plt.ylabel('loss')
    #     plt.savefig(result_path+'curve_loss.png')
    #     plt.close()

    # log.info(OmegaConf.to_yaml(cfg))
    # log.info('acc: %.4f' % (final_acc))
    # log.info('--------------------------------------------------')


if __name__ == '__main__':
    main()
