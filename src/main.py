import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from online_pseudo_labeling import OnlinePseudoLabeling
import time
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from arguments import parse_option
from loader import DatasetPseudo, set_loader
from losses import ProportionLoss
from utils import reproductibility

def set_model(args):
    if args.model_backbone == 'resnet50':
        if args.is_pretrain:
            model = resnet50(weights='DEFAULT')
        else:
            model = resnet50()

    elif args.model_backbone == 'resnet18':
        if args.is_pretrain:
            model = resnet18(weights='DEFAULT')
        else:
            model = resnet18()
    else:
        ValueError(args.model)
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    
    return model

if __name__ == '__main__':
    args = parse_option()
    reproductibility(seed=args.seed)

    train_bags, train_labels, train_proportions, val_loader, test_loader = set_loader(args)
    model = set_model(args).to(args.device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    online_pseudo_labeling = OnlinePseudoLabeling(
        train_bags=train_bags,
        train_labels=train_labels,
        train_proportions=train_proportions,
        num_instances=args.num_instances,
        num_classes=args.num_classes,
        device=args.device,
        sigma=args.sigma,
        eta=args.eta,
        is_cumul=args.is_cumul,
        is_pertur=args.is_pertur,
        )

    train_acces, val_acces, test_acces = [], [], []
    train_p_acces = []
    train_losses, val_losses, test_losses = [], [], []
    best_val_loss = np.inf
    for epoch in range(args.num_epochs):
        ### Training by online pseudo labeling ###
        s_time = time.time()
        online_pseudo_labeling.cal_loss(model)
        pseudo_label = online_pseudo_labeling.decision_pseudo_labeling()

        train_dataset = DatasetPseudo(
            data=train_bags,
            label=train_labels,
            pseudo_label=pseudo_label)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=True,  num_workers=args.num_workers)

        model.train()
        losses = []
        pred_list, label_list, p_label_list = [], [], []
        for data, label, p_label in tqdm(train_loader, leave=False):
            data, label = data.to(args.device), label.to(args.device)
            p_label = p_label.to(args.device)

            y = model(data)
            loss = loss_function(y, p_label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            pred_list.extend(y.argmax(1).cpu().detach().numpy())
            label_list.extend(label.cpu().detach().numpy())
            p_label_list.extend(p_label.cpu().detach().numpy())

        train_loss = np.array(losses).mean()
        pred, label = np.array(pred_list), np.array(label_list)
        p_label = np.array(p_label_list)

        train_pseudo_acc = np.array(p_label == pred).mean()
        train_pseudo_cm = confusion_matrix(
            y_true=p_label, y_pred=pred, normalize='true')

        train_acc = np.array(label == pred).mean()
        train_cm = confusion_matrix(y_true=label, y_pred=pred)

        train_losses.append(train_loss)
        train_acces.append(train_acc)
        train_p_acces.append(train_pseudo_acc)
        
        e_time = time.time()
        print('[Epoch: %d/%d (%ds)] train_loss: %.4f, p_acc: %.4f, acc: %.4f' %
                 (epoch+1, args.num_epochs, e_time-s_time, train_loss,
                  train_pseudo_acc, train_acc))

        ################# validation ####################
        s_time = time.time()

        model.eval()
        losses = []
        pred_list, label_list = [], []
        l1_function = ProportionLoss(metric='l1')
        with torch.no_grad():
            for data, label, proportion in tqdm(val_loader, leave=False):
                (b, n, c, w, h) = data.size()
                data = data.reshape(-1, c, w, h)
                label = label.reshape(-1)

                data, proportion = data.to(args.device), proportion.to(args.device)
                y = model(data)

                label_list.extend(label.cpu().detach().numpy())
                pred_list.extend(y.argmax(1).cpu().detach().numpy())

                confidence = F.softmax(y, dim=1)
                confidence = confidence.reshape(b, n, -1)
                pred_prop = confidence.mean(dim=1)
                loss = l1_function(pred_prop, proportion)
                losses.append(loss.item())

        val_loss = np.array(losses).mean()
        pred, label = np.array(pred_list), np.array(label_list)
        val_acc = np.array(label == pred).mean()
        val_cm = confusion_matrix(y_true=label, y_pred=pred, normalize='true')
    
        val_losses.append(val_loss)
        val_acces.append(val_acc)
        
        e_time = time.time()
        print('[Epoch: %d/%d (%ds)] val l1: %.4f, acc: %.4f' %
                 (epoch+1, args.num_epochs, e_time-s_time, val_loss, val_acc))
        
        ################## test ###################
        s_time = time.time()
        model.eval()
        gt, pred = [], []
        with torch.no_grad():
            for data, label in tqdm(test_loader, leave=False):
                data = data.to(args.device)
                y = model(data)

                gt.extend(label.cpu().detach().numpy())
                pred.extend(y.argmax(1).cpu().detach().numpy())

        test_acc = np.array(np.array(gt) == np.array(pred)).mean()
        test_cm = confusion_matrix(y_true=gt, y_pred=pred, normalize='true')

        e_time = time.time()
        test_acces.append(test_acc)
        print('[Epoch: %d/%d (%ds)] test acc: %.4f' %
                 (epoch+1, args.num_epochs, e_time-s_time, test_acc))
        print('====================================================')

        if val_loss < best_val_loss:
            torch.save(model.state_dict(), f'{args.output_dir}/best_model.pth')
            best_validation_loss = val_loss
            best_test_acc = test_acc
    
    print(('Accuracy: %.4f' % (best_test_acc)))