import datetime
import pytz
import os
import torch
from matplotlib import pyplot as plt
import glob
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import cv2
from sklearn.manifold import TSNE
import torchvision.transforms as transforms


def get_date():
    return datetime.datetime.now(
        pytz.timezone('Asia/Tokyo')).strftime('%Y.%m.%d.%H.%M.%S')


def make_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)  # fix the initial value of the network weight
    torch.cuda.manual_seed(seed)  # for cuda
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True  # choose the determintic algorithm


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


def get_rampup_weight(weight, iteration, rampup):
    alpha = weight * sigmoid_rampup(iteration, rampup)
    return alpha


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def save_confusion_matrix(cm, path, title=''):
    sns.heatmap(cm, annot=True, cmap='Blues_r', fmt=".2f")
    plt.xlabel('pred')
    plt.ylabel('GT')
    plt.title(title)
    plt.savefig(path)
    plt.close()


def cal_OP_PC_mIoU(cm):
    num_classes = cm.shape[0]

    TP_c = np.zeros(num_classes)
    for i in range(num_classes):
        TP_c[i] = cm[i][i]

    FP_c = np.zeros(num_classes)
    for i in range(num_classes):
        FP_c[i] = cm[i, :].sum()-cm[i][i]

    FN_c = np.zeros(num_classes)
    for i in range(num_classes):
        FN_c[i] = cm[:, i].sum()-cm[i][i]

    OP = TP_c.sum() / (TP_c+FP_c).sum()
    PC = (TP_c/(TP_c+FP_c)).mean()
    mIoU = (TP_c/(TP_c+FP_c+FN_c)).mean()

    return OP, PC, mIoU


def create_video_cm(cwd):
    size = (640, 480)  # サイズ指定
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 保存形式
    save = cv2.VideoWriter(cwd+'cm_uni.mp4', fourcc, 10.0, size)  # 動画を保存する形を作成

    print("saving...")
    for epoch in range(100):
        img_path = glob.glob(cwd+'*True*cm_%s.png' % (epoch+1))[0]
        print(img_path)
        img = cv2.imread(img_path)  # 画像を読み込む
        img = cv2.resize(img, (640, 480))  # 上でサイズを指定していますが、念のため
        save.write(img)  # 保存

    print("saved")
    save.release()  # ファイルを閉じる


def visualize_feature_space(path, label, epoch):
    feature = np.load(path)
    f_embedded = TSNE(n_components=2).fit_transform(feature)
    for i in range(label.max()+1):
        x = f_embedded[label == i][:, 0]
        y = f_embedded[label == i][:, 1]
        plt.scatter(x, y, label=i, alpha=0.3)
    plt.axis('off')
    plt.legend()
    plt.title('epoch: %d' % (epoch))
    plt.savefig(path[:-4]+'.png')
    plt.close()


def create_video_feature_space(cwd):
    size = (640, 480)  # サイズ指定
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # 保存形式
    save = cv2.VideoWriter(cwd+'test_feature_space_xx.mp4',
                           fourcc, 1.0, size)  # 動画を保存する形を作成

    print("saving...")
    for epoch in range(10, 101, 10):
        img_path = glob.glob(cwd+'*False*test_feature_%s.png' % (epoch))[0]
        print(img_path)
        img = cv2.imread(img_path)  # 画像を読み込む
        img = cv2.resize(img, (640, 480))  # 上でサイズを指定していますが、念のため
        save.write(img)  # 保存

    print("saved")
    save.release()  # ファイルを閉じる


def show_figure(cwd):

    # path = 'add_proportion_loss/fpl_cifar10_True_0.001_10_1_simple_confidence_100_64_0.01_1'
    # plt.plot(np.load(cwd+path+'/test_acc.npy'),
    #          label='proportion + 0.01 * ce (lr=0.001)')
    # path = 'add_proportion_loss/fpl_cifar10_True_0.0001_10_1_simple_confidence_100_64_0.01_1'
    # plt.plot(np.load(cwd+path+'/test_acc.npy'),
    #          label='proportion + 0.01 * ce (lr=0.0001)')

    # path = 'add_proportion_loss/fpl_cifar10_True_0.001_10_1_simple_confidence_100_64_0.1_1'
    # plt.plot(np.load(cwd+path+'/test_acc.npy'),
    #          label='proportion + 0.1 * ce (lr=0.001)')
    # path = 'add_proportion_loss/fpl_cifar10_True_0.0001_10_1_simple_confidence_100_64_0.1_1'
    # plt.plot(np.load(cwd+path+'/test_acc.npy'),
    #          label='proportion + 0.1 * ce (lr=0.0001)')

    # path = 'add_proportion_loss/fpl_cifar10_True_0.001_10_1_simple_confidence_100_64_1_1'
    # plt.plot(np.load(cwd+path+'/test_acc.npy'),
    #          label='proportion + ce (lr=0.001)')
    # path = 'add_proportion_loss/fpl_cifar10_True_0.0001_10_1_simple_confidence_100_64_1_1'
    # plt.plot(np.load(cwd+path+'/test_acc.npy'),
    #          label='proportion + ce (lr=0.0001)')

    # path = 'add_proportion_loss_mini30/fpl_cifar10_True_0.001_10_1_simple_confidence_100_64_1_0'
    # plt.plot(np.load(cwd+path+'/test_acc.npy'), label='ce (lr=0.001)')

    path = 'add_proportion_loss_mini30/fpl_cifar10_True_0.001_10_1_simple_confidence_100_64_0_1'
    plt.plot(np.load(cwd+path+'/train_acc.npy'), label='proportion (lr=0.001)')
    path = 'add_proportion_loss_mini30/fpl_cifar10_True_0.0001_10_1_simple_confidence_100_64_0_1'
    plt.plot(np.load(cwd+path+'/train_acc.npy'),
             label='proportion (lr=0.0001)')

    # path = 'add_proportion_loss_mini8/fpl_cifar10_True_0.001_10_1_simple_confidence_100_64_0.1_1'
    # plt.plot(np.load(cwd+path+'/label_acc.npy'),
    #          label='proportion + 0.1 * ce  (lr=0.001)')
    # path = 'add_proportion_loss_mini8/fpl_cifar10_True_0.001_10_1_simple_confidence_100_64_0.01_1'
    # plt.plot(np.load(cwd+path+'/label_acc.npy'),
    #          label='proportion + 0.01 * ce  (lr=0.001)')
    # path = 'add_proportion_loss_mini8/fpl_cifar10_True_0.001_10_1_simple_confidence_100_64_1_1'
    # plt.plot(np.load(cwd+path+'/label_acc.npy'),
    #          label='proportion + ce (lr=0.001)')

    # path = 'add_proportion_loss_mini8/fpl_cifar10_True_0.001_10_1_simple_confidence_100_64_1_0'
    # plt.plot(np.load(cwd+path+'/label_acc.npy'), label='ce (lr=0.001)')
    # path = 'debug/fpl_cifar10_True_0.0001_10_1_simple_confidence_100_64_1_0'
    # plt.plot(np.load(cwd+path+'/label_acc.npy'), label='ce (lr=0.0001)')

    plt.legend(bbox_to_anchor=(1, 0), loc='lower right',
               borderaxespad=0, fontsize=7)
    plt.ylim(0, 1)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.savefig(cwd+'add_proportion_loss_mini8/label_acc.png')
    plt.close()


def visualize_theta(cwd, label):
    show_label = 0
    theta_true_list, theta_false_list = [], []
    for epoch in range(100):
        theta = np.load(glob.glob(cwd+'*True*theta_%s.npy' % (epoch+1))[0])
        theta = theta.reshape(-1, theta.shape[-1])
        theta_true = theta[label == show_label][:, show_label]
        theta_false = theta[label != show_label][:, show_label]
        theta_true_list.append(theta_true)
        theta_false_list.append(theta_false)
    theta_true = np.array(theta_true_list).transpose()
    theta_false = np.array(theta_false_list).transpose()

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    im1 = ax1.imshow(theta_true, aspect=1/((theta_true.shape[0]/50)))
    plt.colorbar(im1)
    ax2 = fig.add_subplot(2, 1, 2)
    im2 = ax2.imshow(theta_false, aspect=1/((theta_false.shape[0]/50)))
    plt.colorbar(im2)

    plt.savefig(cwd+'/uni_theta_%d' % show_label)


if __name__ == '__main__':
    cwd = './result/'

    show_figure(cwd)

    # create_video_cm(cwd)

    # _, train_label, _, test_label = load_cifar10(dataset_dir='./data/')
    # test_label_list = []
    # for c in range(test_label.max()+1):
    #     test_label_list.extend(test_label[test_label == c])
    # test_label = np.array(test_label_list)

    # bags_index = np.load('./obj/cifar10/uniform-SWOR-64.npy')
    # # bags_index = np.load('./obj/cifar10/xx-64.npy')
    # train_label = train_label[bags_index].reshape(-1)
    # # for epoch in tqdm(range(10, 101, 10)):
    # #     visualize_feature_space(
    # #         path=cwd+'fpl_cifar10_True_10_1_song_100_64_test_feature_%d.npy' % epoch,
    # #         label=test_label,
    # #         epoch=epoch
    # #     )

    # # create_video_feature_space(cwd)

    # visualize_theta(cwd, train_label)
