import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import copy
import tqdm
import random
import datetime
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# 绘制混淆矩阵图
import seaborn as sns

from model import Model
from dataset import MyDataset


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def focal_loss(inputs, targets):
    gamma = 2
    N = inputs.size(0)
    C = inputs.size(1)
    P = F.softmax(inputs, dim=1)

    class_mask = inputs.new(N, C).fill_(0)
    ids = targets.view(-1, 1)
    class_mask.scatter_(1, ids.data, 1.)
    #print(class_mask)

    probs = (P*class_mask).sum(1).view(-1,1)

    log_p = probs.log()
    #print('probs size= {}'.format(probs.size()))
    #print(probs)

    batch_loss = -(torch.pow((1-probs), gamma))*log_p
    #print('-----bacth_loss------')
    #print(batch_loss)
    loss = batch_loss.mean()
    return loss

class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def smooth_cross_entropy(x, target, smoothing=0.1):
    confidence = 1. - smoothing
    logprobs = F.log_softmax(x, dim=-1)
    nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = -logprobs.mean(dim=-1)
    loss = confidence * nll_loss + smoothing * smooth_loss
    return loss.mean()


def train_model(model, train_dataloader, val_dataloader, optimizer, exp_lr_scheduler, num_epochs=25, save_dir="checkpoints"):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0

    results = {}
    results["train_acc"] = []
    results["train_loss"] = []
    results["val_acc"] = []
    results["val_loss"] = []
    results["time_stamp"] = []

    for epoch in range(num_epochs):
        # train
        for param_group in optimizer.param_groups:
            print("LR", param_group['lr'])

        model.train()  # Set model to training mode

        metrics = defaultdict(float)
        metrics['loss'] = 0
        metrics['num_correct'] = 0
        metrics['num_total'] = 0
        epoch_samples = 0
        label_list = []
        predict_list = []

        train_bar = tqdm.tqdm(train_dataloader)
        i = 0
        for data, label in train_bar:
            data = data.to(device)
            label = label.to(device)

            # forward
            output = model(data)
            # loss = smooth_cross_entropy(output, label, 0.1)
            loss = F.cross_entropy(output, label)

            with torch.no_grad():
                predict = output.argmax(dim=1)
                num_correct = torch.eq(predict, label).sum().float().item()
                # num_correct = lam * predict.eq(targets_a.data).cpu().sum() + (1 - lam) * predict.eq(targets_b.data).cpu().sum()

                label_list.append(label.detach())
                predict_list.append(predict.detach())

            metrics['loss'] += loss.data.cpu().item()
            metrics['num_correct'] += num_correct
            metrics['num_total'] += data.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # statistics
            epoch_samples += 1
            train_bar.set_description("Train [%d|%d]: Loss: %.4f, ACC: %.4f" % (
                epoch, num_epochs,
                metrics['loss'] / epoch_samples,
                metrics['num_correct'] / metrics['num_total'],
            ))

        if epoch % 10 == 0:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, "epoch_%d.pth" % epoch))

        label_list = torch.cat(label_list, dim=0)
        predict_list = torch.cat(predict_list, dim=0)

        cm = confusion_matrix(label_list.detach().cpu().numpy(), predict_list.detach().cpu().numpy())
        precision = precision_score(label_list.detach().cpu().numpy(), predict_list.detach().cpu().numpy(), average="macro")
        recall = recall_score(label_list.detach().cpu().numpy(), predict_list.detach().cpu().numpy(), average="macro")
        f1 = f1_score(label_list.detach().cpu().numpy(), predict_list.detach().cpu().numpy(), average="macro")
        print("confusion_matrix:")
        print(cm)
        print("recall: ", recall)

        train_loss = metrics['loss'] / epoch_samples
        train_acc = metrics['num_correct'] / metrics['num_total']
        results["train_acc"].append(train_acc)
        results["train_loss"].append(train_loss)

        exp_lr_scheduler.step()

        # val
        model.eval()  # Set model to evaluate mode

        metrics = defaultdict(float)
        metrics['loss'] = 0
        metrics['num_correct'] = 0
        metrics['num_total'] = 0
        epoch_samples = 0
        label_list = []
        predict_list = []
        scores_list = []

        val_bar = tqdm.tqdm(val_dataloader)
        for data, label in val_bar:
            data = data.to(device)
            label = label.to(device)

            # forward
            with torch.no_grad():
                output = model(data)
                loss = F.cross_entropy(output, label)

                predict = output.argmax(dim=1)
                num_correct = torch.eq(predict, label).sum().float().item()

                label_list.append(label.detach())
                predict_list.append(predict.detach())
                scores_list.append(torch.softmax(output, dim=1)[:, 1])

            metrics['loss'] += loss.data.cpu().item()
            metrics['num_correct'] += num_correct
            metrics['num_total'] += data.shape[0]

            # statistics
            epoch_samples += 1
            val_bar.set_description("Val   [%d|%d]: Loss: %.4f, ACC: %.4f" % (
                epoch, num_epochs,
                metrics['loss'] / epoch_samples,
                metrics['num_correct'] / metrics['num_total'],
            ))

        label_list = torch.cat(label_list, dim=0)
        predict_list = torch.cat(predict_list, dim=0)
        scores_list = torch.cat(scores_list, dim=0)

        cm = confusion_matrix(label_list.detach().cpu().numpy(), predict_list.detach().cpu().numpy())
        precision = precision_score(label_list.detach().cpu().numpy(), predict_list.detach().cpu().numpy(), average="macro")
        recall = recall_score(label_list.detach().cpu().numpy(), predict_list.detach().cpu().numpy(), average="macro")
        f1 = f1_score(label_list.detach().cpu().numpy(), predict_list.detach().cpu().numpy(), average="macro")
        print("confusion_matrix:")
        print(cm)
        print("recall: ", recall)

        val_loss = metrics['loss'] / epoch_samples
        val_acc = metrics['num_correct'] / metrics['num_total']
        results["val_acc"].append(val_acc)
        results["val_loss"].append(val_loss)
        results["time_stamp"].append(str(datetime.datetime.now()))

        epoch_loss = val_loss
        epoch_acc = val_acc

        # deep copy the model
        if epoch_acc > best_acc:
            print("saving best model")
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(save_dir, "epoch_best.pth"))

            with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
                print("accuracy: ", epoch_acc, file=f)
                print("precision: ", precision, file=f)
                print("recall: ", recall, file=f)
                print("f1: ", f1, file=f)
                print("confusion_matrix:", file=f)
                print(cm, file=f)

            labels_name = [str(i) for i in range(2)]
            plt.figure()
            # plt.rcParams['figure.figsize'] = [15, 15]  # 设置图像大小
            sns.heatmap((cm).astype("int"), annot=True, fmt='d', cmap='Blues', 
                square=True, xticklabels=labels_name, yticklabels=labels_name)
            plt.savefig(os.path.join(save_dir, "cm.png"), format='png')
            plt.close()

            # 计算FPR, TPR和ROC曲线的阈值
            fpr, tpr, thresholds = roc_curve(label_list.detach().cpu().numpy(), scores_list.detach().cpu().numpy())
            
            # 计算AUC
            roc_auc = auc(fpr, tpr)
            
            # 绘制ROC曲线
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                    lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic example')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(save_dir, "auc.png"))
            plt.close()

        train_acc = np.array(results["train_acc"])
        train_loss = np.array(results["train_loss"])
        val_acc = np.array(results["val_acc"])
        val_loss = np.array(results["val_loss"])
        x = np.arange(1, epoch + 2)
        plt.figure()
        plt.plot(x, train_acc, label="train accuracy")
        plt.plot(x, val_acc, label="test accuracy")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.ylim(0, 1)
        plt.title('Accuracy')
        plt.legend()   #打上标签
        plt.savefig(os.path.join(save_dir, "accuracy.png"))
        plt.close()

        plt.figure()
        plt.plot(x, train_loss, label="train loss")
        plt.plot(x, val_loss, label="test loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.ylim(0, 1.5)
        plt.title('Loss')
        plt.legend()   #打上标签
        plt.savefig(os.path.join(save_dir, "loss.png"))
        plt.close()

        # metrics_all = np.stack([train_acc, train_loss, val_acc, val_loss], axis=1)
        # np.savetxt(os.path.join(save_dir, "metrics_all.txt"), metrics_all, fmt="%.4f")
        with open(os.path.join(save_dir, "metrics_all.txt"), "w") as f:
            for j in range(len(train_acc)):
                f.write("%s, %.4f, %.4f, %.4f, %.4f\n" % (
                    results["time_stamp"][j],
                    train_acc[j],
                    train_loss[j],
                    val_acc[j],
                    val_loss[j]
                ))
        
    print(best_acc)

if __name__ == "__main__":
    seed_torch(3047)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model(ndims=2, c_in=12, c_enc=[64, 128, 256], k_enc=[7, 3, 3], 
            s_enc=[1, 2, 2], nres_enc=6, norm="InstanceNorm", num_classes=2)
    model = model.to(device)

    train_set = MyDataset(data_root="data/pkl", phase="train")
    valid_set = MyDataset(data_root="data/pkl", phase="val")
    train_dataloader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
    val_dataloader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    # optimizer_ft = optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.0001)

    exp_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=30, eta_min=2e-6)

    model = train_model(model, train_dataloader, val_dataloader, optimizer_ft, exp_lr_scheduler, num_epochs=30, save_dir="checkpoints/resnet18_auc")