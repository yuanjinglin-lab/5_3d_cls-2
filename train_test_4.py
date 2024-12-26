import os
import torch
import copy
import tqdm
import random
import datetime
import numpy as np
from collections import defaultdict
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from model import Model
from dataset import MyDataset


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化以保证实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
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

    probs = (P * class_mask).sum(1).view(-1, 1)
    log_p = probs.log()
    batch_loss = -(torch.pow((1 - probs), gamma)) * log_p
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

    # 初始化结果存储
    results = {
        "train_acc": [],
        "train_loss": [],
        "val_acc": [],
        "val_loss": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auc": []
    }

    for epoch in range(num_epochs):
        # 打印当前学习率
        for param_group in optimizer.param_groups:
            print("LR", param_group['lr'])

        model.train()  # 设置模型为训练模式

        metrics = defaultdict(float)
        metrics['loss'] = 0
        metrics['num_correct'] = 0
        metrics['num_total'] = 0
        epoch_samples = 0
        label_list = []
        predict_list = []
        scores_list = []

        train_bar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        for data, label in train_bar:
            data = data.to(device)
            label = label.to(device)

            # 前向传播
            output = model(data)
            loss = F.cross_entropy(output, label)

            with torch.no_grad():
                predict = output.argmax(dim=1)
                num_correct = torch.eq(predict, label).sum().float().item()

                label_list.append(label.detach())
                predict_list.append(predict.detach())

            metrics['loss'] += loss.data.cpu().item()
            metrics['num_correct'] += num_correct
            metrics['num_total'] += data.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新进度条
            epoch_samples += 1
            train_bar.set_postfix({
                "Loss": f"{metrics['loss'] / epoch_samples:.4f}",
                "ACC": f"{metrics['num_correct'] / metrics['num_total']:.4f}"
            })

        if (epoch + 1) % 10 == 0:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch + 1}.pth"))

        label_list = torch.cat(label_list, dim=0)
        predict_list = torch.cat(predict_list, dim=0)

        cm = confusion_matrix(label_list.cpu().numpy(), predict_list.cpu().numpy())
        precision = precision_score(label_list.cpu().numpy(), predict_list.cpu().numpy(), average="macro")
        recall = recall_score(label_list.cpu().numpy(), predict_list.cpu().numpy(), average="macro")
        f1 = f1_score(label_list.cpu().numpy(), predict_list.cpu().numpy(), average="macro")
        print("Confusion Matrix:")
        print(cm)
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        train_loss = metrics['loss'] / epoch_samples
        train_acc = metrics['num_correct'] / metrics['num_total']
        results["train_acc"].append(train_acc)
        results["train_loss"].append(train_loss)

        exp_lr_scheduler.step()

        # 验证阶段
        model.eval()  # 设置模型为评估模式

        metrics = defaultdict(float)
        metrics['loss'] = 0
        metrics['num_correct'] = 0
        metrics['num_total'] = 0
        epoch_samples = 0
        label_list = []
        predict_list = []
        scores_list = []

        val_bar = tqdm.tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation")
        for data, label in val_bar:
            data = data.to(device)
            label = label.to(device)

            # 前向传播
            with torch.no_grad():
                output = model(data)
                loss = F.cross_entropy(output, label)

                predict = output.argmax(dim=1)
                num_correct = torch.eq(predict, label).sum().float().item()

                label_list.append(label.detach())
                predict_list.append(predict.detach())
                scores_list.append(torch.softmax(output, dim=1)[:, 1])  # 假设是二分类

            metrics['loss'] += loss.data.cpu().item()
            metrics['num_correct'] += num_correct
            metrics['num_total'] += data.shape[0]

            # 更新进度条
            epoch_samples += 1
            val_bar.set_postfix({
                "Loss": f"{metrics['loss'] / epoch_samples:.4f}",
                "ACC": f"{metrics['num_correct'] / metrics['num_total']:.4f}"
            })

        label_list = torch.cat(label_list, dim=0)
        predict_list = torch.cat(predict_list, dim=0)
        scores_list = torch.cat(scores_list, dim=0)

        cm = confusion_matrix(label_list.cpu().numpy(), predict_list.cpu().numpy())
        precision = precision_score(label_list.cpu().numpy(), predict_list.cpu().numpy(), average="macro")
        recall = recall_score(label_list.cpu().numpy(), predict_list.cpu().numpy(), average="macro")
        f1 = f1_score(label_list.cpu().numpy(), predict_list.cpu().numpy(), average="macro")
        print("Confusion Matrix:")
        print(cm)
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        val_loss = metrics['loss'] / epoch_samples
        val_acc = metrics['num_correct'] / metrics['num_total']
        results["val_acc"].append(val_acc)
        results["val_loss"].append(val_loss)

        # 计算 AUC
        fpr, tpr, thresholds = roc_curve(label_list.cpu().numpy(), scores_list.cpu().numpy())
        roc_auc = auc(fpr, tpr)
        results["auc"].append(roc_auc)

        # 记录其他指标
        results["precision"].append(precision)
        results["recall"].append(recall)
        results["f1"].append(f1)

        # 保存当前折叠的训练和验证结果
        # 可以在此处添加代码将结果保存到文件或其他存储介质

        # 保存最佳模型
        if val_acc > best_acc:
            print(f"Saving best model for epoch {epoch + 1}")
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(save_dir, "epoch_best.pth"))

            with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
                f.write(f"Accuracy: {val_acc:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1 Score: {f1:.4f}\n")
                f.write(f"AUC: {roc_auc:.4f}\n")
                f.write("Confusion Matrix:\n")
                f.write(f"{cm}\n")

            labels_name = [str(i) for i in range(2)]  # 根据类别数调整
            plt.figure()
            sns.heatmap(cm.astype(int), annot=True, fmt='d', cmap='Blues',
                        square=True, xticklabels=labels_name, yticklabels=labels_name)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.savefig(os.path.join(save_dir, "cm.png"), format='png')
            plt.close()

            # 绘制ROC曲线
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(save_dir, "auc.png"))
            plt.close()

    # 绘制准确率和损失曲线
    train_acc_np = np.array(results["train_acc"])
    train_loss_np = np.array(results["train_loss"])
    val_acc_np = np.array(results["val_acc"])
    val_loss_np = np.array(results["val_loss"])
    auc_np = np.array(results["auc"])
    precision_np = np.array(results["precision"])
    recall_np = np.array(results["recall"])
    f1_np = np.array(results["f1"])
    epochs_range = np.arange(1, num_epochs + 1)

    plt.figure()
    plt.plot(epochs_range, train_acc_np, label="Train Accuracy")
    plt.plot(epochs_range, val_acc_np, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_dir, "accuracy.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs_range, train_loss_np, label="Train Loss")
    plt.plot(epochs_range, val_loss_np, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, max(max(train_loss_np), max(val_loss_np)) + 0.5)
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss.png"))
    plt.close()

    # 保存所有指标
    with open(os.path.join(save_dir, "metrics_all.txt"), "w") as f:
        for j in range(len(train_acc_np)):
            f.write(f"{results['train_acc'][j]:.4f}, {results['train_loss'][j]:.4f}, {results['val_acc'][j]:.4f}, {results['val_loss'][j]:.4f}, {results['precision'][j]:.4f}, {results['recall'][j]:.4f}, {results['f1'][j]:.4f}, {results['auc'][j]:.4f}\n")

    # 返回最佳准确率和所有指标
    return best_acc, results


if __name__ == "__main__":
    seed_torch(3047)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载训练集和验证集
    train_set = MyDataset(data_root="data/pkl", phase="train")
    valid_set = MyDataset(data_root="data/pkl", phase="val")

    # 合并训练集和验证集
    full_dataset = ConcatDataset([train_set, valid_set])

    # 检查数据集的长度
    print(f"Full dataset size: {len(full_dataset)}")

    # 如果数据集为空，终止程序
    if len(full_dataset) == 0:
        print("Full dataset is empty. Please check the data loading process.")
        exit(1)

    # 提取所有标签以用于 StratifiedKFold
    labels = []
    for _, label in full_dataset:
        labels.append(label)
    labels = np.array(labels)

    print(f"Labels array shape: {labels.shape}")

    # 确保有足够的数据进行交叉验证
    if len(labels) < 3:
        print("Not enough samples for 3-fold cross-validation. Need at least 3 samples.")
        exit(1)

    # 定义分层 K 折交叉验证
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=3047)

    # 初始化存储跨折叠的指标
    cv_results = {
        "best_acc": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auc": []
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n===== Fold {fold + 1} / {n_splits} =====")

        # 创建训练和验证子集
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        # 检查子集的长度
        print(f"Train subset size: {len(train_subset)}, Val subset size: {len(val_subset)}")

        # 创建 DataLoader，调整 batch_size 为 16
        train_dataloader = DataLoader(train_subset, batch_size=16, shuffle=True, num_workers=4, drop_last=True)
        val_dataloader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=4, drop_last=True)

        # 为每个折叠初始化新的模型
        model = Model(ndims=2, c_in=12, c_enc=[64, 128, 256], k_enc=[7, 3, 3],
                      s_enc=[1, 2, 2], nres_enc=6, norm="InstanceNorm", num_classes=2)
        model = model.to(device)

        # 初始化优化器和调度器
        optimizer_ft = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        exp_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=30, eta_min=2e-6)

        # 定义每个折叠的保存目录
        save_dir = f"check/resnet18_auc_fold_{fold + 1}"
        os.makedirs(save_dir, exist_ok=True)

        # 训练模型，并获取最佳准确率和指标
        best_acc, results = train_model(model, train_dataloader, val_dataloader, optimizer_ft, exp_lr_scheduler,
                                       num_epochs=30, save_dir=save_dir)

        # 记录跨折叠的指标
        cv_results["best_acc"].append(best_acc)
        cv_results["precision"].append(np.mean(results["precision"]))
        cv_results["recall"].append(np.mean(results["recall"]))
        cv_results["f1"].append(np.mean(results["f1"]))
        cv_results["auc"].append(np.mean(results["auc"]))

    # 计算并打印跨折叠的平均指标和标准差
    print("\n===== Cross-Validation Results =====")
    for metric in cv_results:
        metric_values = cv_results[metric]
        mean = np.mean(metric_values)
        std = np.std(metric_values)
        print(f"{metric.capitalize()}: {mean:.4f} ± {std:.4f}")
