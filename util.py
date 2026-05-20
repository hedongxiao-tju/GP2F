import random
import numpy as np
import torch
import torchmetrics
import os

def NodeEva(out, test_idx, data, num_class, device):
    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
    macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
    auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_class).to(device)
    auprc = torchmetrics.classification.AveragePrecision(task="multiclass", num_classes=num_class).to(device)

    accuracy.reset()
    macro_f1.reset()
    auroc.reset()
    auprc.reset()

    pred = out.argmax(dim=1)
    acc = accuracy(pred[test_idx], data.y[test_idx])
    ma_f1 = macro_f1(pred[test_idx], data.y[test_idx])
    roc = auroc(out[test_idx], data.y[test_idx])
    prc = auprc(out[test_idx], data.y[test_idx])
       
    return acc.item(), ma_f1.item(), roc.item(), prc.item()

def GraphEva(logits, labels, num_class, device):
    logits = logits.to(device)
    labels = labels.to(device)

    accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_class).to(device)
    macro_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_class, average="macro").to(device)
    auroc = torchmetrics.classification.AUROC(task="multiclass", num_classes=num_class).to(device)
    auprc = torchmetrics.classification.AveragePrecision(task="multiclass", num_classes=num_class).to(device)
    accuracy.reset()
    macro_f1.reset()
    auroc.reset()
    auprc.reset()
    
    preds = logits.argmax(dim=1)

    acc = accuracy(preds, labels).item()
    ma_f1 = macro_f1(preds, labels).item()
    roc = auroc(logits, labels).item()
    prc = auprc(logits, labels).item()

    return acc, ma_f1, roc, prc

def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False