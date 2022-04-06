#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
post_aly.py: 
"""
import numpy as np
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from numpy import interp
from itertools import cycle
from sklearn.metrics import auc, roc_auc_score, roc_curve, accuracy_score, f1_score, balanced_accuracy_score, brier_score_loss, log_loss, precision_score, recall_score
from dl import feat_dict, args, logger
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os.path as osp
from scipy.special import softmax
import pandas as pd
from pathlib import Path
import os



def process_pred(pred_dict):
    y_true, y_predscore = pred_dict["y_true"], pred_dict["y_predscore"]
    label_map = feat_dict[args.node_label_cols.split("_cat")[0]]
    if not args.graph_info:
        if '3' in label_map.keys():
            label_map.pop('3')
    classes = sorted([int(i) for i in label_map.keys()])

    # Note: Remove the background
    if args.graph_info:
        non_bg_idx = np.where(y_true != classes[-1])
        y_true = y_true[non_bg_idx]
        y_predscore = y_predscore[non_bg_idx][:, :-1]
        classes.pop()
    
    labels = [label_map[str(label_id)] for label_id in classes]

    y_pred = np.argmax(y_predscore, axis=1)
    n_classes = len(classes)
    y_true_onehot = label_binarize(y_true, classes=classes)

    proc_pred_dict = {
        "y_true": y_true,
        "y_true_onehot": y_true_onehot,
        "y_pred": y_pred,
        "y_predscore": y_predscore,
        "classes": classes,
        "n_classes": n_classes,
        "labels": labels,
    }
    return proc_pred_dict


def cal_pred_metrics(pred_dict):
    proc_pred_dict = process_pred(pred_dict)
    y_true = proc_pred_dict["y_true"]
    y_true_onehot = proc_pred_dict["y_true_onehot"]

    y_pred = proc_pred_dict["y_pred"]
    y_predscore = proc_pred_dict["y_predscore"]

    metric_dict = dict()
    cal_basic_metric(y_true, y_pred, y_true_onehot, y_predscore, metric_dict)
    cal_auc(y_true_onehot, y_predscore, metric_dict)

    return metric_dict


class PlotAly(object):
    def __init__(self, pred_dict, plot_path):
        proc_pred_dict = process_pred(pred_dict)
        self.y_true = proc_pred_dict["y_true"]
        self.y_true_onehot = proc_pred_dict["y_true_onehot"]

        self.y_pred = proc_pred_dict["y_pred"]
        self.y_predscore = proc_pred_dict["y_predscore"]

        self.n_classes = proc_pred_dict["n_classes"]
        self.labels = proc_pred_dict["labels"]

        self.plot_path = plot_path

    def plot_metrics(self):
        fpr, tpr, roc_auc = get_fpr_tpr(self.n_classes, self.y_true_onehot, self.y_predscore)
        plot_roc(self.n_classes, fpr, tpr, roc_auc, self.plot_path)
        plot_merged_roc(self.n_classes, fpr, tpr, roc_auc, self.plot_path)
        plot_confusion_mat(self.y_true, self.y_pred, self.labels, self.plot_path)


def get_fpr_tpr(n_classes, y_test_onehot, y_test_predscore):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_onehot[:, i], y_test_predscore[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    # Micro-average: Calculate metrics globally by considering each element of the label indicator matrix as a label.
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_onehot.ravel(), y_test_predscore.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr, tpr, roc_auc


def cal_basic_metric(y_test, y_test_pred, y_test_onehot, y_test_predscore, metric_dict):
    y_test_prob = softmax(y_test_predscore, axis=1)
    
    acc = accuracy_score(y_test, y_test_pred)
    logger.debug(f'Test Acc: {acc}')

    bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    logger.debug(f'Test Ballance Acc: {bal_acc}')

    f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
    logger.debug(f'Test Weighted F1: {f1_weighted}')
    
    pr = precision_score(y_test, y_test_pred, average = 'macro')
    logger.debug(f'Test Precision: {pr}')
    
    recall = recall_score(y_test, y_test_pred, average = 'macro')
    logger.debug(f'Test Recall: {recall}')
    
    f1_macro = f1_score(y_test, y_test_pred, average='macro')
    logger.debug(f'Test F1: {f1_macro}')
    
    brier = brier_score(y_test_onehot, y_test_prob)
    logger.debug(f'Brier Score: {brier}')
    
    ce = log_loss(y_test_onehot, y_test_prob)
    logger.debug(f'Cross Entropy Loss: {ce}')
       
    metric_dict['acc'] = acc
    metric_dict['balance_acc'] = bal_acc
    metric_dict['f1_weighted'] = f1_weighted
    metric_dict['f1_macro'] = f1_macro
    metric_dict['brier_score'] = brier
    metric_dict['cross_entropy'] = ce
    metric_dict['pr'] = pr
    metric_dict['recall'] = recall

def brier_score(y_test_onehot, y_test_prob):
    N = y_test_onehot.shape[0]
    brier = sum(sum((y_test_onehot-y_test_prob)**2))/N
    return brier

def cal_auc(y_test_onehot, y_test_predscore, metric_dict):
    try:
        macro_roc_auc_ovo = roc_auc_score(y_test_onehot, y_test_predscore, multi_class="ovo",
                                          average="macro")
    except ValueError:
        macro_roc_auc_ovo = 0
    # 'weighted': Calculate metrics for each label, and find their average,
    # weighted by support (the number of true instances for each label).
    try:
        weighted_roc_auc_ovo = roc_auc_score(y_test_onehot, y_test_predscore, multi_class="ovo",
                                             average="weighted")
    except ValueError:
        weighted_roc_auc_ovo = 0
    try:
        macro_roc_auc_ovr = roc_auc_score(y_test_onehot, y_test_predscore, multi_class="ovr",
                                          average="macro")
    except ValueError:
        macro_roc_auc_ovr = 0
    try:
        weighted_roc_auc_ovr = roc_auc_score(y_test_onehot, y_test_predscore, multi_class="ovr",
                                             average="weighted")
    except ValueError:
        weighted_roc_auc_ovr = 0
    logger.info("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
                "(weighted by prevalence)"
                .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
    logger.info("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
                "(weighted by prevalence)"
                .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))

    metric_dict['macro_auc_ovo'] = macro_roc_auc_ovo
    metric_dict['weighted_auc_ovo'] = weighted_roc_auc_ovo
    metric_dict['macro_auc_ovr'] = macro_roc_auc_ovr
    metric_dict['weighted_auc_ovr'] = weighted_roc_auc_ovr


def plot_roc(n_classes, fpr, tpr, roc_auc, save_path):
    label_encode_dict = {str(k): v for k, v in feat_dict[args.node_label_cols.split("_cat")[0]].items()}
    if n_classes == 3:
        n_row, n_col = 1, 3
        fig_w, fig_h = 20, 10
    elif n_classes == 5:
        n_row, n_col = 2, 3
        fig_w, fig_h = 27, 20
    else:
        raise ValueError
    fig, axs = plt.subplots(n_row, n_col, figsize=(fig_w, fig_h))

    # fig, axs = plt.subplots(1, 3, figsize=(14, 20))
    lw = 2

    for i, class_id in enumerate(range(n_classes)):
        if n_row == 1:
            ax = axs[i]
        else:
            row, col = np.unravel_index(i, (n_row, n_col))
            ax = axs[row][col]
        ax.plot(fpr[class_id], tpr[class_id], color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[class_id])
        ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax.set_xlabel('False Positive Rate', fontsize=25)
        ax.set_ylabel('True Positive Rate', fontsize=25)
        if i != 0:
            ax.set_ylabel('')

        ax.set_title(f'ROC for {label_encode_dict[str(class_id)]}-{class_id}', fontsize=25)
        ax.legend(loc="lower right", fontsize=20)
    plt.show()
    fig.savefig(osp.join(save_path, f"ROC_for_each_class.png"))


def plot_merged_roc(n_classes, fpr, tpr, roc_auc, save_path):
    # First aggregate all false positive rates
    lw = 2
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    # Calculate metrics for each label, and find their unweighted mean (by linear interpolation).
    # This does not take label imbalance into account.
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(10, 10))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('Extension of Receiver operating characteristic to multi-class', fontsize=20)
    plt.legend(loc="lower right", fontsize=15)
    plt.savefig(osp.join(save_path, f"merged_roc.png"))
    plt.show()
    
    roc = pd.DataFrame(columns=['fpr','tpr'])
    roc['fpr'] = fpr["macro"]
    roc['tpr'] = tpr["macro"]
    if not Path(f"../test/ROC").exists():
        os.mkdir(f"../test/ROC")
    roc.to_csv(f'../test/ROC/roc_{args.model}.csv',index=False)
    


def plot_confusion_mat(y_true, y_pred, labels, save_path):
    conf_mat = confusion_matrix(y_true, y_pred, normalize="true")
    
    cm = pd.DataFrame(columns=['true','predict'])
    cm['true'] = y_true
    cm['predict'] = y_pred
    if not Path(f"../test/cm").exists():
        os.mkdir(f"../test/cm")
    cm.to_csv(f"../test/cm/cm_{args.model}.csv",index=False)
    
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(conf_mat, annot=True, fmt=".3f", linewidths=.5, cmap="YlGnBu", ax=ax, annot_kws={"fontsize":30})
    ax.set_ylabel('True labels', fontsize=23)
    ax.set_xlabel('Predicted labels', fontsize=23)

    ax.set_xticklabels(labels, fontsize=20)
    ax.set_yticklabels(labels, fontsize=20)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")
    plt.show()
    fig.savefig(osp.join(save_path, f"confusion_mat.png"))
    fig.savefig(osp.join(save_path, f"confusion_mat.eps"))