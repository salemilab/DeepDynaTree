#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
train.py: 
"""
import sys 
sys.path.append("..")
import torch
from datetime import datetime
from tensorboardX import SummaryWriter
from collections import defaultdict
from copy import deepcopy
from dl.utils import *
from dl.info import *
from dl import logger, args
from dl.dataloader import gen_label_weight
from dl.post_aly import PlotAly, cal_pred_metrics
import csv
import numpy as np
import pandas as pd
from pathlib import Path

class Trainer(object):
    def __init__(self, args, model, dl_dict, exp_path, device, **kwags):
        self.args = args
        self.device = device
        self.model = torch.nn.DataParallel(model, device_ids=range(args.num_gpus)) if args.num_gpus > 0 else model
        self.model.to(self.device)

        self.ce_loss_f = model.ce_loss
        label_weights = gen_label_weight(self.args)
        logger.info(f"Label weights: {label_weights} for {args.node_label_cols}.")
        self.label_weights = torch.tensor(label_weights, dtype=torch.float32).to(self.device)

        self.exp_path = exp_path
        self.log_path = osp.join(self.exp_path, "logs")
        self.weight_path = osp.join(self.exp_path, "weight")
        self.weight_path_dict = create_weight_dir(self.weight_path, METRICS)
        self.summary_path = osp.join(self.exp_path, "summary")
        self.pred_path = osp.join(self.exp_path, "preds")
        self.plot_path = osp.join(self.exp_path, "plots")
        os.makedirs(self.pred_path, exist_ok=True)
        os.makedirs(self.weight_path, exist_ok=True)
        os.makedirs(self.plot_path, exist_ok=True)

        # self.out_feat_path = osp.join(self.exp_path, 'out_feat', self.args.restore_metric)
        # os.makedirs(self.out_feat_path, exist_ok=True)

        self.dl_dict = dl_dict
        self.train_dl = self.dl_dict['train']
        self.valid_dl = self.dl_dict['valid']
        self.test_dl = self.dl_dict['test']

        # Record the model and config
        self.epoch = 1
        write_config(self.weight_path, self.args.mode, vars(self.args))
        logger .info(self.model)

        # Optimizer
        # Note: construct optimizer after moving model to cuda
        self.optimizer = pytorch_optimizer(self.model, args.optimizer, args.init_lr, args.weight_decay_rate)
        self.lr_scheduler = pytorch_lr_scheduler(self.optimizer, args.lr_decay_mode, args.lr_decay_step_or_patience,
                                                 args.lr_decay_rate)

        self.flag_dict = create_flag_dict(METRICS, MIN_METRICS, MAX_METRICS)
        summary_writer = SummaryWriter(self.summary_path, filename_suffix=datetime.now().strftime('_%m-%d-%y_%H-%M-%S'))
        self.summary = Summary(summary_writer, METRICS, SUMMARY_ITEMS)
        self.graph_info = args.graph_info
        torch.set_printoptions(precision=10)

    def train(self):
        logger.info('*******Building the model*******')

        if self.args.restore:
            self.load_weight()

        with timer('Duration of training'):
            for epoch in range(1, self.args.max_epochs):
                train_metrics_dict = self.train_one_epoch(self.train_dl)
                logger.debug(train_metrics_dict)
                logger.info('==> Epoch: {}, Train, {}'.format(epoch, format_metric_dict(train_metrics_dict)))
                logger.info('==================')

                valid_metrics_dict, valid_pred_dict = self.eval_one_epoch('valid', self.valid_dl)
                logger.debug('{}'.format(format_metric_dict(valid_metrics_dict)))
                # test_metrics_dict, _ = self.eval_one_epoch("test", self.test_data_loader)

                valid_aly_result_dict = self.aly_pred('valid', valid_metrics_dict, valid_pred_dict)

                # _ = self.aly_pred("test", test_data_dict)
                logger.info("==================")

                self.epoch += 1
                if valid_aly_result_dict["early_stop"]:
                    logger.info("========Best model=========")
                    logger.info("{}".format(self.flag_dict))
                    break

    def eval(self, phase):
        logger.info("*******Evaluating the model*******")
        self.load_weight()
        dl = self._get_dl(phase)

        scope = f"best-{phase}"
        metrics_dict, pred_dict = self.eval_one_epoch(scope, dl)
        aly_result_dict = self.aly_pred(phase, metrics_dict, pred_dict)
        logger.info("{}, {}, {}".format(scope, format_metric_dict(metrics_dict), format_metric_dict(aly_result_dict)))

        plot_aly = PlotAly(pred_dict, self.plot_path)
        if args.permutation_test == True:
            if not Path(f"../test/{args.model}{args.permutation_num}").exists():
                os.mkdir(f"../test/{args.model}{args.permutation_num}")

            with open(f"../test/{args.model}{args.permutation_num}/{args.permutation_feat}-{phase}.csv",'a',newline="") as f:
                w = csv.writer(f)
                row = list(metrics_dict.values())
                w.writerow(row[1:])
        else:
            plot_aly.plot_metrics()
            np.save(osp.join(self.pred_path, f"{phase}.npy"), pred_dict)

    def _get_dl(self, cohort):
        if cohort == "train":
            return self.train_dl
        elif cohort == "valid":
            return self.valid_dl
        elif cohort == "test":
            return self.test_dl

    def train_one_epoch(self, dl):
        self.model.train()
        metrics_dict = defaultdict(list)
        if self.graph_info:
            for i_batch, batched_graph in enumerate(dl, start=1):
                logger.debug(f"i_batch: {i_batch}")
                self.optimizer.zero_grad()
                y_pred, info = self.model(batched_graph)
                y_true = batched_graph.ndata['label']

                if self.args.loss == "ce":
                    loss_dict = self.ce_loss_f(y_pred, y_true, weight=self.label_weights)
                else:
                    raise NotImplementedError
                [metrics_dict[k].append(v.item()) for k, v in loss_dict.items()]

                total_loss = loss_dict['loss']

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()

                if i_batch % self.args.log_train_freq == 0:
                    logger.info("{}-[{}/{} ({:.0f}%)]: train-{}".format(
                        self.epoch, i_batch, len(dl), 100. * (i_batch / len(dl)),
                        {k: v.item() for k, v in loss_dict.items()}))
                    loss_dict["scope"] = "train_batch"
                    self.summary.add_summary(self.epoch * len(dl) + i_batch, **loss_dict)

            train_metrics_dict = {"scope": "train"}
            train_metrics_dict.update({k: np.average(np.array(v)) for k, v in metrics_dict.items()})
            return train_metrics_dict
        else:
            for i_batch, (samples, labels,sims) in enumerate(dl, start=1):
                logger.debug(f"i_batch: {i_batch}")
                self.optimizer.zero_grad()
                if args.model in ["deepset","transet"]:
                    samples = np.expand_dims(samples.cpu(),axis=2)
                    samples = torch.from_numpy(samples).float().cuda()
                y_pred, info = self.model(samples)
                y_true = labels

                if self.args.loss == "ce":
                    loss_dict = self.ce_loss_f(y_pred, y_true, weight=self.label_weights)
                else:
                    raise NotImplementedError
                [metrics_dict[k].append(v.item()) for k, v in loss_dict.items()]

                total_loss = loss_dict['loss']

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()

                if i_batch % self.args.log_train_freq == 0:
                    logger.info("{}-[{}/{} ({:.0f}%)]: train-{}".format(
                        self.epoch, i_batch, len(dl), 100. * (i_batch / len(dl)),
                        {k: v.item() for k, v in loss_dict.items()}))
                    loss_dict["scope"] = "train_batch"
                    self.summary.add_summary(self.epoch * len(dl) + i_batch, **loss_dict)

            train_metrics_dict = {"scope": "train"}
            train_metrics_dict.update({k: np.average(np.array(v)) for k, v in metrics_dict.items()})
            return train_metrics_dict

    def eval_one_epoch(self, scope, dl):
        self.model.eval()
        metrics_dict = defaultdict(list)
        y_trues, y_predscores = [], []
        with torch.no_grad():
            if self.graph_info:
                for i_batch, batched_graph in enumerate(dl, start=1):
                    if self.args.embedding:
                        y_pred, embedding, info = self.model(batched_graph)
                    else:
                        y_pred, info = self.model(batched_graph)
                    y_predscores.append(y_pred.to(torch.device("cpu")).numpy())

                    y_true = batched_graph.ndata['label']
                    y_trues.append(y_true.to(torch.device("cpu")).numpy())
                    if self.args.embedding:
                        if i_batch == 1:
                            embeds = embedding.detach().cpu().numpy()
                            labels = np.hstack((y_true.detach().cpu().numpy().reshape(-1,1),batched_graph.ndata['org_feat'].detach().cpu().numpy()[:,0].reshape(-1,1)))
                        else:
                            embeds = np.vstack((embeds,embedding.detach().cpu().numpy()))
                            labels = np.vstack((labels,np.hstack((y_true.detach().cpu().numpy().reshape(-1,1),batched_graph.ndata['org_feat'].detach().cpu().numpy()[:,0].reshape(-1,1)))))
                    # for limitation analysis
                    if self.args.mis_aly:
                        y_predlabel = np.argmax(y_pred.to(torch.device("cpu")).numpy(),axis=1)
                        y_truelabel = y_true.to(torch.device("cpu")).numpy()
                        cor = np.where(y_predlabel==y_truelabel)
                        cor_sample = batched_graph.ndata['org_feat'][cor,:].to(torch.device("cpu")).numpy()
                        mis = np.where(y_predlabel!=y_truelabel)
                        mis_sample = batched_graph.ndata['org_feat'][mis,:].to(torch.device("cpu")).numpy()
                        script_dir = osp.realpath(__file__)
                        js_f = osp.normpath(osp.join(script_dir, "../../aly/feat_dict.json"))
                        with open(js_f, 'r') as infile:
                            feat_dict = json.load(infile)
                        feat_col = feat_dict["raw_feats"]
                        cor_df = pd.DataFrame(cor_sample[0,:,:], columns = ['sim','state_id']+feat_col[0:11])
                        cor_df['label'] = y_truelabel[cor]
                        score = y_pred.to(torch.device("cpu")).numpy()
                        cor_df['pred_label'] = y_predlabel[cor]
                        cor_df['pred_score_0'] = score[cor,0].flatten()
                        cor_df['pred_score_1'] = score[cor,1].flatten()
                        cor_df['pred_score_2'] = score[cor,2].flatten()
                        mis_df = pd.DataFrame(mis_sample[0,:,:], columns = ['sim','state_id']+feat_col[0:11])
                        mis_df['label'] = y_truelabel[mis]
                        mis_df['pred_label'] = y_predlabel[mis]
                        mis_df['pred_score_0'] = score[mis,0].flatten()
                        mis_df['pred_score_1'] = score[mis,1].flatten()
                        mis_df['pred_score_2'] = score[mis,2].flatten()
                        if i_batch == 1:
                            header_flag = True
                        else:
                            header_flag = False
                        if not Path(f"../test/mis_aly").exists():
                            os.mkdir(f"../test/mis_aly")
                        cor_df.to_csv('../test/mis_aly/cor_org.csv', mode='a', index=False, header=header_flag)
                        mis_df.to_csv('../test/mis_aly/mis_org.csv', mode='a', index=False, header=header_flag)
                    if self.args.loss == "ce":
                        loss_dict = self.ce_loss_f(y_pred, y_true, weight=self.label_weights)
                    else:
                        raise NotImplementedError

                    [metrics_dict[k].append(v.item()) for k, v in loss_dict.items()]

                    if i_batch % self.args.log_valid_freq == 0:
                        logger.info('{}-[{}/{} ({:.0f}%)]: {}-{}'.format(
                            self.epoch, i_batch, len(dl), 100. * (i_batch / len(dl)),
                            scope, {k: v.item() for k, v in loss_dict.items()}))
                        loss_dict['scope'] = '{}_batch'.format(scope)
                        logger.debug('Batch: {}'.format(self.epoch * len(dl) + i_batch))
                        if self.args.mode == 'train':
                            self.summary.add_summary(self.epoch * len(dl) + i_batch, **loss_dict)
                    logger.debug('Eval_batch: {}'.format(i_batch))
                valid_metrics_dict = {'scope': scope}
                valid_metrics_dict.update({k: np.average(np.array(v)) for k, v in metrics_dict.items()})

                valid_pred_dict = {"y_true": np.concatenate(y_trues).reshape(-1),
                               "y_predscore": np.concatenate(y_predscores)}
                if self.args.embedding:
                    if not Path(f"../test/embedding").exists():
                        os.mkdir(f"../test/embedding")
                    np.save(f'../test/embedding/{args.model}_embedding.npy',embeds)
                    np.save(f'../test/embedding/{args.model}_labels.npy',labels)
                return valid_metrics_dict, valid_pred_dict
                
            else:
                for i_batch, (samples,labels,sims) in enumerate(dl, start=1):
                    if args.model in ["deepset","transet"]:
                        samples = np.expand_dims(samples.cpu(),axis=2)
                        samples = torch.from_numpy(samples).float().cuda()
                    if self.args.embedding:
                        y_pred, embedding, info = self.model(samples)
                    else:
                        y_pred, info = self.model(samples)
                    y_predscores.append(y_pred.to(torch.device("cpu")).numpy())

                    y_true = labels
                    y_trues.append(y_true.to(torch.device("cpu")).numpy())
                    if self.args.embedding:
                        if i_batch == 1:
                            embeds = embedding.detach().cpu().numpy()
                            embed_labels = np.hstack((y_true.detach().cpu().numpy().reshape(-1,1),sims.detach().cpu().numpy().reshape(-1,1)))
                        else:
                            embeds = np.vstack((embeds,embedding.detach().cpu().numpy()))
                            embed_labels = np.vstack((embed_labels,np.hstack((y_true.detach().cpu().numpy().reshape(-1,1),sims.detach().cpu().numpy().reshape(-1,1)))))
                    if self.args.loss == "ce":
                        loss_dict = self.ce_loss_f(y_pred, y_true, weight=self.label_weights)
                    else:
                        raise NotImplementedError

                    [metrics_dict[k].append(v.item()) for k, v in loss_dict.items()]

                    if i_batch % self.args.log_valid_freq == 0:
                        logger.info('{}-[{}/{} ({:.0f}%)]: {}-{}'.format(
                            self.epoch, i_batch, len(dl), 100. * (i_batch / len(dl)),
                            scope, {k: v.item() for k, v in loss_dict.items()}))
                        loss_dict['scope'] = '{}_batch'.format(scope)
                        logger.debug('Batch: {}'.format(self.epoch * len(dl) + i_batch))
                        if self.args.mode == 'train':
                            self.summary.add_summary(self.epoch * len(dl) + i_batch, **loss_dict)
                    logger.debug('Eval_batch: {}'.format(i_batch))
                valid_metrics_dict = {'scope': scope}
                valid_metrics_dict.update({k: np.average(np.array(v)) for k, v in metrics_dict.items()})

                valid_pred_dict = {"y_true": np.concatenate(y_trues).reshape(-1),
                               "y_predscore": np.concatenate(y_predscores)}
                if self.args.embedding:
                    if not Path(f"../test/embedding").exists():
                        os.mkdir(f"../test/embedding")
                    np.save('tabnet_embedding.npy',embeds)
                    np.save('tabnet_labels.npy',embed_labels)
                return valid_metrics_dict, valid_pred_dict
                

    def aly_pred(self, scope, metric_dict, pred_dict=None):
        logger.debug('Epoch: {}'.format(self.epoch))

        if scope == 'valid':
            if self.args.lr_decay_mode in ['step', 'warmup']:
                self.lr_scheduler.step()
            elif self.args.lr_decay_mode == 'plateau':
                reduce_lr_on_plateau(self.lr_scheduler, metric_dict, 'loss')

            for param_group in self.optimizer.param_groups:
                if param_group['lr'] < self.args.min_lr:
                    param_group['lr'] = self.args.min_lr
                metric_dict.update({'lr': param_group['lr']})

            # save_weight_update_flag(model, weight_dir_dict, flag_dict, metric_dict, epoch)
            save_model_update_flag(self.model, self.optimizer, self.weight_path_dict, self.flag_dict,
                                   metric_dict, MIN_METRICS, MAX_METRICS, self.epoch)
            logger.debug('Flag dict: {}'.format(self.flag_dict))
        pred_metrics = cal_pred_metrics(pred_dict)
        metric_dict.update(pred_metrics)
        self.summary.add_summary(self.epoch, **metric_dict)
        logger.info(metric_dict)

        result_dict = dict()
        result_dict['early_stop'] = early_stop(self.flag_dict, self.epoch, self.args.early_stopping, scope)
        return result_dict

    def load_weight(self):
        logger.info("*******Restoring the model weight based on {}*******".format(self.args.restore_metric))
        restore_file = osp.join(self.weight_path_dict[self.args.restore_metric], 'model.pth.tar')
        if restore_file.endswith('.tar'):
            checkpoint = torch.load(restore_file)
            self.epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info('=> loaded checkpoint from model.path.tar')
        else:
            logger.error('==> Load fail: no checkpoint for {}'.format(restore_file))