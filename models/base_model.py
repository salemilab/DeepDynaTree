#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
base_model.py:
"""

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from xgboost import XGBClassifier, XGBRFClassifier
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

class Model():
    def __init__(self):
        self.best_classifer = None
        self.best_val = float('inf')
        self.best_stats = None
        self.n_search = 40

    def random_search_hyperparams(self, **kwargs):
        raise NotImplementedError

    def fit(self, **kwargs):
        raise NotImplementedError


class LinearSVM(Model):
    def __init__(self):
        super(LinearSVM, self).__init__()

    def random_search_hyperparams(self):
        reg_values = [0.05, 0.1, 0.15, 1e-2, 1e-3, 1e-4]
        reg = reg_values[np.random.randint(0, len(reg_values))]
        return reg

    def fit(self, x, y, cv_func, treeids, label_weight):
        for i in range(self.n_search):
            # Given a set of potential values, increase the number of iterations
            reg = self.random_search_hyperparams()

            # Create a two-layer network
            classifer = SGDClassifier(loss='hinge', penalty='l1', alpha=reg, learning_rate='optimal',
                                      class_weight=label_weight)

            # CV Train
            cv_val, monitor_metric = cv_func(classifer, x, y, treeids)

            # Save best values
            if cv_val < self.best_val:
                self.best_val = cv_val
                self.best_classifer = classifer

            # Print results
            # print(f'lr: optimal, reg: {reg}, {monitor_metric}: {cv_val}')
        print(f'best validation {monitor_metric} achieved: %f' % self.best_val)


class Sfmx(Model):
    def __init__(self, **kwargs):
        super(Sfmx, self).__init__()
        #self.n_search = 40
        self.multi_class = kwargs.get('multi_class', 'multinomial')

    def random_search_hyperparams(self):
        reg_values = [0.05, 0.1, 0.15, 1e-2, 1e-3, 1e-4]
        reg = reg_values[np.random.randint(0, len(reg_values))]
        return reg

    def fit(self, x, y, cv_func, treeids, label_weight):
        for i in range(self.n_search):
            # Given a set of potential values, increase the number of iterations
            reg = self.random_search_hyperparams()

            softmax_r = LogisticRegression(penalty='l2', C=1/reg, solver='newton-cg', multi_class=self.multi_class,
                                           max_iter=5000, class_weight=label_weight)

            # CV Train
            cv_val, monitor_metric = cv_func(softmax_r, x, y, treeids)

            # Save best values
            if cv_val < self.best_val:
                self.best_val = cv_val
                self.best_classifer = softmax_r

            # Print results
            # print(f'lr: optimal, reg: {reg}, {monitor_metric}: {cv_val}')
        print(f'best validation {monitor_metric} achieved: {self.best_val}')


class RF(Model):
    def __init__(self, **kwargs):
        super(RF, self).__init__()
        self.n_search = 20

    def random_search_hyperparams(self, **kwargs):
        hp_dict = {}
        max_depths = [5, 7, 10, 15, None]

        hp_dict['n_estimators'] = np.random.choice([50, 70, 100, 120, 140])
        hp_dict['max_depth'] = max_depths[np.random.randint(len(max_depths))]
        hp_dict['min_samples_split'] = np.random.randint(3, 15)
        hp_dict['min_samples_leaf'] = np.random.randint(2, 30)
        hp_dict['max_features'] = np.random.choice(['auto', 'sqrt', 'log2', .3, .5, .7])
        hp_dict['class_weight'] = np.random.choice(['balanced', 'balanced_subsample'])

        return hp_dict

    def fit(self, x, y, cv_func, treeids, label_weight):
        for i in range(self.n_search):
            # Given a set of potential values, increase the number of iterations
            hp_dict = self.random_search_hyperparams()

            rf = RandomForestClassifier(n_estimators=hp_dict['n_estimators'], criterion='gini',
                                        max_depth=hp_dict['max_depth'],
                                        min_samples_split=hp_dict['min_samples_split'],
                                        min_samples_leaf=hp_dict['min_samples_leaf'],
                                        class_weight=label_weight)

            # CV Train
            cv_val, monitor_metric = cv_func(rf, x, y, treeids)

            # Save best values
            if cv_val < self.best_val:
                self.best_val = cv_val
                self.best_classifer = rf

            # Print results
            # print(f'Dict: {hp_dict}, {monitor_metric}: {cv_val}')
        print(f'best validation {monitor_metric} achieved: {self.best_val}')


class XGB(Model):
    def __init__(self, **kwargs):
        super(XGB, self).__init__()
        self.n_search = 80

    def random_search_hyperparams(self):
        learning_rates = [0.001, 0.005, 0.01, 0.03, 0.05, 0.07, 0.08, 0.1, 0.15, 0.2]
        max_depths = [5, 10, 15, 20, 25, 30, 35, 40]
        lambdas = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        #scale_pos_weight = [0.5, 1, 1.5, 2]

        hp_dict = {}

        hp_dict['learning_rate'] = learning_rates[np.random.randint(len(learning_rates))]
        hp_dict['subsample'] = (-0.5) * np.random.rand() + 0.5  # (0.5, 1]
        hp_dict['lambda'] = lambdas[np.random.randint(len(lambdas))]
        hp_dict['max_depth'] = max_depths[np.random.randint(len(max_depths))]
        #hp_dict['lambda'] = lambdas[np.random.randint(len(lambdas))]
        #hp_dict['scale_pos_weight'] = scale_pos_weight[np.random.randint(len(scale_pos_weight))]
        hp_dict['objective'] = "multi:softprob"
        hp_dict['verbosity'] = 0
        hp_dict['num_class'] = 3
        return hp_dict

    def weighted_train(self, dmatrix, monitor_metric, treeids):
        if not monitor_metric:
            monitor_metric = 'mlogloss'
            
        for i in range(self.n_search):
            # Given a set of potential values, increase the number of iterations
            hp_dict = self.random_search_hyperparams()
            
            gkf = KFold(n_splits=5)
            
            cv_results = xgb.cv(dtrain=dmatrix, params=hp_dict, nfold=5, folds=gkf, metrics=monitor_metric)
        
            cv_val = cv_results['test-mlogloss-mean'].tail(1).values

            #xgb = XGBRFClassifier(objective="multi:softprob", verbosity=0, use_label_encoder=False, **hp_dict)

            # CV Train
            #cv_val, monitor_metric = cv_func(xgb, x, y, treeids)

            # Save best values
            if cv_val < self.best_val:
                self.best_val = cv_val
                self.best_classifier = hp_dict

            # Print results
            # print(f'Dict: {hp_dict}, {monitor_metric}: {cv_val}')
        print(f'best validation {monitor_metric} achieved: {self.best_val}')





