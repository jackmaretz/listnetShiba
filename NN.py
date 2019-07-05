# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

import numpy as np
import six
import pickle
import scipy
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import serializers
from tqdm import tqdm
import scipy.stats as ss
import StandardScaler
import plot_result
import predictTargets
import NNfuncs


######################################################################################
# モデル定義
class Model(chainer.Chain):
    """
    一番普通の３層MLP（回帰）
    """
    def __init__(self, n_in, n_units1, n_units2, n_out):
        super(Model, self).__init__(
            l1=L.Linear(n_in, n_units1),
            l2=L.Linear(n_units1, n_units2),
            l3=L.Linear(n_units2, n_out),
        )
        
    def __call__(self, x, t):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h = F.relu(self.l3(h2))
        self.loss = F.mean_squared_error(h, t)
        return self.loss

    def predict(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h = F.relu(self.l3(h2))
        return h.data

        
class NN(NNfuncs.NN):
    def __init__(self, resumemodelName=None):
        self.resumemodelName = resumemodelName
        self.train_loss, self.test_loss = [], []
        if self.resumemodelName is not None:
            print("load resume model!")
            self.loadModel(self.resumemodelName)

    def trainModel(self, x_train, y_train, x_test, y_test, n_epoch, batchsize):
        print("Start training and validation loop......")
        N = len(x_train)
        N_test = len(x_test)
        for epoch in six.moves.range(1, n_epoch + 1):
            print('epoch', epoch)           
            # training
            perm = np.random.permutation(N)
            sum_loss = 0
            for i in tqdm(six.moves.range(0, N, batchsize)):
                x = chainer.Variable(np.asarray(x_train[perm[i:i + batchsize]]))
                t = chainer.Variable(np.asarray(y_train[perm[i:i + batchsize]]))

                self.optimizer.update(self.model, x, t)
                sum_loss += float(self.model.loss.data) * len(t.data)

            print('train mean loss={}'.format(sum_loss / N))
            self.train_loss.append(sum_loss / N)

            perm = np.random.permutation(N_test)
            sum_loss = 0
            for j in tqdm(six.moves.range(0, N_test, batchsize)):
                x = chainer.Variable(np.asarray(x_test[perm[j:j + batchsize]]), volatile='on')
                t = chainer.Variable(np.asarray(y_test[perm[j:j + batchsize]]), volatile='on')
                loss = self.model(x, t)
                sum_loss += float(loss.data) * len(t.data)
            print('test  mean loss={}'.format(sum_loss / N_test))
            self.test_loss.append(sum_loss / N_test)

    def fit(self, fit_X, fit_y, batchsize=100, n_epoch=10, n_units1=512, n_units2=128, tv_ratio=0.95, optimizerAlgorithm="Adam", savefigName="result.png", savemodelName="MLP.model"):
        train_X, train_y, validate_X, validate_y = self.splitData(fit_X, fit_y, tv_ratio)
        print("The number of data, train:", len(train_X), "validate:", len(validate_X))                # トレーニングとテストのデータ数を表示

        if self.resumemodelName is None:
            self.initializeModel(Model, train_X, n_units1, n_units2, optimizerAlgorithm)

        self.trainModel(train_X, train_y, validate_X, validate_y, n_epoch, batchsize)

        plot_result.loss(self.train_loss, self.test_loss)
        self.saveModels(savemodelName)

