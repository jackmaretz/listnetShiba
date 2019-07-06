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
from sklearn.preprocessing import StandardScaler
import plot_result
import NNfuncs


######################################################################################
# Define model
class Model(chainer.Chain):
    """
    ListNet - Listwise comparison of ranking.
    The original paper:
        http://research.microsoft.com/en-us/people/tyliu/listnet.pdf

    NOTICE:
        The top-k probability is not written.
        This is listwise approach with neuralnets, 
        comparing two arrays by Jensen-Shannon divergence.

    """
    def __init__(self, n_in, n_units1, n_units2, n_out):
        super(Model, self).__init__(
            l1=L.Linear(n_in, n_units1),
            l2=L.Linear(n_units1, n_units2),
            l3=L.Linear(n_units2, n_out),
        )


    def __call__(self, x, t):
        h1 = self.l1(x)
        y = self.l3(F.relu(self.l2(F.relu(self.l1(x)))))
        # self.loss = self.listwise_cost(y_data, t_data)
        self.loss = self.jsd(t, y)
        return self.loss


    def predict(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h = F.relu(self.l3(h2))
        return h.data

    def kld(self, vec_true, vec_compare):
        ind = vec_true.data * vec_compare.data > 0
        ind_var = chainer.Variable(ind)
        include_nan = vec_true * F.log(vec_true / vec_compare)
        z = chainer.Variable(np.zeros((len(ind), 1), dtype=np.float32))
        # return np.nansum(vec_true * np.log(vec_true / vec_compare))
        return F.sum(F.where(ind_var, include_nan, z))

    def jsd(self, vec_true, vec_compare):
        vec_mean = 0.5 * (vec_true + vec_compare)
        return 0.5 * self.kld(vec_true, vec_mean) + 0.5 * self.kld(vec_compare, vec_mean)

    def topkprob(self, vec, k=5):
        vec_sort = np.sort(vec)[-1::-1]
        topk = vec_sort[:k]
        ary = np.arange(k)
        return np.prod([np.exp(topk[i]) / np.sum(np.exp(topk[i:])) for i in ary])

    def listwise_cost(self, list_ans, list_pred):
        return - np.sum(self.topkprob(list_ans) * np.log(self.topkprob(list_pred)))


class ListNet(NNfuncs.NN):
    """
    ListNet training function.
    Usage (Initialize):
        RankModel = ListNet()

    Usage (Traininng):
        Model.fit(X, y)

    With options:
        Model.fit(X, y, batchsize=100, n_epoch=200, n_units1=512, n_units2=128, tv_ratio=0.95, optimizerAlgorithm="Adam", savefigName="result.pdf", savemodelName="ListNet.model"):

    """
    def __init__(self, resumemodelName=None):
        self.resumemodelName = resumemodelName
        self.train_loss, self.test_loss = [], []
        self.train_acc, self.test_acc = [], []
        if resumemodelName is not None:
            print("load resume model!")
            self.loadModel(resumemodelName)

    # リストネットの誤差関数
    def ndcg(self, y_true, y_score, k=3):
        y_true = y_true.ravel()
        y_score = y_score.ravel()
        y_true_sorted = sorted(y_true, reverse=True)
        ideal_dcg = 0
        for i in range(k):
            ideal_dcg += (2 ** y_true_sorted[i] - 1.) / np.log2(i + 2)
        dcg = 0
        argsort_indices = np.argsort(y_score)[::-1]
        for i in range(k):
            dcg += (2 ** y_true[argsort_indices[i]] - 1.) / np.log2(i + 2)
        ndcg = dcg / ideal_dcg
        return ndcg

    # リストネットのトレーニング専用関数
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
                x = chainer.Variable(np.asarray(x_test[perm[j:j + batchsize]]))
                t = chainer.Variable(np.asarray(y_test[perm[j:j + batchsize]]))
                loss = self.model(x, t)
                sum_loss += float(loss.data) * len(t.data)
            print('test  mean loss={}'.format(sum_loss / N_test))
            self.test_loss.append(sum_loss / N_test)

            train_score = self.model.predict(chainer.Variable(x_train))
            test_score = self.model.predict(chainer.Variable(x_test))
            train_ndcg = self.ndcg(y_train, train_score)
            test_ndcg = self.ndcg(y_test, test_score)
            self.train_acc.append(train_ndcg)
            self.test_acc.append(test_ndcg)
            print("epoch: {0}".format(epoch + 1))
            print("NDCG@100 | train: {0}, test: {1}".format(train_ndcg, test_ndcg))


    def fit(self, fit_X, fit_y, batchsize=1, n_epoch=10, n_units1=256, n_units2=64, tv_ratio=0.70, optimizerAlgorithm="Adam", savefigName="resultlist.pdf", savemodelName="ListNet.model"):
        train_X, train_y, validate_X, validate_y = self.splitData(fit_X, fit_y, tv_ratio)
        print("The number of data, train:", len(train_X), "validate:", len(validate_X))                # トレーニングとテストのデータ数を表示

        if self.resumemodelName is None:
            self.initializeModel(Model, train_X, n_units1, n_units2, optimizerAlgorithm)
        
        self.trainModel(train_X, train_y, validate_X, validate_y, n_epoch, batchsize)

        plot_result.acc(self.train_acc, self.test_acc)
        plot_result.loss(self.train_loss, self.test_loss)
        self.saveModels(savemodelName)

    def test(self, fit_X, fit_y, batchsize=100, n_epoch=1, tv_ratio=0.95, optimizerAlgorithm="Adam"):
        """
        usage:
        Model = ListNet(MODELNAME)
        Model.test(fit_X, fit_y)
        """
        
        train_X, train_y, validate_X, validate_y = self.splitData(fit_X, fit_y, tv_ratio)
        print("The number of data, train:", len(train_X), "validate:", len(validate_X))                # トレーニングとテストのデータ数を表示        
        self.trainModel(train_X, train_y, validate_X, validate_y, n_epoch, batchsize)


################################################################################################
## end of file ##
################################################################################################



