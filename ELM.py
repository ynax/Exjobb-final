from hpelm import ELM as elm
from numpy import array, zeros
from pandas import DataFrame

'''
    Make something to get the parameters maybe.. ?
'''


class ELM:
    # 'tanh',
    # 'sigm',
    # 'rbf_l1',
    # 'rbf_l2',
    # 'rbf_linf'
    def __init__(self,
                 m=100,
                 kernel='tanh',
                 w=1,
                 outputs=2,
                 type='wc',
                 batch=1000,
                 **kwargs):
        self.w = w
        self.kernel = kernel
        self.m = m
        self.batch = batch
        self.type = type
        self.outputs = outputs

    def fit(self, X, Y):
        X = array(X)
        Y = array(Y)
        self.ELM = elm(
            inputs=X.shape[1],
            outputs=self.outputs
        )
        self.ELM.add_neurons(self.m, self.kernel)
        Y = self.make_Ys(Y)
        self.ELM.train(
            X,
            Y,
            self.type,
            w=[self.w, 1],
            batch=self.batch
        )

    def predict(self, X):
        X = array(X)
        Ypred = self.ELM.predict(X)
        Ypred = self.reg_2_class(Ypred)
        return DataFrame(Ypred)

    def reg_2_class(self, reg):
        mask = reg[:, 0] >= reg[:, 1]
        ypreds = zeros((reg.shape[0], 1))
        ypreds[mask] = 1
        return ypreds

    def make_Ys(self, Y):
        temp = zeros((Y.shape[0], 2))
        df2 = DataFrame(temp, columns=['Fraud', 'Genuine'])
        # df2.loc[target,'Fraud'] = 1
        # df2.loc[~target,'Genuine'] = 1
        df2.loc[Y == 1, 'Fraud'] = 1
        df2.loc[Y == 0, 'Genuine'] = 1
        T = df2.values
        return T
