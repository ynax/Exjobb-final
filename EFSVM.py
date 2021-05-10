
'''
    Entropy-based Fuzzy Support Vector Machine for imbalanced datasets
    https://www.sciencedirect.com/science/article/pii/S0950705116303495
'''

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC


def dotKernel(X):
    return np.dot(X, X.T)


def lin_Kernel(u, v):
    return u * v.T


def rbf_Kernel(u, v, c=1):
    return np.exp(- ((u - v) * (u - v).T) / (1 * c**2))


def gaussianKernel(X, y=None, s=2):
    from scipy.spatial.distance import pdist, squareform
    'https://stats.stackexchange.com/questions/15798/how-to-calculate-a-gaussian-kernel-effectively-in-numpy'
    'If input is a matrix then output is square for with distance between each sample'
    # this is an NxD matrix, where N is number of items and D its dimensionalites

    if y is None and X.shape[0] == 1:
        temp = np.zeros((2, X.shape[1]))
        temp[0, :] = X
        temp[1, :] = y
        X = temp
    pairwise_dists = squareform(pdist(X, 'euclidean'))
    K = np.exp(-pairwise_dists ** 2 / s ** 2)  # element wise
    return K


def build_Q(X, y, KERNEL):
    '''
        min 1/2 x^T P x + q^Tx
        subject to, Gx <= h
                    Ax =b
                    lb <= x <= ub

        For EFSVMs:
                maximize: ∑_{i=1}^N α_i - 1/2 ∑_{i=1}^∑_{j=1}^N α_i α_j y_i y_j K(x_i, x_j)
                s.t.      ∑_{i=1}^N y_i α_i = 0,
                          0 ≦ α_i ≦ s_i C, ∀ i=1,2,...,N
        As it stand this cannot be solved by the standard method, which requires P to be Positive Definite!
        INSTEAD solve the minimization problem:
                minimize: - ∑_{i=1}^N α_i + 1/2 ∑_{i=1}^∑_{j=1}^N α_i α_j y_i y_j K(x_i, x_j)
                s.t. --||--

         Qij = yi*yj*K(xi,xj)
    '''
    # X = np.mat([[1, 2], [3, 4]])
    # y = np.mat([1, -1]).T  # columns vector
    N = len(y)

    P = np.ones((N, N))
    q = np.ones((N, 1))

    P = np.multiply(np.multiply(y.T, P), y)
    gaussKer = gaussianKernel(X)
    P = -np.multiply(P, gaussKer)

    P
    isPositiveDef = np.all(np.linalg.eigvals(P) > 0)
    print('P is positive definite: {}'.format(isPositiveDef))
    '''
        P is now [y1y1A1, y1y2A2;y2y1A3 y2y2A4]
    '''

    return P, q


def gaussian_kernel(x, z, sigma):
    n = x.shape[0]
    m = z.shape[0]
    xx = np.dot(np.sum(np.power(x, 2), 1).reshape(n, 1), np.ones((1, m)))
    zz = np.dot(np.sum(np.power(z, 2), 1).reshape(m, 1), np.ones((1, n)))
    return np.exp(-(xx + zz.T - 2 * np.dot(x, z.T)) / (2 * sigma ** 2))


def kernel(u, v, name=None):
    if name is None:
        return u * v.T
        # return u * v.T


def FM_L(beta, l, m):
    if beta < 0:
        print('Wrong beta, cant be less than 0')
    if beta > 1 / (m - 1):
        beta = 1 / (m - 1) / 2
        print('Wrong beta, cant be greater 1/(m-1)')

    'l = 1,2,..,m'
    return 1 - beta * (l - 1)


def changeY(y):
    '''
        Change the set of classes to -1,1
        Simple, assuming y ∈ {0,1}
    '''

    y[y == 0] = -1
    return y


class entropy:
    'minority are positives i.e +1 and majority are negatives -1'

    def __init__(self, K=2):
        # print('Entropy for KNNs with K: {}'.format(K))
        self.K = K
        pass

    def KNN_fit(self, X, y):
        '''
            Input training X,y.
            Assumes binary and majority is 0.
            Output: Matrix P with # probabilities of pos and neg for each samples
        '''
        print('Fitting KNN')
        self.KNN = NearestNeighbors()  # n_neighbors=K)
        self.KNN.fit(X)
        nearest = self.KNN.kneighbors(X, n_neighbors=self.K, return_distance=False)
        print('Done fitting KNN')
        self.n = nearest
        N = y.shape[0]
        self.N = N
        MASK_pos = (y == 1)
        self.MASK_neg = ~MASK_pos

        P = np.zeros(shape=(N, 2))
        'P[i,1] - Positives'
        'P[i,0] - Negatives'
        print('Constructing nearest neighbour probabilities')
        for i in range(N):
            pos_prob = (y[nearest[i]] == 1).sum() / self.K  # y[nearest[i]].sum() / self.K
            P[i, 1] = pos_prob
            P[i, 0] = 1 - pos_prob
        self.P = P
        return self.P
#

    @ property
    def entropy(self):
        '''
            entropy of sample i
            probabilites are based on K-nearest neighbours and are calculated as:

            prob_pos  = (# of positives of k) / k
            prbo_neg = (# of negatives of k) / k

        '''
        print('Doing Entropy vector from probability matrix')
        prob_pos = self.P[:, 1]
        prob_neg = self.P[:, 0]
        self.H = -prob_pos * np.log(prob_pos) + prob_neg * np.log(prob_neg)
        'Values that have probability zero will be nan through np.log(0)'
        self.H[np.isnan(self.H)] = 0

    def algorithm_1(self, m=10):
        '''
            Algorithm for separating the negative samples.
            m - subset of negative class
        '''
        self.S = {}
        H_max = max(self.H)
        H_min = min(self.H)
        H_diff = H_max - H_min
        print('Creating fuzzy subsets')
        for l in range(1, m + 1):
            upper_bound = H_min + l / m * H_diff
            lower_bound = H_min + (l - 1) / m * H_diff

            mask_lower = lower_bound <= self.H
            mask_upper = self.H <= upper_bound
            self.S[l] = mask_lower & mask_upper & self.MASK_neg


class EFSVM:
    'Have to be carefull when specifying Ys, positives have to be 1s and negatives have to be -1 '

    def __init__(self,
                 kernel='rbf',
                 C=1,
                 K=5,
                 m=10,
                 beta=0.05,
                 gamma=1,
                 class_weight='balanced',
                 rng=None):
        self.KERNEL = kernel
        self.Entropy = entropy(K)
        self.m = m
        self.C = C
        self.gamma = gamma
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.randint(10**6)
#         Environment.LOGGER.debug('ENSVM random state: {}'.format(self.rng))
        self.class_weight = class_weight
        if beta is not None:
            self.beta = beta
        else:
            self.beta = 1 / (m - 1) / 2

    def fit(self, X, y):
        '''
            ****************************************************************
                                Construct Entropy
            ****************************************************************
        '''
        self.X = np.array(X)
        self.N = self.X.shape[0]
        'Fit KNN'
        self.Entropy.KNN_fit(X, y)
        'Get entropy for all fitted..'
        self.Entropy.entropy
        self.Entropy.algorithm_1(self.m)
        'Assign fuzzy mebers to all subsets'

        # MASK_pos = ~self.Entropy.MASK_neg

        '''
            ****************************************************************
                                Construct Fuzzy Membership
            ****************************************************************
        '''
        FUZZY_membership = np.zeros(len(y))
        FUZZY_membership[~self.Entropy.MASK_neg] = 1

        print('Getting fuzzy memerships')
        for l in range(1, self.m + 1):
            # print('Fuzzy membership subset: {:.2f}'.format(l))
            # FM_l = #FM_L(self.beta, l, self.m)
            # print(FM_l)
            mask = self.Entropy.S[l]
            FUZZY_membership[mask] = FM_L(self.beta, l, self.m)

        self.x = FUZZY_membership  # * self.C

        svm = SVC(C=1 * self.C, kernel=self.KERNEL, gamma=self.gamma, random_state=self.rng, class_weight=self.class_weight)
        svm.fit(self.X, y, sample_weight=self.x)
        self.predict = svm.predict
