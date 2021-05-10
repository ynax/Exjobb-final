
'''
    RUTSVM-CIL
    Reduced Universum Twin Support Vector Machine for Class Imbalanced Learning


    Two Options: Linear or Non-Linear (non-linear is with kernel)

    Create universum from: https://dl.acm.org/doi/pdf/10.1145/1143844.1143971

    Algorithm layout: (https://www.sciencedirect.com/science/article/pii/S0031320319304510?casa_token=B2oRQ9ewR4oAAAAA:wTEAulbim5XG97hCO8ty37DlLSvW3D9hrEJVjjDpENvqTd-qSEafP4FYDvogsEXdg3NLVpkkcw#bib0016)

        Input: X1 ∈ ℝ^(r,n)  , X2 ∈ ℝ^(s,n)  and U ∈ ℝ^(d,n), d = s-r and g = ⌈ r/2 ⌉ (ceil(r/2))
        Output: weight and bias vectors w_i, b_i, i=1,2 (for class "1" and class "2")

        Procedure:

            Step 1: Construct matrices X_2* ∈ ℝ^(r,n)  and U* ∈ ℝ^(g,n) by using randomly
            selected samples from negative class and Universum.

            Step 2: Set constraints for optimization problem of positive class hyperplane using
            X_2*  and U*. And for negative class X_1 ∈ ℝ^(r,n) with universum U ∈ ℝ^(d,n) treated
            as beloning to positive class

            Step 3: Solve the QPPs in the dual form and obtain Lagrangian multipliers α_1, μ_2
                and α_2, μ_2.

            Step 4: Calculate w_i, b_i, i=1,2 using the Lagrangian Multipliers

            Step 5: Return w_i, b_i, i=1,2 for the construction of the hyperplanes of the two classes.


        Predictions are made as: y_i = min_i |K(x^t,D^t)w_i+b_i| for i ∈ {1,2}


        Select Universum: "Universum data are generated from training data via randomly averageing paris of samples from different classes."
        From 'Least squares twin support vector machine with universum data for classification.' Int. J. Syst. Sci., 47 (15) (2016), pp. 3637-3645
'''

import numpy as np
from pandas import DataFrame
from qpsolvers import solve_qp


def Un(class1, class2, N, minority_bias):
    '''
        Assumes class2 is minority class
        input:  class1 and class2 ∈ ℝ^n to generate Universum from.
                N, number of universum samples.
        Output: U, universum samples shape N ✖ n_fearures
    '''
    index1 = np.random.choice(class1.shape[0], N)
    index2 = np.random.choice(class2.shape[0], N)
    # if len(index2) > len(index1):
    # 'To make the universum more inclined to be where minorities are.'
    # minority_weight = 1 / minority_weight

    U = (class1[index1] + class2[index2] * minority_bias) / 2

    return U


def getU(X, y, N=100, minority_bias=1):
    X = np.array(X)
    y = np.array(y)

    labels = list(set(y))

    'Assumes binary classes'
    mask = y == labels[0]
    if mask.sum() < (~mask).sum():
        'Make sure that class2 is minority class'
        mask = ~mask
    U = Un(X[mask], X[~mask], N, minority_bias)
    return U


def getStars(X, N):
    'N number of random points'
    indices = np.random.choice(X.shape[0], N, replace=False)
    return X[indices]


class KERNELs:
    def RBF(u, v, sig=1, axis=1, **kwargs):
        return np.exp(- np.linalg.norm(u - v, axis=axis)**2 / (2 * sig**2))

    def DOT(u, v, **kwargs):
        return np.dot(u, v.T)

    def __init__(self, kernel_name='RBf', kernel_params=1):
        KERNELS = dict(
            RBF=KERNELs.RBF,
            DOT=KERNELs.DOT
        )

        self.kernel = KERNELS[kernel_name]
        self.kernel_params = kernel_params

    def KERNEL(self, u, v):  # , kernel_name='RBF', kernel_params=dict(sig=10)):
        '''
            Have to be numpy arrays nothing else!

            shape is:
            u.shape =  n_samples1, n_dimensions
            v.shape =  n_dimensions, n_samples2

            kernel kernel multiplication is such that if two mats are input then the output is the the kernels for each element in both against all
            samples in the other.

            Output should be a row vector.
            Where rows represent the samples (row i) in U and columns represent the samples (column j) in V

            Output:
                    Matrix of size (n_samples1, n_samples2)

        '''

        u_samples, u_dims = u.shape
        v_dims, v_samples = v.shape
        v = v.T
        if u_samples == 1 or v_samples == 1:
            s = self.kernel(u, v, self.kernel_params, axis=1)
            return s.reshape(u_samples, v_samples)
        else:
            if not u_dims == v_dims:
                raise ValueError('Dimensions have to be equal not: {} and {}'.format(u_dims, v_dims))
            u = u.reshape(u_samples, 1, u_dims)
            return self.kernel(u, v, self.kernel_params, axis=2)


class RUTSVM:

    def __init__(self,
                 kernel_name='RBF',
                 kernel_params=1,
                 C1=1,
                 C2=1,
                 Cu=1,
                 sigma=10**-1,
                 epsi=10**-2,
                 dmax=500,
                 minority_bias='None'):
        self.C1 = C1
        self.C2 = C2
        self.Cu = Cu
        self.sigma = sigma
        self.epsi = epsi
        self.KERNEL = KERNELs(kernel_name, kernel_params)
        self.dmax = dmax
        self.minority_bias = minority_bias

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y, dtype=int)
        y[y == 0] = -1
        self.labels = list(set(y))

        'Assuming binary '
        mask = y == self.labels[0]

        'Make sure that class 2 is the negative class (i.e majority)'
        if mask.sum() > (~mask).sum():
            mask = ~mask

        self.X1 = X1 = X[mask]
        self.X2 = X2 = X[~mask]

        n_samples1, _ = X1.shape
        n_samples2, _ = X2.shape

        if self.minority_bias == 'None':
            self.minority_bias = 1
        # raise ValueError("minority_bias can't be less than or equal to 0, your bias: {}".format(minority_bias)) if minority_bias <= 0 else None
        r = n_samples1
        s = n_samples2

        d = min(s - r, self.dmax)
        g = int(np.ceil(r / 2))

        U = Un(X1, X2, d, self.minority_bias)

        'Step 1'
        X2_star = getStars(X2, r)
        U_star = getStars(U, g)

        e1 = np.ones(shape=(n_samples1, 1))
        e2 = np.ones(shape=(n_samples2, 1))
        eg = np.ones(shape=(g, 1))
        ed = np.ones(shape=(d, 1))

        '''
        With Kernel!

        max_{α_1,μ_1}    e_1^t α_1 - 1/2 (α_1 F* - μ_1^t P*)(E^t E)^-(F*^t α_1 - P*^t μ_1) + (ϵ-1)e_g^t μ_1
        subject to,     0≤α_1≦C_1, 0≤μ_1≦C_u

        AND

        max_{α_2,μ_2}    e_1^t α_2 - 1/2 (α_2 E - μ_2^t P)(F^t F)^-(E^t α_2 - P^t μ_2) + (1-ϵ)e_d^t μ_2
        subject to,     0≤α_1≦C_2, 0≤μ_2≦C_u

        K(x^t,D^t) = (k(x,x_1),k(x,x_2),...,k(x,x_(2r)))

        In UTSVM, the universum data points satisfy the constraint of lying in between an insensitive tube.
        # '''
        self.D = D = np.append(X1, X2_star, 0)
        self.E = E = np.append(self.KERNEL.KERNEL(X1, D.T), e1, 1)
        self.ETE = ETE = E.T @ E
        self.invETE = invETE = np.linalg.inv(ETE + self.sigma * np.eye(2 * r + 1))

        F_star = np.append(self.KERNEL.KERNEL(X2_star, D.T), e1, 1)
        self.F = F = np.append(self.KERNEL.KERNEL(X2, D.T), e2, 1)

        self.FTF = FTF = F.T @ F
        self.invFTF = invFTF = np.linalg.inv(FTF + self.sigma * np.eye(2 * r + 1))
        del F

        P_star = np.append(self.KERNEL.KERNEL(U_star, D.T), eg, 1)
        self.P = P = np.append(self.KERNEL.KERNEL(U, D.T), ed, 1)

        self.B1 = B = (np.append(F_star, P_star, 0) @ invETE @ np.append(F_star, P_star, 0).T) + np.eye(r + g) * self.sigma
        self.q1 = q = np.append(e1, (self.epsi - 1) * eg, 0)

        lb = np.zeros((r + g, ))
        lb2 = np.zeros((r + d, ))
        ub1 = np.append(np.ones((r, )) * self.C1, np.ones((g, )) * self.Cu, 0)
        ub2 = np.append(np.ones((r, )) * self.C2, np.ones((d, )) * self.Cu, 0)

        '''
                # Solve first opt problem: min 1/2 x^T B x - q^T x
            x = SOL
            alpha1 = x[0:r]
            mu1 = x[r:]
                # calculate w1 and b1
            temp = -np.linalg.inv(EE + self.sig*np.eye(2*r+1)) @ (F_star^T*alpha1 - P_star^T*mu1)
            self.w1 = temp[0:-1]
            self.b1 = temp[-1]
            '''
        print('Solving first QPP. Size: {}'.format(B.shape))
        # np.linalg.cholesky(B)
        # print('it is ')
        sol = solve_qp(P=B, q=-q.reshape((-1,)), lb=lb, ub=ub1)
        print('Done with first QPP.')
        self.alpha1 = sol[0:r]
        self.mu1 = sol[r:]
        self.T = temp = -np.linalg.inv(ETE + self.sigma * np.eye(2 * r + 1)) @ (F_star.T @ self.alpha1 - P_star.T @ self.mu1)
        self.w1 = temp[0:-1].reshape(-1, 1)
        self.b1 = temp[-1]

        'Second optimizaton problem'
        self.B = B = np.append(E, P, 0) @ invFTF @ np.append(E, P, 0).T + np.eye(r + d) * self.sigma
        self.q = q = np.append(e1, (1 - self.epsi) * ed, 0)
        print('Solving second QPP. Size: {}'.format(B.shape))
        sol = solve_qp(P=B, q=-q.reshape((-1,)), lb=lb2, ub=ub2)
        print('Done with second QPP')
        self.alpha2 = sol[0:r]
        self.mu2 = sol[r:]
        temp = -np.linalg.inv(FTF + self.sigma * np.eye(2 * r + 1)) @ (E.T @ self.alpha2 - P.T @ self.mu2)
        self.w2 = temp[0:-1].reshape(-1, 1)
        self.b2 = temp[-1]

        '*****************************************************************************************************'

        '''
            Solve second opt problem: min 1/2 x^T B x - q^T x
        x = SOL
        alpha2 = x[0:r]
        mu2 = x[r:]
        temp = np.linalg.inv(FF+ self.sig*np.eye(2*r+1)) @ (E^T*alpha2 + P^T*mu2)
        self.w2 = temp[0:-1]
        self.b2 = temp[-1]
        '''

        pass

    def predict(self, X):
        X = np.array(X)
        self.K = K = self.KERNEL.KERNEL(X, self.D.T)
        self.v1 = v1 = abs(K @ self.w1 + self.b1)
        self.v2 = v2 = abs(K @ self.w2 + self.b2)

        mask = v1 < v2
        predicted = np.ones((X.shape[0], 1))
        predicted[mask] = self.labels[0]
        predicted[~mask] = self.labels[1]
        predicted = DataFrame(predicted)
        return predicted
