import numpy as np
from tqdm import tqdm


class BoltzmannMachine:

    def __init__(self, train, val=None, test=None, eta=1e-3, K=200, T=500):
        self.train = train
        self.val = val
        self.test = test
        self.eta = eta
        self.K = K
        self.T = T
        self.theta = np.random.rand(train.shape[1])
        self.w = np.random.rand(train.shape[1], train.shape[1])
        self.F = 0
        self.delta = []
        self.val_p = 0
        self.mi = 0
        self.invXij = 0

    def mean_field(self):

        # clamped statistics
        Si_c = np.mean(self.train, axis=0)
        Sij_c = np.dot(self.train.T, self.train) / self.train.shape[0]

        # mean field equations (with no hidden neurons)
        self.mi = Si_c # mean firing rate

        # linear response
        Xij = Sij_c - np.dot(self.mi.T, self.mi)
        self.invXij = np.linalg.inv(Xij)

        # compute w and theta
        self.w = -self.invXij
        # (subtract additional constraint on the weights)
        np.fill_diagonal(self.w, 1. / (1 - self.mi ** 2) - np.diag(self.invXij))
        self.theta = np.arctanh(self.mi) - np.dot(self.w, self.mi)

        # compute the mean field free energy F
        S = np.dot((1 + self.mi), np.log(0.5 * (1 + self.mi))) + np.dot((1 - self.mi), np.log(0.5 * (1 - self.mi)))
        self.F = - 0.5 * np.dot(np.dot(self.mi, self.w), self.mi) - np.dot(self.theta, self.mi) + 0.5 * S

        # validate results
        self.val_p = np.vstack(self.predict('val'))

        print('training terminated')

    def exact(self):

        N = self.train.shape[1]

        # clamped statistics
        Si_c = np.mean(self.train, axis=0)
        Sij_c = np.dot(self.train.T, self.train) / self.train.shape[0]

        # possible states between 2 neurons
        Si = np.array([1, 1, -1, -1])
        Sj = np.array([-1, 1, -1, 1])

        # K learning steps
        for k in range(self.K):
            # sequential stochastic updates of w (T=500)
            diff = 0
            for _ in range(self.T):

                # choose two random neurons i and j, where i != j
                i, j = np.random.randint(0, N, size=2)
                if i == j:
                    while i == j:
                        j = np.random.randint(0, N, size=1)

                # compute p(s)
                P = np.exp(self.w[i, j] * Si * Sj + self.w[i, j] * Si + self.w[i, j] * Sj)
                # compute Z
                Z = np.sum(P)

                # use p(s) and Z to compute free statistics
                Si_f = np.sum(Si * P * (1. / Z))
                Sij_f = np.sum(Si * Sj * P * (1. / Z))

                # compute gradients
                delta_Si = Si_c[i] - Si_f
                delta_Sij = Sij_c[i, j] - Sij_f

                # update w and theta only for neurons i and j
                self.w[i, j] += self.eta * delta_Sij
                self.theta[i] += self.eta * delta_Si

                # store the change at every update
                diff += abs(self.eta * delta_Sij)

            self.delta.append(abs(diff) / self.T)

    def predict(self, mode):
        print('validating...')

        if mode is 'val':
            assert self.val is not None
            data = self.val
        else:
            assert self.test is not None
            data = self.test

        # compute p(s)for every sample
        P = []
        Z = np.exp(-self.F)
        for m in tqdm(range(len(data))):
            p = np.exp(0.5 * np.dot(np.dot(data[m, :], self.w), data[m, :]) + np.dot(self.theta, data[m, :])) / Z
            P.append(p)

        return P





