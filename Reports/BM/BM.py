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
        self.delta_w = []
        self.val_p = 0
        self.mi = 0
        self.invXij = 0

    def mean_field(self):

        print('training....')

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

    def monte_carlo(self, burn_in, alpha):

        N = self.train.shape[1]

        # clamped statistics
        Si_c = np.mean(self.train, axis=0)
        Sij_c = np.dot(self.train.T, self.train) / self.train.shape[0]

        # initialize states
        states = np.random.choice([-1, 1], N)

        Si_f = np.zeros(states.shape)
        Sij_f = np.zeros((len(states), len(states)))

        # K learning steps
        for k in range(self.K):

            # sequential stochastic dynamics (T=500)
            for t in range(self.T + burn_in):

                # h(s)
                h_s = np.dot(self.w, states) + self.theta
                # flip operator
                flip = 0.5 * (1 + np.tanh(-states * h_s))

                # gibbs sampling
                flipped = flip > np.random.rand(N)
                states[flipped] *= -1

                # burn-in period
                if t > burn_in:

                    # free statistics (sum)
                    Si_f += states
                    Sij_f += np.dot(states.T, states)

            # free statistics (mean)
            Si_f /= self.T
            Sij_f /= self.T

            # compute gradients
            delta_Si = Si_c - Si_f
            delta_Sij = Sij_c - Sij_f

            # add momentum according to alpha
            if k == 0:
                momentum_w = np.zeros(delta_Sij.shape)
                momentum_theta = np.zeros(delta_Si.shape)
            else:
                momentum_w = alpha * w_
                momentum_theta = alpha * theta_

            # update w and theta
            self.w += self.eta * delta_Sij + momentum_w
            self.theta += self.eta * delta_Si + momentum_theta

            # store gradients to calculate momentum in the next iter
            w_ = self.eta * delta_Sij
            theta_ = self.eta * delta_Si

            # store the change at every update
            self.delta_w.append(self.eta * np.mean(abs(delta_Sij)) + self.eta * np.mean(abs(delta_Si)))

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





