import numpy as np
from tqdm import tqdm

def energy(w,s,b):
    return np.dot(s, np.dot(w, s.T))+ np.dot(b.T, s.T)
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

    def exact(self, burn_in):

        N = self.train.shape[1]

        # clamped statistics
        Si_c = np.mean(self.train, axis=0)
        Sij_c = np.dot(self.train.T, self.train) / self.train.shape[0]

        # initialize states
        states = np.random.choice([-1, 1], N)

        Si_f = np.zeros(states.shape)
        Sij_f = np.zeros((len(states), len(states)))


        # EXTRA_CODE
        E=energy(self.w,states,self.theta)
        Delta_E=[]
        for i,spin in enumerate(states):
            temp_states=np.array(states)
            temp_states[i]*=-1
            Delta_E.append(energy(self.w,temp_states,self.theta)-E)
        beta=1/max(Delta_E)
        self.betas=[]
        # EXTRA_CODE


        # K learning steps
        self.E_std=[]
        for k in range(self.K):
            beta*=1.05
            self.betas.append(beta)
            # sequential stochastic dynamics (T=500)
            E_all=[]
            for t in range(self.T + burn_in):
                i = np.random.randint(N, size=1)
                h_s = np.dot(self.w*beta, states) + self.theta
                flip = 0.5 * (1 + np.tanh(-states * h_s))
                if np.random.uniform(low=0.0, high=1.0)<flip[i]:
                    states[i]*=-1
                if t > burn_in:
                    Si_f += states
                    Sij_f += np.dot(states.T, states)
                    E_all.append(energy(self.w,states,self.theta))
            self.E_std.append(np.std(E_all))
            Si_f /= self.T
            Sij_f /= self.T

            # states = np.stack(states)

            # # compute p(s)
            # E = np.diagonal(np.dot(states, np.dot(self.w, states.T))) + np.dot(self.theta.T, states.T)
            # P = np.exp(0.5 * E)
            #
            # # compute Z
            # Z = np.sum(P)
            #
            # # use p(s) and Z to compute free statistics
            # # todo: find out how to calculate free stats
            # Si_f = np.sum(P* (1. / Z) * states
            # Sij_f = np.sum(np.dot(states.T, states) * P * (1. / Z))

            # Si_f = np.mean(states_, axis=0)
            # Sij_f = (1. / self.T) * np.dot(states_.T, states_)

            # compute gradients
            delta_Si = Si_c - Si_f
            delta_Sij = Sij_c - Sij_f

            # update w and theta only for neurons i and j
            self.w += self.eta * delta_Sij
            self.theta += self.eta * delta_Si

            # store the change at every update
            self.delta_w.append(self.eta * np.sum(abs(delta_Sij)) + self.eta * np.sum(abs(delta_Si)))

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





