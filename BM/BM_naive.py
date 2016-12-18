import numpy as np
import matplotlib.pylab as plt
import seaborn

# hyperparameters
N = 10
M = 100
T = 500
K = 200

# training data
S = np.random.choice([1, -1], [M, N])

# possible states between 2 neurons
Si = np.array([1, 1, -1, -1])
Sj = np.array([-1, 1, -1, 1])

# clamped statistics
Si_c = np.sum(S, axis=1) / float(M)
Sij_c = np.array([np.sum(S[:, i] * S[:, j]) for i in range(N) for j in range(N)]) / float(M)
Sij_c = np.reshape(Sij_c, (N, N))

# initialize w, eta and theta (bias term)
w = np.random.rand(N, N)
eta = 1e-3
theta = np.random.rand(N)
change = np.zeros([K])

# plotting
fig = plt.figure()


# BM learns for K steps
for k in range(K):
    # sequential stochastic updates of w (T=500)
    diff = 0
    for _ in range(T):

        # choose two random neurons i and j, where i != j
        i, j = np.random.randint(0, N, size=2)
        if i == j:
            while i == j:
                j = np.random.randint(0, N, size=1)

        # compute p(s)
        P = np.exp(w[i, j] * Si * Sj + w[i, j] * Si + w[i, j] * Sj)
        # compute Z
        Z = sum(P)

        # use p(p) and Z to compute free statistics
        Si_f = sum(Si * P * (1. / Z))
        Sij_f = sum(Si * Sj * P * (1. / Z))

        # compute gradients
        delta_Si = Si_c[i] - Si_f
        delta_Sij = Sij_c[i, j] - Sij_f

        # update w and theta only for neurons i and j
        w[i, j] += eta * delta_Sij
        theta[i] += eta * delta_Si

        # store the change at every update
        diff += abs(delta_Sij)

    change[k] = abs(diff) / T

plt.plot(np.arange(K), change)
plt.xlabel('iterations K=200')
plt.ylabel('average size of change in w')
plt.show()
fig.savefig('changeW_vs_iter.png')