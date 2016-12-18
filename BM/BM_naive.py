import numpy as np

N = 10
M = 50
T = 500
K = 200

# training data
S = np.random.choice([1, -1], [M, N])

# possible states between 2 neurons
Si = [1, 1, -1, -1]
Sj = [-1, 1, -1, 1]

# clamped statistics
Si_c = np.sum(S, axis=1) / float(M)
Sij_c = np.array([np.sum(S[:, i] * S[:, j]) for i in range(N) for j in range(i+1, N)])

# initialize w
w = np.random.nor

for k in range(K):
    for t in range(T):

         # calculate p(s)
        P = np.exp(w * Si * Sj + w * Si + w * Sj)
        # calculate Z
        Z = sum(P)

        # analytical solution
        Si_f = sum(Si  * P * (1. / Z)))
        Sij_f = sum(Sj * S2 * P * (1. / Z))

