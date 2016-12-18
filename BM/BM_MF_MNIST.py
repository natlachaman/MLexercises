import numpy as np
import math
from sympy import Symbol, nsolve, tanh
import matplotlib.pylab as plt
import seaborn

# 4 possible states patterns between 2 neurons
S1 = np.array([1, 1, -1, -1])
S2 = np.array([1, -1, 1, -1])

# vector of w values
W = np.linspace(0, 1, 10)

# symbolic m
symm = Symbol('m')

m, m_true = [], []
X, X_true = [], []

for w in W:

    # approximate m_i (mirrored values)
    f = tanh(w * symm + w) - symm
    m_ = nsolve(f, symm, 0.5)
    m.append(m_)

    # approximate X_i,j
    X.append( (1. / 1 - (m_ ** 2)) - w  )

    # calculate p(s)
    P = np.exp(w * S1 * S2 + w * S1 + w * S2)
    # calculate Z
    Z = sum(P)

    # analytical solution
    m_true.append(sum(S1  * P * (1. / Z)))
    X_true.append(sum(S1 * S2 * P * (1. / Z) - m_**2) )


fig1 = plt.figure(0)
plt.plot(W, m, 'r-', label='MF approx')
plt.plot(W, m_true,'k', label='Analytical')
plt.xlabel('w')
plt.ylabel('m')
plt.title('Mean Firing rate')
plt.legend(loc=4)

# fig1.savefig('firing_mean.png')

fig2 = plt.figure(1)
plt.plot(W, X, 'r-', label='MF approx')
plt.plot(W, X_true,'k', label='Analytical')
plt.xlabel('w')
plt.ylabel('X')
plt.title('Linear Response')
plt.legend()

# fig2.savefig('linear_response.png')

plt.show()



