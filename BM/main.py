import scipy.io
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from BM import BoltzmannMachine


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def training(clfs):

    # train every classifier
    for c in clfs:
        c.mean_field()

    # get validation accuracy for each
    accuracy = []
    for c in clfs:
        # sig = sigmoid(np.vstack(c.val_p))
        sig = sigmoid(np.vstack(c.val_p))
        hits = np.array(sig > 0.5)
        accuracy.append(np.mean(hits))

    return accuracy


def classify(clfs, ytest):

    # test bM
    P = []
    for c in clfs:
        p = c.predict('test')
        P.append(p)

    # calculate performance accuracy
    sig = np.vstack(P).T
    pred = np.argmax(sig, axis=1)[:, None]
    hits = np.array(pred == ytest, dtype=np.int32)
    accuracy = np.mean(hits)

    fig = plt.figure()
    for j in range(10):
        i = np.random.randint(0, len(ytest), 1)
        plt.subplot(2, 5, j+1)
        plt.imshow(xtest[i].reshape(28, 28).T)
        sns.set_style("whitegrid", {'axes.grid': False})
        plt.text(x=10.5, y=-3.0, s='true: ' + str(ytest[i][0, 0]),  color='blue')
        plt.text(x=10.5, y=1.0, s='pred: ' + str(pred[i][0, 0]),  color='red')
        plt.axis('off')
    plt.show()
    fig.savefig('predictions.png')


def add_noise(data, thrs):
    r, c = data.shape
    noise = np.random.rand(r, c)
    mask = np.array([noise < thrs], dtype=np.float32)

    return np.squeeze(np.abs(data - mask))


if __name__ == '__main__':
    '''
        which exercise do you want to run?
        * 'random': classify randomly generated dataset with BM computing the exact solution
        * 'MNIST' = classify MNIST dataset with 10 different BMs using mean field estimation
    '''

    # data = 'MNIST'
    data = 'MNIST'

    if data is 'random':

        # hyper-parameters
        N = 10 # num of neurons
        M = 100 # num training samples
        T = 500 # num of sequential dynamic steps
        K = 200 # num of training iters
        eta = 1e-3

        # training data
        S = np.random.choice([1, -1], [M, N])

        # BM
        bm = BoltzmannMachine(S, K=K, T=T, eta=eta)
        # train BM
        bm.exact()

        fig = plt.figure()
        plt.plot(np.arange(K), bm.delta)
        plt.xlabel('iterations K=200')
        plt.ylabel('average change rate in w')
        plt.show()
        fig.savefig('changeW_vs_iter.png')

    elif data is 'MNIST':

        # hyper-parameters
        C = 10 # num of classes

        # read data sets
        mat = scipy.io.loadmat('mnistAll.mat')['mnist'][0][0]
        xtrain, xtest, ytrain, ytest = mat[0], mat[1], mat[2], mat[3]

        # prepare train data set
        ntrain = xtrain.shape[-1]

        xtrain = xtrain.T.reshape(ntrain, -1) / 255.
        xtrain_ = np.array(xtrain > 0.5, dtype=np.float32)
        xtrain = add_noise(xtrain_, 0.1)

        ytrain = np.squeeze(ytrain)

        # prepare val/test data sets
        ntest = xtest.shape[-1]
        split = 500

        xval = xtest.T[:split].reshape(split, -1) / 255.
        xval = np.array(xval > 0.5, dtype=np.float32)
        yval = np.squeeze(ytest[:split])

        xtest = xtest.T[split:].reshape(ntest - split, -1) / 255.
        xtest = np.array(xtest > 0.5, dtype=np.float32)
        ytest = ytest[split:]

        # # plot input and noisy input
        # plt.figure()
        # plt.subplot(2,1,1)
        # plt.imshow(xtrain_[105].reshape(28,28).T)
        # plt.subplot(2,1,2)
        # plt.imshow(xtrain[105].reshape(28,28).T)
        # sns.set_style("whitegrid", {'axes.grid' : False})
        # plt.show()

        # create classifiers
        clfs = []
        for c in range(C):
            clfs.append(BoltzmannMachine(xtrain[ytrain == c, :], xval[yval == c, :], xtest))

        # train them
        acc = training(clfs)

        # # plot mean firing rate and linear response
        # plt.figure()
        # plt.subplot(2,1,1)
        # plt.imshow(clfs[7].mi.reshape(28, 28).T)
        # plt.subplot(2,1,2)
        # plt.imshow(clfs[7].Xij)
        # plt.colorbar()
        # sns.set_style("whitegrid", {'axes.grid' : False})
        # plt.show()

        # test them
        classify(clfs, ytest)

    else:
        raise Exception('Not a valid data set. Try either "random" or "MINIST"')







