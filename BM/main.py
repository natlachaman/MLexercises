import scipy.io
import numpy as np
import multiprocessing as mp

import matplotlib.pyplot as plt
import seaborn

from BM import BoltzmannMachine


def training(clfs, C):

    # parallelize training
    processes = [mp.Process(target=clfs[c].mean_field, args=()) for c in range(C)]

    # start processes
    for p in processes:
        p.start()

    # join BMs' outputs
    for p in processes:
        p.join()

    # get validation accuracy for every clf
    accuracy = [c.queue.get() for c in clfs]

    # todo: plot acc
    return accuracy


def classify(clfs, ytest):

    # test bM
    P = []
    for c in clfs:
        conf = c.predict('test')
        confs.append(conf)

    # todo: add sigmoid
    confs = np.vstack(confs).T
    cl = np.argmax(confs, axis=1)

    hits = np.array(cl == ytest, dtype=np.int32)
    accuracy = np.mean(hits)

    # todo: add classification plots
    print 'Accuracy on test set: {}'.format(accuracy)

def add_noise(data, thrs):

    noise = np.random.rand(data)


if __name__ == '__main__':
    '''
        which exercise do you want to run?
        * 'random': classify randomly generated dataset with BM computing the exact solution
        * 'MNIST' = classify MNIST dataset with 10 different BMs using mean field estimation
    '''

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

        # train and test data sets
        mat = scipy.io.loadmat('mnistAll.mat')['mnist'][0][0]
        xtrain, xtest, ytrain, ytest = mat[0], mat[1], mat[2], mat[3]

        # train data reshape
        ntrain = xtrain.shape[-1]
        xtrain = xtrain.T.reshape(ntrain, -1)
        ytrain = np.squeeze(ytrain)

        # split set into val and test set
        ntest = xtest.shape[-1]
        xval = xtest.T[:1000].reshape(1000, -1)
        yval = np.squeeze(ytest[:1000])
        xtest = xtest.T[1000:].reshape(ntest - 1000, -1)
        ytest = ytest[1000:]

        # create classifiers
        clfs = []
        for c in range(C):
            clfs.append(BoltzmannMachine(xtrain[ytrain == c, :], xval[yval == c, :], xtest))

        # train them
        accuracy = training(clfs, C)

        # classify them
        classify(clfs, ytest)

    else:
        raise Exception('Not a valid data set. Try either "random" or "MINIST"')






