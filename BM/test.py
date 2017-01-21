import scipy.io
import numpy as np

from sklearn.metrics import confusion_matrix
import itertools

import matplotlib.pyplot as plt
import seaborn as sns

from BMimproved import BoltzmannMachine


def training(clfs, yval):

    # train every classifier and get validation accuracy for each
    sig = []
    for c in clfs:
        c.mean_field()
        sig.append(c.val_p)
    sig = np.hstack(sig)
    pred = np.argmax(sig, axis=1)
    hits = np.array(pred == yval, dtype=np.int32)
    accuracy = np.mean(hits)

    print("Accuracy on validation test: " + str(accuracy))


def classify(clfs, ytest, noise):

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

    # fig = plt.figure()
    # for j in range(10):
    #     i = np.random.randint(0, len(ytest), 1)
    #     plt.subplot(2, 5, j+1)
    #     plt.imshow(xtest[i].reshape(28, 28).T)
    #     sns.set_style("whitegrid", {'axes.grid': False})
    #     plt.text(x=10.5, y=-3.0, s='true: ' + str(ytest[i][0, 0]),  color='blue')
    #     plt.text(x=10.5, y=1.0, s='pred: ' + str(pred[i][0, 0]),  color='red')
    #     plt.axis('off')
    # fig.savefig('predictions.png')
    #
    labels = [str(i) for i in range(10)]
    pred = list(itertools.chain(*pred))
    ytest = list(itertools.chain(*ytest))

    fig0 = plt.figure()
    ax = fig0.gca()
    cm = confusion_matrix(ytest, pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.set_style("whitegrid", {'axes.grid': False})
    axc = ax.matshow(cm, cmap=plt.cm.Blues)
    # plt.title('Confusion matrix')
    plt.colorbar(axc)

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.tight_layout()
    plt.xlabel('Predicted')
    plt.ylabel('True')

    fig0.savefig('confusion{}.png'.format(str(noise)), bbox_inches='tight')

    plt.show()

    print("Accuracy on the test set: " + str(accuracy))

    return accuracy


def add_noise(data, thrs):
    r, c = data.shape
    noise = np.random.rand(r, c)
    mask = np.array([noise < thrs], dtype=np.float32)

    return np.squeeze(np.abs(data - mask))


if __name__ == '__main__':

    print(''' Which exercise would you like to run? \n
                Type key name: \n
                * random : to classify a randomly generated data set with BM computing the exact solution \n
                * MNIST : to classify MNIST dataset with 10 different BMs using mean field estimation \n
          ''')

    opt = True
    while opt:

        data = str(raw_input())

        if data == 'random':

            # hyper-parameters
            N = 10 # num of neurons
            M = 1000 # num training samples
            T = 500 # num of sequential dynamic steps
            K = 200 # num of training iters
            eta = 0.001
            burn_in = 400

            # training data
            S = np.random.choice([1, -1], [M, N])

            # BM
            bm = BoltzmannMachine(S, K=K, T=T, eta=eta)
            # train BM
            bm.exact(burn_in)

            fig = plt.figure()
            plt.plot(bm.delta_w)
            plt.xlabel('iterations K=200')
            plt.ylabel('mean of abs change rate of w')
            plt.show()

            fig.savefig('change_w_theta.png')

            print('''What would you like to do now?\n
                     Type key name:\n
                     * random : to run this exercise again\n
                     * MNIST : to run BM on MNIST data set\n
                     * end : to exit''')

        elif data == 'MNIST':

            # todo: data augmentation skimage.trasnform rotation

            noiseLevel = np.arange(0.1, 0.6, 0.1)
            accuracy = []

            for n in noiseLevel:
                # hyper-parameters
                C = 10 # num of classes

                # read data sets
                mat = scipy.io.loadmat('mnistAll.mat')['mnist'][0][0]
                xtrain, xtest, ytrain, ytest = mat[0], mat[1], mat[2], mat[3]

                # prepare train data set
                ntrain = xtrain.shape[-1]

                xtrain = xtrain.T.reshape(ntrain, -1) / 255.
                xtrain_ = np.array(xtrain > 0.5, dtype=np.float32)
                if n > 0:
                    xtrain = add_noise(xtrain_, n)

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

                # plot input and noisy input
                fig = plt.figure()
                sns.set_style("whitegrid", {'axes.grid' : False})
                # _, ax = plt.subplot(2, 1, 1)
                # plt.imshow(xtrain_[105].reshape(28, 28).T)
                # ax.axis('off')
                # plt.xticks([])
                # plt.yticks([])
                # _, ax = plt.subplot(1, 1, 1)
                ax = fig.gca()
                plt.imshow(xtrain[90].reshape(28, 28).T)
                ax.axis('off')
                plt.xticks([])
                plt.yticks([])

                fig.savefig('noise{}.png'.format(str(n)), bbox_inches='tight')
                plt.show()

                # create classifiers
                clfs = []
                for c in range(C):
                    clfs.append(BoltzmannMachine(xtrain[ytrain == c, :], xval, xtest))

                # train them
                training(clfs, yval)

                # plot mean firing rate and linear response
                fig0 = plt.figure()
                ax = fig0.gca()
                sns.set_style("whitegrid", {'axes.grid' : False})
                # plt.subplot(2, 1, 1)
                plt.imshow(clfs[6].mi.reshape(28, 28).T)
                ax.axis('off')
                plt.xticks([])
                plt.yticks([])
                # plt.subplot(2, 1, 2)
                # plt.imshow(clfs[7].invXij)
                # plt.colorbar()
                fig0.savefig('meanFR{}.png'.format(str(n)), bbox_inches='tight')
                plt.show()

                # test them
                acc = classify(clfs, ytest, n)
                accuracy.append(acc)

            fig1 = plt.figure()
            labels = [str(i) for i in noiseLevel]
            plt.plot(accuracy)
            plt.xlabel('noise level')
            plt.ylabel('accuracy on test set')
            plt.xticks(accuracy, labels)
            plt.show()
            fig1.savefig('noise_vs_accuracy.png')

            print('''What would you like to do now?\n
                     Type key name:\n
                     * MNIST : to run this exercise again\n
                     * random : to run BM on a randomly generated data set\n
                     * end : to exit''')

        elif data == 'end':

            exit()

        else:

            print('Not a valid data set. Try either "random" or "MNIST" as written here.')
            print("Try again:")






