import numpy as np
import math
import matplotlib.pyplot as plt
import os
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier


def log_loss(p, y):
    z = p * y
    if z > 18:
        return math.exp(-z)
    if z < -18:
        return -z
    return math.log(1.0 + math.exp(-z))


def main():
    n_epochs = 20
    # avg_start = 385000
    avg_start = 781265
    alpha = .0000005
    clfs = [(SGDClassifier(loss="log", learning_rate="bottou", alpha=alpha,
                           eta0=2., power_t=1.), [], [], "SGD"),
            (SGDClassifier(average=avg_start, loss="log",
                           learning_rate="bottou", alpha=alpha,
                           eta0=2., power_t=.75), [], [], "ASGD")]
    print("training", n_epochs, "epochs")
    for n in range(n_epochs):
        print("epoch:", n + 1)
        for i in range(1, 157):
            if i % 30 == 0:
                print("    train chunk:", i)
            X = load_sparse_csr(os.path.join("data", "rcv1", "data_batch" + str(i) + ".npz"))
            y = np.load(os.path.join("data", "rcv1", "labels_batch" +
                                     str(i) + ".npy")).ravel()

            for clf, loss, _, _ in clfs:
                # if n % 2 == 0:
                clf.partial_fit(X, y, classes=[-1, 1])
                # else:
                #     pred = clf.decision_function(X)
                #     loss.append(np.mean(list(map(log_loss, pred, y))))

        for i in range(1, 5):
            print("    test chunk:", i)
            X = load_sparse_csr(os.path.join("data", "rcv1_test", "data_batch" + str(i) + ".npz"))
            y = np.load(os.path.join("data", "rcv1_test", "labels_batch" +
                                     str(i) + ".npy")).ravel()

            for clf, loss, _, _ in clfs:
                pred = clf.decision_function(X)
                loss.append(np.mean(list(map(log_loss, pred, y))))

        # if n % 2 == 1:
        for clf, loss, total_loss, name in clfs:
            mean_loss = np.mean(loss)
            mean_loss += clf.alpha * np.linalg.norm(clf.coef_) ** 2
            total_loss.append(mean_loss)
            print(name, total_loss)
            loss[:] = []

    print("results:")
    for clf, _, total_loss, name in clfs:
        print(name, total_loss)
        plt.plot(total_loss, label=name, marker=".")

    plt.xlabel('epoch')
    plt.ylabel('average cost')
    plt.legend(loc=0, prop={'size': 11})
    plt.show()


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

if __name__ == "__main__":
    main()
