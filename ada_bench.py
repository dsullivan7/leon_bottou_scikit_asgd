import numpy as np
import math
import matplotlib.pyplot as plt
import os
import time
from scipy.sparse import csr_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn import datasets
from sklearn.externals import joblib
mem = joblib.Memory("./cache")

@mem.cache
def get_data():
    data = datasets.load_svmlight_file("data/amazon7")
    return data[0], data[1]


def log_loss(p, y):
    z = p * y
    if z > 18:
        return math.exp(-z)
    if z < -18:
        return -z
    return math.log(1.0 + math.exp(-z))


def hinge_loss(p, y):
    z = p * y
    if z <= 1.0:
        return (1.0 - z)
    return 0.0

loss_func = log_loss

tuning_params = {"alpha": [.0000001, .00000001, .000000001, .0000000001], "eta0": [.001, .01, .1, 1., 10., 20., 30., 50., 100.]}

def grid_search():
    X = load_sparse_csr(os.path.join("data", "rcv1", "data_batch1.npz"))
    y = np.load(os.path.join("data", "rcv1", "labels_batch1.npy")).ravel()

    clfs = [
        # SGDClassifier(loss="log", learning_rate="optimal"),
        SGDClassifier(loss="log", learning_rate="adagrad", n_iter=20),
    ]
    for clf_i in clfs:
        clf = GridSearchCV(clf_i, tuning_params, cv=5, scoring='precision')
        clf.fit(X, y)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_estimator_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() / 2, params))
        print()

def main():
    n_epochs = 40
    alpha = .000000001
    clfs = [
        (SGDClassifier(loss="log", learning_rate="invscaling", alpha=alpha, eta0=30.),
         [], [], [], [], "invscaling"),
        # (SGDClassifier(loss="log", learning_rate="constant", alpha=alpha, eta0=.01),
        #  [], [], [], [], "constant"),
        (SGDClassifier(loss="log", learning_rate="adagrad", alpha=alpha, eta0=30.),
         [], [], [], [], "adadelta"),
        # (SGDClassifier(loss="log", learning_rate="adadelta", alpha=alpha, eta=30.),
        #  [], [], [], [], "adadelta"),
    ]
    print("training", n_epochs, "epochs")
    total_times = {"optimal": 0, "adagrad": 0, "constant": 0, "adadelta": 0, "invscaling": 0}
    # data = joblib.load("data/amazon7.pkl.tar")
    # s = time.time()
    # X, y = get_data()
    # e = time.time()
    # print(e - s)

    for n in range(n_epochs):
        print("epoch:", n + 1)
        for i in range(1, 10):
        # for i in range(1, 2):
            if i % 30 == 0:
                print("    train chunk:", i)

            X = load_sparse_csr(os.path.join("data", "rcv1", "data_batch" + str(i) + ".npz"))
            y = np.load(os.path.join("data", "rcv1", "labels_batch" +
                                     str(i) + ".npy")).ravel()

            for clf, loss, _, score_arr, _, name in clfs:
                if n % 2 == 0:
                    t1 = time.time()
                    clf.partial_fit(X, y, classes=[-1, 1])
                    t2 = time.time()
                    total_times[name] += t2 - t1
                else:
                    df = clf.decision_function(X)
                    loss.append(np.mean(list(map(loss_func, df, y))))
                    pred = clf.predict(X)
                    score_arr.append(1.0 - np.mean(pred == y))

        if n % 2 == 1:
            for i in range(1, 5):
                print("    test chunk:", i)
                X = load_sparse_csr(os.path.join("data", "rcv1_test", "data_batch" + str(i) + ".npz"))
                y = np.load(os.path.join("data", "rcv1_test", "labels_batch" +
                                         str(i) + ".npy")).ravel()

                for clf, loss, _, score_arr, _, _ in clfs:
                    pred = clf.predict(X)
                    score_arr.append(1.0 - np.mean(pred == y))

            for clf, loss, total_loss, score_arr, total_score, name in clfs:
                mean_score = np.mean(score_arr)
                total_score.append(mean_score)
                print(name, total_score)

                mean_loss = np.mean(loss)
                # mean_loss += clf.alpha * np.linalg.norm(clf.coef_) ** 2
                mean_loss += clf.alpha * np.dot(clf.coef_.ravel(), clf.coef_.ravel()) * .5
                total_loss.append(mean_loss)
                print(name, total_loss)
                loss[:] = []
                score_arr[:] = []

    print("results:")
    for clf in total_times:
        print(clf, "total training time", total_times[clf])

    for clf, _, total_loss, _, total_score, name in clfs:
        print(name, total_loss)
        plt.plot(total_loss, label=name, marker=".")

    plt.xlabel('epoch')
    plt.ylabel('average cost')
    plt.legend(loc=0, prop={'size': 11})
    plt.show()

    for clf, _, total_loss, _, total_score, name in clfs:
        print(name, total_score)
        plt.plot(total_score, label=name, marker=".")

    plt.xlabel('epoch')
    plt.ylabel('average error')
    plt.legend(loc=0, prop={'size': 11})
    plt.show()


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

if __name__ == "__main__":
    main()
