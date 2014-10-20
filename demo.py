import numpy as np
import math
import matplotlib.pyplot as plt
import os
import time
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
    avg_start = 250000
    alpha = .000001
    clfs = [(SGDClassifier(loss="log", learning_rate="bottou", alpha=alpha,
                           eta0=4., power_t=1.), [], [], "SGD"),
            (SGDClassifier(average=avg_start, loss="log",
                           learning_rate="bottou", alpha=alpha,
                           eta0=4., power_t=.75), [], [], "ASGD")]
    print("training", n_epochs, "epochs")
    total_times = {"SGD": 0, "ASGD": 0}
    for n in range(n_epochs):
        print("epoch:", n + 1)
        for i in range(10):
            print("    chunk:", i + 1)
            X = np.load(os.path.join("data", "data_batch" + str(i) + ".npy"))
            y = np.load(os.path.join("data", "labels_batch" +
                                     str(i) + ".npy")).ravel()

            for clf, loss, _, name in clfs:
                if i < 5:
                    t1 = time.time()
                    clf.partial_fit(X, y, classes=[-1, 1])
                    t2 = time.time()
                    print("time", name, t2 - t1)
                    total_times[name] += t2 - t1
                else:
                    pred = clf.decision_function(X)
                    loss.append(np.mean(list(map(log_loss, pred, y))))

        for clf, loss, total_loss, name in clfs:
            mean_loss = np.mean(loss)
            mean_loss += clf.alpha * np.dot(clf.coef_.ravel(), clf.coef_.ravel()) * .5
            total_loss.append(mean_loss)
            print(name, total_loss)
            loss[:] = []

    print("results:")
    for clf in total_times:
        print(clf, "total training time", total_times[clf])

    for clf, _, total_loss, name in clfs:
        print(name, total_loss)
        plt.plot(total_loss, label=name, marker=".")

    plt.xlabel('epoch')
    plt.ylabel('average cost')
    plt.legend(loc=0, prop={'size': 11})
    plt.show()

if __name__ == "__main__":
    main()
