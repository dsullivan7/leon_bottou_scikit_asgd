import numpy as np
import math
import matplotlib.pyplot as plt
import os
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
                           eta0=1., power_t=1.), [], [], "SGD"),
            (SGDClassifier(average=avg_start, loss="log",
                           learning_rate="bottou", alpha=alpha,
                           eta0=1., power_t=.75), [], [], "ASGD")]
    print("training", n_epochs, "epochs")
    for n in range(n_epochs):
        print("epoch:", n + 1)
        for i in range(10):
            print("    chunk:", i + 1)
            X = np.load(os.path.join("data", "data_batch" + str(i) + ".npy"))
            y = np.load(os.path.join("data", "labels_batch" +
                                     str(i) + ".npy")).ravel()

            for clf, loss, _, _ in clfs:
                if i < 5:
                    clf.partial_fit(X, y, classes=[-1, 1])
                else:
                    pred = clf.decision_function(X)
                    loss.append(np.mean(list(map(log_loss, pred, y))))

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

if __name__ == "__main__":
    main()
